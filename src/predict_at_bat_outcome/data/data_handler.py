from glob import glob
from math import ceil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union


class Data:
    '''
    Class that will contain the data for modeling and manages all functions in
    regards to manipulating and augmenting the data.

    Attributes:
    data -- pandas DataFrame; the data in its current state of transformation.
    train -- pandas DataFrame; the training set of the data.
    dev -- pandas DataFrame; the dev set of the data.
    test -- pandas DataFrame; the test set of the data.
    split -- tuple of size (float, float, float); contains the train, dev, and
    test split sizes.
    source -- string; path to the original source of the data.
    '''
    def __init__(
            self,
            sources: Optional[Union[str, List[str]]] = None,
            on: Optional[Union[str, List[str]]] = None,
            data: Optional[pd.DataFrame] = None
            ):
        '''
        Initialize the Data class by reading in the data.

        Args:
        source -- string; directory or CSV where raw data is stored.
        data -- pandas DataFrame; pre-collected data.
        '''
        self.data = self.train = self.dev = self.test = self.splits = self.sources = self.norm_mean = self.norm_std = None  # noqa: E501

        if type(data) == pd.DataFrame:
            self.data = data
            self.sources = ['Pre-collected',]
        elif sources and not data:
            if type(sources) is str:
                sources = [sources,]
            self.data = self.collect_data(sources[0])
            if 'game_date' in self.data.columns:
                self.data['season'] = pd.to_datetime(self.data['game_date']).dt.year  # needs to be refactored
            self.sources = [sources[0],]
            if len(sources) > 1:
                sources_ons = list(zip(sources[1:], [on]*len(sources[1:])))
                for source, on in sources_ons:
                    self.sources.append(source)
                    next_source = self.collect_data(source)
                    next_source['batter'] = next_source['mlb_id'].str.split('-').str[-1].astype(int)
                    self.data = self.data.merge(
                        next_source,
                        on=on,
                        how='left',
                        )

    def collect_data(self, source: str) -> pd.DataFrame:
        '''
        Returns collected data from given source directory and all
        sub-directories.

        Args:
        source -- string; directory or CSV of the data.

        Returns:
        data -- pandas DataFrame; all CSV files from source aggregated together.
        '''
        if source[-4:] == '.csv':
            csv_files = glob(f'{source}')
        else:
            csv_files = glob(f'{source}/**/*.csv', recursive=True)

        if not csv_files:
            raise ValueError(
                f'No CSV files found at "{source}". '
                'Check that your path is correct.')

        dfs = []
        for csv in csv_files:
            dfs.append(pd.read_csv(csv))

        df_cols = set(dfs[0].columns.tolist())
        for i, df in enumerate(dfs):
            df_comp_cols = set(df.columns.tolist())
            if df_comp_cols == df_cols:
                continue
            raise pd.errors.DataError(
                f'CSV 1 and CSV {i+1}\'s columns do not match. All columns '
                f'must match to be properly concatenated.\nCSV 1: {df_cols}'
                f'\nCSV {i+1}: {df_comp_cols}')

        data = pd.concat(dfs, ignore_index=True)

        return data

    def clean(self):
        self._drop_missing()
        self._keep_results()
        self._move_results()
        self._redistribute_results()
        self._add_la_xy()
        self._factorize_strings(
            ['stand','p_throws','if_fielding_alignment','of_fielding_alignment']
            )
        self.data = self.data.fillna(0.)

    def split(self, split: Tuple[float, float, float]):
        self.splits = split
        if len(split) != 3 or split[0]+split[1]+split[2] != 1:
            raise ValueError(
                'Split error. Given split needs to be len = 3 and add up to 1.'
                )
        
        data_len = len(self.data)
        self.data = self.data.reset_index(drop=True)
        
        train_len = ceil(split[0] * data_len)
        dev_len = ceil(split[1] * data_len)
        test_len = ceil(split[2] * data_len)

        if (train_len + dev_len + test_len) != data_len:
            train_len = (data_len - (dev_len + test_len))

        self.train = self.data.iloc[:train_len].reset_index(drop=True)
        self.dev = self.data.iloc[train_len:(train_len + dev_len)].reset_index(drop=True)
        self.test = self.data.iloc[(train_len + dev_len):].reset_index(drop=True)

    def shuffle(self, seed: int = np.random.randint(1, 1e+6)):
        '''
        Sets self.data to a shuffled version of itself.

        Args:
        seed - int; seed to be used to creat perm
        '''
        data_len = len(self.data)
        perm = np.random.RandomState(seed=seed).permutation(data_len)
        self.data = self.data.iloc[perm].reset_index(drop=True)

    def create_XY(
            self,
            x: Union[str, List[str]],
            y: str,
            data: Union[pd.DataFrame, List[pd.DataFrame]] = None,
            ) -> Dict[str, np.ndarray]:
        '''
        Returns the X and Y data for a given DataFrame. If list of DF given, 
        first is train, second is dev, third is test, any others are ignored.

        Args:
        x -- string or list of strings; specifies columns to be X in DataFrame
        y -- string or list of strings; specifies columns to be Y in DataFrame
        data -- pd.DataFrame or list; data to be used to create X and Y
            if no data specified, self.data is used

        return:
        xy_dict -- dictionary of numpy arrays; number of arrays depends on `data`
        '''
        def map_results(df):
            map = {label: i for i, label in enumerate(Data.get_labels())}
            df = df.replace(map)
            return df

        if not data:
            data = self.data

        sets = (('X_train', 'Y_train'), ('X_dev', 'Y_dev'), ('X_test', 'Y_test'))
        xy_dict = {}
        if type(data) == list:
            for i, df in enumerate(data):
                if i > 2: break
                xy_dict[sets[i][0]] = df.loc[:, x].to_numpy()
                xy_dict[sets[i][1]] = map_results(df.loc[:, [y]]).to_numpy().squeeze()
        else:
            xy_dict['X'] = data.loc[:, x].to_numpy()
            xy_dict['Y'] = map_results(data.loc[:, [y]]).to_numpy().squeeze()
        
        return xy_dict

    def normalize(self, xy_dict) -> Dict[str, np.array]:
        if not self.norm_mean and self.norm_std:
            self.norm_mean = np.mean(xy_dict['X_train'])
            self.norm_std = np.std(xy_dict['X_train'])
        if 'X_train' in xy_dict.keys():
            for dataset in ('X_train', 'X_dev', 'X_test'):
                xy_dict[dataset] = (xy_dict[dataset] - self.norm_mean) / self.norm_std
        else:
            xy_dict['X'] = (xy_dict['X'] - self.norm_mean) / self.norm_std
        return xy_dict

    def pytorch_prep(self, xy_dict, batch_size, device):
        tensors = {
            k: torch.from_numpy(v).to(torch.float).to(device) for k, v in xy_dict.items()
            }
        if 'X_train' in xy_dict.keys():
            for y in ('Y_train', 'Y_dev', 'Y_test'):
                tensors[y] = tensors[y].to(torch.long)

            train_dl = DataLoader(
                TensorDataset(tensors['X_train'], tensors['Y_train']),
                batch_size = batch_size
                )
            dev_dl = DataLoader(TensorDataset(tensors['X_dev'], tensors['Y_dev']))
            test_dl = DataLoader(TensorDataset(tensors['X_test'], tensors['Y_test']))

            return {'train': train_dl, 'dev': dev_dl, 'test': test_dl}
        else:
            tensors['Y'] = tensors['Y'].to(torch.long)
            dl = DataLoader(
                TensorDataset(tensors['X'], tensors['Y']),
                batch_size = batch_size
                )
            return dl

    def _drop_missing(self):
        self.data = self.data.loc[~self.data.mlb_id.isnull()]

    def _keep_results(self):
        keep = ['field_out', 'double', 'single', 'double_play', 'sac_fly',
                'fielders_choice', 'grounded_into_double_play', 'force_out',
                'triple', 'home_run', 'field_error','fielders_choice_out',
                'triple_play', 'sac_bunt']
        self.data = self.data.loc[self.data.events.isin(keep)]

    def _move_results(self):
        field_out = ['fielders_choice', 'triple_play','sac_bunt',
            'fielders_choice_out', 'double_play', 'field_error',
            'force_out', 'grounded_into_double_play', 'sac_fly']
        non_hr_xbh = ['double', 'triple']

        self.data.loc[self.data.events.isin(field_out), ['events']] = 'field_out'
        self.data.loc[self.data.events.isin(non_hr_xbh), ['events']] = 'non_hr_xbh'

    def _redistribute_results(self):
        row_ceil = 70_000
        field_outs = self.data.loc[self.data.events.eq('field_out'), 'events'].count()
        reduced_field_outs = field_outs - row_ceil
        singles = self.data.loc[self.data.events.eq('single'), 'events'].count()
        reduced_singles = singles - row_ceil

        self.data = self.data.drop(self.data[self.data['events'].eq('field_out')].sample(reduced_field_outs).index)
        self.data = self.data.drop(self.data[self.data['events'].eq('single')].sample(reduced_singles).index)

    def _add_la_xy(self):
        self.data.loc[:, 'la_xy'] = np.tan((self.data.loc[:, 'hc_x'] - 128.) / (208. - self.data.loc[:, 'hc_y'])) * 180. / np.pi * 0.75
        self.data = self.data.rename(columns={'launch_angle': 'la_z'})

    def _factorize_strings(self, cols: List[str]):
        for col in cols:
            self.data[col] = pd.factorize(self.data[col])[0] + 1

    @staticmethod
    def get_labels():
        return ['field_out', 'single', 'non_hr_xbh', 'home_run']