from glob import glob
from math import ceil
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union


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
            source: Optional[str] = None,
            data: Optional[pd.DataFrame] = None
            ):
        '''
        Initialize the Data class by reading in the data.

        Args:
        source -- string; directory or CSV where raw data is stored.
        data -- pandas DataFrame; pre-collected data.
        '''
        self.data = self.train = self.dev = self.test = self.splits = self.source = None  # noqa: E501

        if data:
            self.data = data
            self.source = 'Pre-collected'
        elif source and not data:
            self.data = self.collect_data(source)
            self.source = source

    def collect_data(self, source: str) -> pd.DataFrame:
        '''
        Returns collected data from given source directory and all
        sub-directories.

        Args:
        source -- string; directory or CSV where data is stored.

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

    def normalize(self):
        return

    def create_XY(
            self,
            x: Union[str, List[str]],
            y: Union[str, List[str]],
            data: Union[pd.DataFrame, List[pd.DataFrame]] = None,
            ) -> Tuple[np.ndarray]:
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
        def onehot(df: pd.DataFrame) -> np.array:
            unique, inverse = np.unique(df.to_numpy(), return_inverse=True)
            onehot = np.eye(unique.shape[0])[inverse].transpose()
            return onehot

        if not data:
            data = self.data

        sets = (('X_train', 'Y_train'), ('X_dev', 'Y_dev'), ('X_test', 'Y_test'))
        xy_dict = {}
        if type(data) == list:
            for i, df in enumerate(data):
                if i > 2: break
                xy_dict[sets[i][0]] = df.loc[:, x].to_numpy().transpose()
                xy_dict[sets[i][1]] = onehot(df.loc[:, y])
        else:
            xy_dict['X'] = df.loc[:, x].to_numpy().transpose()
            xy_dict['Y'] = onehot(df.loc[:, y])
        
        return xy_dict