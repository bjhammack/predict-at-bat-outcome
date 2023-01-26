from glob import glob
import pandas as pd
from typing import Optional


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
        self.data = self.train = self.dev = self.test = self.split = self.source = None  # noqa: E501

        if data:
            self.data = data
            self.source = 'Pre-collected'
        elif source and not data:
            self.collect_data(source)

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
