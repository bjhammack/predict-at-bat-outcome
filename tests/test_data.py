from src.predict_at_bat_outcome.data import data_handler
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from typing import Dict


@pytest.fixture(scope='module')
def data_class() -> data_handler.Data:
    '''
    Returns Data class from data_handler to be used during a module's testing.
    '''
    return data_handler.Data()


@pytest.fixture(scope='class')
def read_data() -> Dict[str, pd.DataFrame]:
    '''
    Returns dict of two pd.DataFrames to validate the TestReadingData tests.
    '''
    test_data1 = pd.read_csv('tests/data/validation_data/full_dummy1.csv')
    test_data2 = pd.read_csv('tests/data/validation_data/full_dummy2.csv')
    
    return {'test_data1': test_data1, 'test_data2': test_data2}


class TestReadingData:
    '''
    Suite of tests to test the reading in of data by data_handler.py.
    '''
    def test_good_path_single_csv(self, data_class, read_data):
        '''
        Tests if collect_data() can gather data from single DF.
        '''
        df = data_class.collect_data('tests/data/dummy_data1/test_data1.csv')
        assert_frame_equal(df, read_data['test_data1'])

    def test_good_path_directory(self, data_class, read_data):
        '''
        Tests if collect_data() can gather data from a directory.
        '''
        df = data_class.collect_data('tests/data/dummy_data1')
        assert_frame_equal(df, read_data['test_data1'])

    def test_bad_path(self, data_class):
        '''
        Tests if collect_data() raises the correct error w/ no CSV files.
        '''
        with pytest.raises(ValueError, match='No CSV files'):
            data_class.collect_data('tests/data/doesnt_exist')

    def test_empty_path(self, data_class):
        '''
        Tests if collect_data() raises the correct error for an empty dir.
        '''
        with pytest.raises(ValueError, match='No CSV files'):
            data_class.collect_data('tests/data/dummy_data4')

    def test_sub_directories(self, data_class, read_data):
        '''
        Tests if collect_data() can gather data from multiple sub-directories.
        '''
        df = data_class.collect_data('tests/data/dummy_data2/')
        assert_frame_equal(df, read_data['test_data2'])

    def test_bad_concat(self, data_class):
        '''
        Tests if collect_data() raises the correct error for mismatching DFs.
        '''
        with pytest.raises(pd.errors.DataError, match='All columns must'):
            data_class.collect_data('tests/data/dummy_data3')
