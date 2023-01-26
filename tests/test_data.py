from src.predict_at_bat_outcome.data.data_handler import Data
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest


@pytest.fixture(scope='class')
def read_data():
    data_class = Data()
    test_data1 = pd.read_csv('tests/data/validation_data/full_dummy1.csv')
    test_data2 = pd.read_csv('tests/data/validation_data/full_dummy2.csv')
    
    return {
        'Data': data_class,
        'test_data1': test_data1,
        'test_data2': test_data2
        }


class TestReadingData:
    def test_good_path_single_csv(self, read_data):
        df = read_data['Data'].collect_data('tests/data/dummy_data1/test_data1.csv')
        assert_frame_equal(df, read_data['test_data1'])

    def test_good_path_directory(self, read_data):
        df = read_data['Data'].collect_data('tests/data/dummy_data1')
        assert_frame_equal(df, read_data['test_data1'])

    def test_bad_path(self, read_data):
        with pytest.raises(ValueError, match='No CSV files'):
            read_data['Data'].collect_data('tests/data/doesnt_exist')

    def test_empty_path(self, read_data):
        with pytest.raises(ValueError, match='No CSV files'):
            read_data['Data'].collect_data('tests/data/dummy_data4')

    def test_sub_directories(self, read_data):
        df = read_data['Data'].collect_data('tests/data/dummy_data2/')
        assert_frame_equal(df, read_data['test_data2'])

    def test_bad_concat(self, read_data):
        with pytest.raises(pd.errors.DataError, match='All columns must match'):
            read_data['Data'].collect_data('tests/data/dummy_data3')