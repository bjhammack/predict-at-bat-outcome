from src.predict_at_bat_outcome.data import data_handler
import numpy as np
from numpy.testing import assert_array_equal
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


@pytest.fixture(scope='class')
def split_data() -> Dict[str, pd.DataFrame]:
    '''
    Returns dict of <> pd.DataFrames to validate TestSplittingData.
    '''
    test_train1 = pd.DataFrame({'a': [i for i in range(1, 81)]})
    test_dev1 = pd.DataFrame({'a': [i for i in range(81, 91)]})
    test_test1 = pd.DataFrame({'a': [i for i in range(91, 101)]})

    test_train2 = pd.DataFrame({'a': [i for i in range(1, 95)]})
    test_dev2 = pd.DataFrame({'a': [i for i in range(95, 98)]})
    test_test2 = pd.DataFrame({'a': [i for i in range(98, 101)]})

    test_train3 = pd.DataFrame({'a': [i for i in range(1, 76)]})
    test_dev3 = pd.DataFrame({'a': [i for i in range(76, 81)]})
    test_test3 = pd.DataFrame({'a': [i for i in range(81, 101)]})
    
    return {'train1': test_train1, 'dev1': test_dev1, 'test1': test_test1,
            'train2': test_train2, 'dev2': test_dev2, 'test2': test_test2,
            'train3': test_train3, 'dev3': test_dev3, 'test3': test_test3,
    }


@pytest.fixture(scope='class')
def xy_data() -> Dict[str, pd.DataFrame]:
    data = pd.DataFrame(
        {'a': [i for i in range(1, 101)],
        'b': list(np.random.RandomState(seed=10).choice((1,2), 100)),
        'c': [i for i in range(101, 201)],}
        )
    X1 = np.array([[i, i+100] for i in range(1, 101)])
    Y1 = np.array([[1,0] if i == 1 else [0,1] for i in data['b']])
    X_train = X1[:70]
    X_dev = X1[70:85]
    X_test = X1[85:]
    Y_train = Y1[:70]
    Y_dev = Y1[70:85]
    Y_test = Y1[85:]

    return_dict = {
        'data': data,
        'X1': X1, 'Y1': Y1,
        'X_train': X_train, 'Y_train': Y_train,
        'X_dev': X_dev, 'Y_dev': Y_dev,
        'X_test': X_test, 'Y_test': Y_test,
    }

    return return_dict


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


class TestSplittingData:
    '''
    Suite of tests to test the splitting of data by data_handler.py.
    '''
    def test_split_80_10_10(self, data_class, split_data):
        '''
        Tests if Data.split() can split data correctly with 80/10/10 split.
        '''
        data_class.data = pd.DataFrame({'a': [i for i in range(1, 101)]})
        data_class.split((0.8, 0.1, 0.1))

        assert_frame_equal(data_class.train, split_data['train1'])
        assert_frame_equal(data_class.dev, split_data['dev1'])
        assert_frame_equal(data_class.test, split_data['test1'])

    def test_split_95_25_25(self, data_class, split_data):
        '''
        Tests if Data.split() can split data correctly with 95/2.5/2.5 split.
        '''
        data_class.data = pd.DataFrame({'a': [i for i in range(1, 101)]})
        data_class.split((0.95, 0.025, 0.025))

        assert_frame_equal(data_class.train, split_data['train2'])
        assert_frame_equal(data_class.dev, split_data['dev2'])
        assert_frame_equal(data_class.test, split_data['test2'])

    def test_split_75_05_20(self, data_class, split_data):
        '''
        Tests if Data.split() can split data correctly with 75/05/20 split.
        '''
        data_class.data = pd.DataFrame({'a': [i for i in range(1, 101)]})
        data_class.split((0.75, 0.05, 0.20))

        assert_frame_equal(data_class.train, split_data['train3'])
        assert_frame_equal(data_class.dev, split_data['dev3'])
        assert_frame_equal(data_class.test, split_data['test3'])

    def test_split_too_big(self, data_class):
        '''
        Tests is Data.split() catches splits > sum == 1.0.
        '''
        data_class.data = pd.DataFrame({'a': [i for i in range(1, 101)]})
        with pytest.raises(ValueError, match='Split error.'):
            data_class.split((0.8, 0.1, 0.11))

    def test_split_too_small(self, data_class):
        '''
        Tests is Data.split() catches splits < sum == 1.0.
        '''
        data_class.data = pd.DataFrame({'a': [i for i in range(1, 101)]})
        with pytest.raises(ValueError, match='Split error.'):
            data_class.split((0.8, 0.09, 0.10))

    def test_split_too_long(self, data_class):
        '''
        Tests is Data.split() catches splits len > 3.
        '''
        data_class.data = pd.DataFrame({'a': [i for i in range(1, 101)]})
        with pytest.raises(ValueError, match='Split error.'):
            data_class.split((0.8, 0.1, 0.09, 0.01))

    def test_split_too_short(self, data_class):
        '''
        Tests is Data.split() catches splits len < 3.
        '''
        data_class.data = pd.DataFrame({'a': [i for i in range(1, 101)]})
        with pytest.raises(ValueError, match='Split error.'):
            data_class.split((0.9, 0.1,))


class TestShufflingData:
    def test_shuffle_with_seed(self, data_class):
        test_df = pd.DataFrame({'a': [9, 3, 6, 7, 4, 2, 1, 8, 5, 10]})
        data_class.data = pd.DataFrame({'a': [i for i in range(1, 11)]})
        data_class.shuffle(seed=10)

        assert_frame_equal(data_class.data, test_df)


class TestCreatingXY:
    def test_single_xy_shapes(self, data_class, xy_data):
        data_class.data = xy_data['data']
        XY_dict = data_class.create_XY(x=['a','c'], y='b')

        assert_array_equal(XY_dict['X'].shape, xy_data['X1'].shape)
        assert_array_equal(XY_dict['Y'].shape, xy_data['Y1'].shape)

    def test_single_xy(self, data_class, xy_data):
        data_class.data = xy_data['data']
        XY_dict = data_class.create_XY(x=['a', 'c'], y='b')

        assert_array_equal(XY_dict['X'], xy_data['X1'])
        assert_array_equal(XY_dict['Y'], xy_data['Y1'])

    def test_full_set_xy_shapes(self, data_class, xy_data):
        data_class.data = xy_data['data']
        data_class.shuffle()
        data_class.split((0.7, 0.15, 0.15))
        XY_dict = data_class.create_XY(
            x=['a', 'c'],
            y='b',
            data=[data_class.train, data_class.dev, data_class.test])

        assert_array_equal(XY_dict['X_train'].shape, xy_data['X_train'].shape)
        assert_array_equal(XY_dict['Y_train'].shape, xy_data['Y_train'].shape)
        assert_array_equal(XY_dict['X_dev'].shape, xy_data['X_dev'].shape)
        assert_array_equal(XY_dict['Y_dev'].shape, xy_data['Y_dev'].shape)
        assert_array_equal(XY_dict['X_test'].shape, xy_data['X_test'].shape)
        assert_array_equal(XY_dict['Y_test'].shape, xy_data['Y_test'].shape)

    def test_full_set_xy(self, data_class, xy_data):
        data_class.data = xy_data['data']
        # data_class.shuffle()
        data_class.split((0.7, 0.15, 0.15))
        XY_dict = data_class.create_XY(
            x=['a', 'c'],
            y='b',
            data=[data_class.train, data_class.dev, data_class.test])

        assert_array_equal(XY_dict['X_train'], xy_data['X_train'])
        assert_array_equal(XY_dict['Y_train'], xy_data['Y_train'])
        assert_array_equal(XY_dict['X_dev'], xy_data['X_dev'])
        assert_array_equal(XY_dict['Y_dev'], xy_data['Y_dev'])
        assert_array_equal(XY_dict['X_test'], xy_data['X_test'])
        assert_array_equal(XY_dict['Y_test'], xy_data['Y_test'])
