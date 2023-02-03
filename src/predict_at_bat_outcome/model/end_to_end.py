import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
    sys.path.insert(0, '.')
import logging
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from data.data_handler import Data
from train import train
from typing import Dict, Tuple
from utils import timer, set_log_file

@timer
def get_data(source: str) -> Tuple[Dict[str, DataLoader], Tuple[str]]:
    atbats = Data(source)
    atbats.clean()
    atbats.shuffle(seed=1)
    classes = tuple(atbats.data.result.unique())
    atbats.split((0.9, 0.05, 0.05))
    xy_dict = atbats.create_XY(
        x=['exit_velocity', 'launch_angle', 'pitch_velocity'],
        y='result',
        data=[atbats.train, atbats.dev, atbats.test]
        )
    atbats.normalize(xy_dict)
    dls = atbats.pytorch_prep(xy_dict)
    
    return dls, classes


def get_hidden_layers():
    return [
        nn.Linear(3, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    ]


def get_hyperparameters():
    return {
        'hidden_layers': get_hidden_layers(),
        'loss_func': nn.CrossEntropyLoss(),
        'optimizer': Adam,
        'lr': 0.001,
        'epochs': 10,
        'batch_size': 1500,
    }

@timer
def main(data_source):
    dls, classes = get_data(data_source)
    logging.info(f'Data source: {data_source}')
    hparams = get_hyperparameters()
    logging.info(f'Hyperparameters: ')
    for k, v in hparams.items():
        logging.info(f'\t{k}: {v}')

    model = train(dls['train'], dls['dev'], hparams)


if __name__ == '__main__':
    log_loc = 'training_logs'
    logging.basicConfig(filename=set_log_file(log_loc), level=logging.INFO)
    main('F:/baseball/active_player_abs/')