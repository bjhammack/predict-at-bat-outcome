import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
    sys.path.insert(0, '.')
import logging
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.data_handler import Data
from test import evaluate
from train import train
from typing import Dict, Tuple
from utils import timer, set_dated_file


@timer
def get_data(
        sources: str,
        split: Tuple[float],
        batch: int,
        device,
        ) -> Tuple[Dict[str, DataLoader], Tuple[str]]:
    atbats = Data(sources)

    logging.info(f'Data source: {sources}')

    atbats.clean()
    atbats.shuffle(seed=1)
    classes = atbats.get_labels()
    atbats.split(split)
    xy_dict = atbats.create_XY(
        x=[
            'launch_speed',
            'la_z',
            'la_xy',
            'ft/s',
            'effective_speed',
            'release_spin_rate',
            'zone',
            'stand',
            'p_throws',
            'if_fielding_alignment',
            'of_fielding_alignment',
        ],
        y='events',
        data=[atbats.train, atbats.dev, atbats.test]
        )
    xy_dict = atbats.normalize(xy_dict)

    logging.info(f'Data normalization mean: {atbats.norm_mean}; std: {atbats.norm_std}')

    dls = atbats.pytorch_prep(xy_dict, batch, device)
    
    return dls, classes


def get_hidden_layers():
    return [
        nn.Linear(11, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    ]


def get_hyperparameters():
    return {
        'split': (0.8, 0.1, 0.1),
        'hidden_layers': get_hidden_layers(),
        'loss_func': nn.CrossEntropyLoss(),
        'optimizer': Adam,
        'scheduler': OneCycleLR,
        'lr': 3e-3,
        'epochs': 1000,
        'batch_size': 1000,
        'weight_decay': 1e-4,
        'max_lr': 1e-2,
    }


@timer
def main(data_sources, save_path, checkpoint_path, version):
    logging.info(f'Model will be saved at: \'{save_path}\'')
    logging.info(f'Checkpoints will be saved at: \'{checkpoint_path}\'')

    device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    hparams = get_hyperparameters()
    dls, classes = get_data(data_sources, hparams['split'], hparams['batch_size'], device)

    for dl in ('train', 'dev', 'test'):
        if dl == 'train':
            logging.info(f'{dl} size: ~{len(dls[dl])} * {hparams["batch_size"]}')
        else:
            logging.info(f'{dl} size: {len(dls[dl])}')
    logging.info(f'Classes: {classes}')
    logging.info(f'Hyperparameters:')
    for k, v in hparams.items():
        if type(v) == list:
            logging.info(f'\t{k}:')
            for i in v:
                logging.info(f'\t\t{i}')
        else:
            logging.info(f'\t{k}: {v}')

    writer = SummaryWriter(comment=f'_{version}')
    model = train(dls, hparams, writer, save_path, checkpoint_path, device)
    eval = evaluate(model, hparams['loss_func'], dls['test'])
    writer.close()

    logging.info(f"TEST: loss: {eval['loss']:.4f}; accuracy: {eval['accuracy']:.2f}")

    class_acc = {i: [0, 0] for i in classes}
    for label, pred in list(zip(eval['labels'], eval['predictions'])):
        class_acc[classes[label]][1] += 1
        if pred == label:
            class_acc[classes[label]][0] += 1

    logging.info('Label perforamce:')
    for k, v in class_acc.items():
        logging.info(f'\t{k}: {v[0]} of {v[1]} correct; {(v[0]/v[1])*100:.2f}%')


if __name__ == '__main__':
    vers = 'v5.1'
    pre = f'model-{vers}'
    log_loc = 'logs'
    save_loc = 'saved_models'
    check_loc = 'checkpoints'
    check_suf = '_<EPOCH>.pt'

    logging.basicConfig(filename=set_dated_file(log_loc, pre, '.log'), level=logging.INFO)
    main(
        data_sources = [
            'F:/baseball/statcast_pitch/',
            'F:/baseball/statcast_running/',
            ],
        save_path = set_dated_file(save_loc, pre, '.pt'),
        checkpoint_path = set_dated_file(check_loc, pre, check_suf),
        version=vers,
        )
