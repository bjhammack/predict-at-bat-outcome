import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
    sys.path.insert(0, '.')
import logging
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from data.data_handler import Data
from test import evaluate
from train import train
from typing import Dict, Tuple
from utils import timer, set_dated_file

@timer
def get_data(
        source: str,
        split: Tuple[float]
        ) -> Tuple[Dict[str, DataLoader], Tuple[str]]:
    atbats = Data(source)
    logging.info(f'Data source: {source}')
    atbats.clean()
    atbats.shuffle(seed=1)
    atbats.data = atbats.data.iloc[:5000]
    
    classes = tuple(atbats.data.result.unique())
    atbats.split(split)
    logging.info(f'{atbats.test.result.value_counts()}')
    xy_dict = atbats.create_XY(
        x=['exit_velocity', 'launch_angle', 'pitch_velocity'],
        y='result',
        data=[atbats.train, atbats.dev, atbats.test]
        )
    atbats.normalize(xy_dict)
    logging.info(f'Data normalization mean: {atbats.norm_mean}; std: {atbats.norm_std}')
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
        'split': (0.9, 0.05, 0.05),
        'hidden_layers': get_hidden_layers(),
        'loss_func': nn.CrossEntropyLoss(),
        'optimizer': Adam,
        'lr': 0.001,
        'epochs': 1,
        'batch_size': 1500,
    }


@timer
def main(data_source, save_path, checkpoint_path):
    logging.info(f'Model will be saved at: \'{save_path}\'')
    logging.info(f'Checkpoints will be saved at: \'{checkpoint_path}\'')
    hparams = get_hyperparameters()
    dls, classes = get_data(data_source, hparams['split'])
    for dl in ('train', 'dev', 'test'):
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

    model = train(dls, hparams)
    eval = evaluate(model, hparams['loss_func'], dls['test'])
    logging.info(f"TEST: loss: {eval['loss']:.4f}; accuracy: {eval['accuracy']:.2f}")

    class_acc = {i: [0, 0] for i in classes}
    for label, pred in list(zip(eval['labels'], eval['predictions'])):
        class_acc[classes[label]][1] += 1
        if pred == label:
            class_acc[classes[label]][0] += 1
    logging.info(f'Test labels correct/total: {class_acc}')


if __name__ == '__main__':
    vers = 'v0.1'
    pre = f'model-{vers}'
    log_loc = 'training_logs'
    save_loc = 'saved_models'
    check_loc = 'training_checkpoints'
    check_suf = '_<EPOCH>.model'

    logging.basicConfig(filename=set_dated_file(log_loc, pre, '.log'), level=logging.INFO)
    main(
        data_source = 'F:/baseball/active_player_abs/',
        save_path = set_dated_file(save_loc, pre, '.model'),
        checkpoint_path = set_dated_file(check_loc, pre, check_suf),
        )
