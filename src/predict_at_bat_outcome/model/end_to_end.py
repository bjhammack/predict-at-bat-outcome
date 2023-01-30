import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
    sys.path.insert(0, '.')

from torch import nn
from torch.optim import Adam
from data.data_handler import Data
from train import train


def main():
    abs = Data('F:/baseball/active_player_abs/')
    abs.clean()
    abs.shuffle(seed=1)
    classes = list(abs.data.result.unique())
    abs.split((0.9, 0.05, 0.05))
    xy_dict = abs.create_XY(
        x=['exit_velocity', 'launch_angle', 'pitch_velocity'],
        y='result',
        data=[abs.train, abs.dev, abs.test]
        )
    dls = abs.pytorch_prep(xy_dict)

    hidden_layers = [
        nn.Linear(3, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
    ]
    hparams = {
        'hidden_layers': hidden_layers,
        'loss_func': nn.CrossEntropyLoss(),
        'optimizer': Adam,
        'lr': 0.001,
        'epochs': 1,
        'batch_size': 10000,
    }

    model = train(dls['train'], dls['dev'], hparams)


if __name__ == '__main__':
    main()