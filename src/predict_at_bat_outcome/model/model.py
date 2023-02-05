import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Union


class NeuralNetwork(nn.Module):
    def __init__(self, hidden_layers: List[Union[nn.Linear, nn.ReLU]]):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(*hidden_layers, nn.Softmax(dim=1))

    def forward(self, x):
        logits = self.layers(x)
        return logits


def clear_checkpoints(path):
	for file in os.listdir(path):
		os.remove(path+file)


def save_checkpoint(file, check_dict):
    file = file.replace('<EPOCH>', str(check_dict['epoch']))
    torch.save(check_dict, file)


def load_checkpoint(file, hparams, train=True):
	model = NeuralNetwork(hparams['hidden_layers'])
	optimizer = hparams['optimizer'](
        model.parameters(),
        lr=hparams['lr'],
        weight_decay=hparams['weight_decay']
        )

	checkpoint = torch.load(file)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	if train:
		model.train()
	else:
		model.eval()

	return (model, optimizer, loss, epoch)


def save_model(model, file):
	torch.save(model.state_dict(), file)


def load_model(file):
	model = torch.load(file)
	model.eval()
	return model