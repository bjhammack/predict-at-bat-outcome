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