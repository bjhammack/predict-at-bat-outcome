import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
    sys.path.insert(0, '.')
import logging
from model import NeuralNetwork
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
from data.data_handler import Data
from typing import Any, Dict
from utils import timer


@timer
def train(dls: Dict[str, DataLoader], hparams: Dict[str, Any]):
    model = NeuralNetwork(hparams['hidden_layers'])

    loss_func = hparams['loss_func']
    optimizer = hparams['optimizer'](model.parameters(), lr=hparams['lr'])

    epochs = hparams['epochs']
    batch_size = hparams['batch_size']

    best_vloss = 1_000_000
    pbar = tqdm(range(1, epochs+1))
    for epoch in pbar:
        model.train(True)
        avg_loss = train_epoch(epoch, model, dls['train'], loss_func, optimizer, batch_size, pbar)
        model.train(False)
        avg_vloss = validate_dev(model, loss_func, dls['dev'])
        logging.info(
            f'EPOCH {epoch}: Train loss: {avg_loss:.4f}; '
            f'Dev loss: {avg_vloss.item():.4f}'
            )
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

    return model


def train_epoch(epoch, model, train_dl, loss_func, optimizer, batch_size, pbar):
    running_loss = 0.
    last_loss = 0.
    correct_labels = 0
    total_labels = 0

    for i, data in enumerate(train_dl):
        inputs, labels = data
        # inputs_m, inputs_s = inputs.mean(), inputs.std()
        # inputs = (inputs - inputs_m) / inputs_s

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        # schedular.step()

        running_loss += loss.item()
        _, prediction = torch.max(outputs, 1)
        correct_labels += (prediction == labels).sum().item()
        total_labels += prediction.shape[0]
        acc = correct_labels / total_labels

        if i % batch_size == batch_size-1:
            last_loss = running_loss / batch_size
            pbar.set_description(f'Epoch {epoch}: batch {i+1}; loss: {last_loss:.4f}; acc: {acc:.2f}')
            running_loss = 0.

    return last_loss   


def validate_dev(model, loss_func, dev_dl):
    running_vloss = 0.0
    for i, vdata in enumerate(dev_dl):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_func(voutputs, vlabels)
        running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    
    return avg_vloss
