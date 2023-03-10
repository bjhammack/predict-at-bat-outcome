import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
    sys.path.insert(0, '.')
import logging
from model import NeuralNetwork, clear_checkpoints, save_checkpoint, save_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from typing import Any, Dict
from utils import timer


@timer
def train(
        dls: Dict[str, DataLoader],
        hparams: Dict[str, Any],
        writer: SummaryWriter,
        save_path,
        checkpoint_path,
        device,
        ):
    clear_checkpoints('/'.join(checkpoint_path.split('/')[:-1])+'/')
    
    model = NeuralNetwork(hparams['hidden_layers'])
    model.to(device)
    
    train_sample = iter(dls['train'])
    input, labels = next(train_sample)
    writer.add_graph(model, input)

    loss_func = hparams['loss_func']
    optimizer = hparams['optimizer'](
        model.parameters(),
        lr=hparams['lr'],
        weight_decay=hparams['weight_decay']
        )
    scheduler = hparams['scheduler'](
        optimizer,
        max_lr=hparams['max_lr'],
        steps_per_epoch = int(len(dls['train'])),
        epochs=hparams['epochs'],
        anneal_strategy='linear',
        )

    epochs = hparams['epochs']
    best_vloss = 1_000_000
    pbar = tqdm(range(1, epochs+1))
    for epoch in pbar:
        model.train(True)
        loss, train_acc, model, optimizer, scheduler = train_epoch(
            epoch,
            model,
            dls['train'],
            loss_func,
            optimizer,
            scheduler,
            pbar,
            device,
            )
        model.train(False)
        vloss, dev_acc = validate_dev(model, loss_func, dls['dev'], device)
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Loss/dev', vloss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/dev', dev_acc, epoch)
        logging.info(
            f'EPOCH {epoch}: Train loss: {loss:.4f}; '
            f'Dev loss: {vloss.item():.4f}'
            )
        if vloss < best_vloss:
            best_vloss = vloss
        if epoch % 25 == 0:
            check_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            save_checkpoint(checkpoint_path, check_dict)

    save_model(model, save_path)
    return model


def train_epoch(epoch, model, train_dl, loss_func, optimizer, scheduler, pbar, device):
    running_loss = 0.
    last_loss = 0.
    correct_labels = 0
    total_labels = 0
    for i, data in enumerate(train_dl):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, prediction = torch.max(outputs, 1)
        correct_labels += (prediction == labels).sum().item()
        total_labels += prediction.shape[0]
        acc = correct_labels / total_labels

        if i % len(train_dl) == len(train_dl)-1:
            last_loss = running_loss / len(train_dl)
            pbar.set_description(f'Epoch {epoch}: batch {i+1}; loss: {last_loss:.4f}; acc: {acc:.2f}')
            running_loss = 0.

    return last_loss, acc, model, optimizer, scheduler


def validate_dev(model, loss_func, dev_dl, device):
    running_loss = 0.0
    total_labels = 0
    correct_labels = 0
    for i, vdata in enumerate(dev_dl):
        inputs, labels = vdata
        outputs = model(inputs)

        loss = loss_func(outputs, labels)
        running_loss += loss

        _, prediction = torch.max(outputs, 1)
        correct_labels += (prediction == labels).sum().item()
        total_labels += prediction.shape[0]
    avg_loss = running_loss / (i + 1)
    acc = correct_labels / total_labels
    
    return avg_loss, acc
