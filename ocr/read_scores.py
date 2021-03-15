import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models

import numpy as np
import cv2
from easydict import EasyDict
from utils import calc_iou, plot_boxes, plot_metrics
import time

from dataloading.dataset import ScoreServeDataset

def get_mlp(dims, add_batchnorm=False):
    if len(dims)<3:
        print('get_mlp(): less than 2 layers!')
    mlp = []
    for i in range(len(dims)-1):
        mlp.append(nn.Linear(dims[i], dims[i+1]))
        if i<len(dims)-2:
            mlp.append(nn.ReLU())
            if add_batchnorm:
                mlp.append(nn.BatchNorm1d(dims[i+1]))
    return nn.Sequential(*mlp)

class ScoreServePredictor(torch.nn.Module):
    def __init__(self, conv_dims, nc_points, nc_sets, nc_matches, nc_serve):
        super(ScoreServePredictor, self).__init__()

        self.conv1 = nn.Conv2d(3, conv_dims[0], (3,3), stride=(1, 2))
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(conv_dims[0], conv_dims[1], (3,3), stride=(1, 2))
        self.pool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(conv_dims[1], conv_dims[2], (3,3), stride=(1, 2))

        self.mlp_points = get_mlp([4*5*conv_dims[2], 512, 256, nc_points])
        self.mlp_sets = get_mlp([4*5*conv_dims[2], 512, 256, nc_sets])
        self.mlp_matches = get_mlp([4*5*conv_dims[2], 512, 256, nc_matches])
        self.mlp_serves = get_mlp([4*5*conv_dims[2], 512, 256, nc_serve])

    def forward(self, images):
        batch_size = len(images)

        x = images
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
     
        x = x.reshape((batch_size, -1))

        preds_points = self.mlp_points(x)
        preds_sets = self.mlp_sets(x)
        preds_matches = self.mlp_matches(x)
        preds_serves = self.mlp_serves(x)
        
        return EasyDict(
            points=preds_points,
            sets=preds_sets,
            matches=preds_matches,
            serves=preds_serves
        )

def train_epoch(model, dataloader):
    model.train()
    epoch_losses = []
    for i_batch, batch in enumerate(dataloader):
        optimizer.zero_grad()
        out = model(batch['images'].to(device))
        loss_points = criterion(out.points, batch['points'].to(device))
        loss_sets = criterion(out.sets, batch['sets'].to(device))
        loss_matches = criterion(out.matches, batch['matches'].to(device))
        loss_serves = criterion(out.serves, batch['serves'].to(device))

        loss = loss_points + loss_sets + loss_matches + loss_serves

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
    return np.mean(epoch_losses)     

@torch.no_grad()
def val_epoch(model, dataloader):
    model.eval()
    epoch_accs = []
    for i_batch, batch in enumerate(dataloader):
        batch_size = len(batch['points'])
        out = model(batch['images'].to(device))

        acc_points = torch.sum(torch.argmax(out.points, dim=-1).cpu() == batch['points']).item() / batch_size
        acc_sets = torch.sum(torch.argmax(out.sets, dim=-1).cpu() == batch['sets']).item() / batch_size
        acc_matches = torch.sum(torch.argmax(out.matches, dim=-1).cpu() == batch['matches']).item() / batch_size
        acc_serves = torch.sum(torch.argmax(out.serves, dim=-1).cpu() == batch['serves']).item() / batch_size

        epoch_accs.append(np.mean([acc_points, acc_sets, acc_matches, acc_serves]))
    return np.mean(epoch_accs)



if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = ScoreServeDataset('./data/frames.json', './data/frames')
    dataloader = DataLoader(dataset, batch_size=8, num_workers=1, shuffle=True)
    data = dataset[0]
    batch = next(iter(dataloader))    

    learning_rates = np.logspace(-2.5, -3.5, 5)[1:-1]
    dict_loss = {lr: [] for lr in learning_rates}
    dict_acc = {lr: [] for lr in learning_rates}    

    for lr in learning_rates:
        print(f'LR: {lr}')

        model = ScoreServePredictor([16, 32, 64],
            len(dataset.points_to_index), 
            len(dataset.sets_to_index),
            len(dataset.matches_to_index),
            len(dataset.serve_to_index)
        )
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(32):
            loss = train_epoch(model, dataloader)

            t0 = time.time()
            acc = val_epoch(model, dataloader)
            t1 = time.time()
            print(f'Time: {(t1-t0) / len(dataset)}')
            quit()


            print(f'\r epoch {epoch:03.0f} loss {loss:0.5f} acc {acc:0.2f}', end='')
            dict_loss[lr].append(loss)
            dict_acc[lr].append(acc)

        print()

    plot_name = 'plot_prediction.png'
    metrics = {'train-loss': dict_loss, 'train-acc': dict_acc}
    plot_metrics(metrics, plot_name, size=(8,5))    