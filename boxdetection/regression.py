import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models

import numpy as np
import cv2
from utils import calc_iou, plot_boxes, plot_metrics

from dataloading.dataset import ScoreBoxCropDataset

'''
TODO:
- MaxPool / Relu? (Check lecture)
- Random offsets as augmentation (as big as the space around box allows)
'''

W, H = 1280, 720

class ScoreBoxDetector(torch.nn.Module):
    def __init__(self, conv_dims, linear_dims):
        super(ScoreBoxDetector, self).__init__()

        self.conv1 = nn.Conv2d(3, conv_dims[0], (3,3), stride=2)
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(conv_dims[0], conv_dims[1], (3,3), stride=2)
        self.pool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(conv_dims[1], conv_dims[2], (3,3), stride=2)
        self.pool3 = nn.MaxPool2d((2,2))

        self.lin1 = nn.Linear(3*7*conv_dims[2], linear_dims[0])
        self.lin2 = nn.Linear(linear_dims[0], linear_dims[1])
        self.lin3 = nn.Linear(linear_dims[1], 4) #Outputs: [x0, y0, x1, y1], TODO: also predict score

    def forward(self, images):
        """Forward pass for score box regression

        Args:
            images (Tensor): [B,3,256,512] fixed size currently expected

        Returns:
            scores: scores
            boxes: boxes
        """
        batch_size = len(images)

        x = images
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)     
     
        x = x.reshape((batch_size, -1))
        # print(x.shape)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        x = F.relu(x)  

        boxes = x
        return boxes

def train_epoch(model, dataloader, max_batches=99):
    model.train()
    epoch_losses = []
    for i_batch, batch in enumerate(dataloader):
        if i_batch==max_batches:
            break

        optimizer.zero_grad()
        predictions = model(batch['images'].to(device))
        targets = batch['boxes'] / torch.tensor((W, H, W, H))
        loss = criterion(predictions, targets.to(device))

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
    return np.mean(epoch_losses)  

@torch.no_grad()
def val_epoch(model, dataloader):
    model.train()
    epoch_accs = []
    for i_batch, batch in enumerate(dataloader):

        optimizer.zero_grad()
        predictions = model(batch['images'].to(device))
        predictions = predictions.cpu().detach().numpy() * np.array((W,H,W,H))
        targets = batch['boxes'].cpu().detach().numpy()

        diff = predictions - targets
        acc = np.mean(np.abs(diff))

        epoch_accs.append(acc)
    return np.mean(epoch_accs)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = ScoreBoxCropDataset('./data/frames.json', './data/frames')
    dataloader = DataLoader(dataset, batch_size=8, num_workers=1, shuffle=True)  

    learning_rates = np.logspace(-3.5, -4.5, 3)     
    dict_loss = {lr: [] for lr in learning_rates}
    dict_acc = {lr: [] for lr in learning_rates}

    best_acc = np.inf
    best_model = None

    for lr in learning_rates:
        print(f'LR: {lr}')
        model = ScoreBoxDetector([16, 32, 64], [512, 256]).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()        
        for epoch in range(32):
            loss = train_epoch(model, dataloader, max_batches=99)
            acc = val_epoch(model, dataloader)
            print(f'\r epoch {epoch:03.0f} loss {loss:0.5f} acc {acc:0.2f}', end='')
            dict_loss[lr].append(loss)
            dict_acc[lr].append(acc)

        if acc < best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'./checkpoints/best_regression.pth')
            print(f'Saved model with acc {acc:0.2f}')
            best_model = model

        print()

    plot_name = 'plot_regression.png'
    metrics = {'train-loss': dict_loss, 'train-acc': dict_acc}
    plot_metrics(metrics, plot_name, size=(8,5))

    batch=next(iter(dataloader))
    pred = best_model(batch['images'].to(device))
    print(pred * torch.tensor((W,H,W,H), device=device))
    print(batch['boxes'])