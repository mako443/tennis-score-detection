import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models

import numpy as np
import cv2
from utils import calc_iou, plot_boxes, plot_metrics

from dataloading.dataset import ScoreBoxDataset

'''
Module to train a Faster-RCNN model for score box detection
=> Abandoned after a few trials because of unstable model conversion, R-CNN overhead seems unjustified.
'''

def train_epoch(model, dataloader, max_batches = 99):
    model.train()
    epoch_losses = []
    for i_batch, batch in enumerate(dataloader):
        if i_batch==max_batches:
            break

        optimizer.zero_grad()
        targets = [{k: v.to(device) for k,v in t.items()} for t in batch['targets']]
        losses = model(batch['images'].to(device), targets)

        loss = sum(losses.values())
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item() / len(losses))
    return np.mean(epoch_losses)

@torch.no_grad()
def val_epoch(model, dataloader, max_batches=99):
    model.eval()
    epoch_accs = []
    for i_batch, batch in enumerate(dataloader):
        if i_batch==max_batches:
            break
        preds = model(batch['images'].to(device))
        targets = batch['targets']
        for i in range(len(targets)):
            scores = preds[i]['scores']
            boxes = preds[i]['boxes']
            if len(scores) == 0:
                epoch_accs.append(0.0)
            else:
                predicted_box = boxes[torch.argmax(scores)].cpu().detach().numpy()
                gt_box = targets[i]['boxes'][0].cpu().detach().numpy()

                iou = calc_iou(predicted_box, gt_box)
                epoch_accs.append(iou)
    return np.mean(epoch_accs)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = ScoreBoxDataset('./data/frames.json', './data/frames')
dataloader = DataLoader(dataset, batch_size=4, collate_fn=ScoreBoxDataset.collage_fn)
data = dataset[0]
batch = next(iter(dataloader))

learning_rates = np.logspace(-3, -5, 3)
dict_loss = {lr: [] for lr in learning_rates}
dict_acc = {lr: [] for lr in learning_rates}

max_batches = 1

for lr in learning_rates:
    print(f'lr: {lr:0.6f}')
    # model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, num_classes=2, pretrained_backbone=True, trainable_backbone_layers=2)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2, pretrained_backbone=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer = optim.Adam(params, lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    for epoch in range(32):
        loss = train_epoch(model, dataloader, max_batches)
        acc = val_epoch(model, dataloader, max_batches)
        print(f'\r epoch {epoch:03.0f} loss {loss:0.2f} acc {acc:0.2f}', end='')
        dict_loss[lr].append(loss)
        dict_acc[lr].append(acc)

        scheduler.step()

    print('\n')

plot_name = 'plot.png'
metrics = {'train-loss': dict_loss, 'train-acc': dict_acc}
plot_metrics(metrics, plot_name, size=(8,5))
    
quit()

images, targets = batch['images'], batch['targets']
model.eval()
preds = model(images.cuda())
print(preds[0]['boxes'])
print(preds[0]['scores'])
img = plot_boxes(batch['raw_images'][0], preds[0]['boxes'], preds[0]['scores'])
cv2.imshow("", img); cv2.waitKey()