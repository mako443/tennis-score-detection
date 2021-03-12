import torch
import numpy as np
import cv2
from utils import plot_bbox

from boxdetection.regression import ScoreBoxDetector
from dataloading.dataset import ScoreBoxCropDataset

W, H = 1280, 720

dataset = ScoreBoxCropDataset('./data/frames.json', './data/frames')

model = ScoreBoxDetector([16, 32, 64], [512, 256])
model.load_state_dict(torch.load('./checkpoints/best_regression_acc2.25.pth'))

for i in range(len(dataset)):
    data = dataset[i]
    boxes = model(torch.unsqueeze(data['images'], dim=0))
    boxes = boxes.detach().numpy()[0] * np.array((W,H,W,H))
    
    img = data['np_images']
    img = plot_bbox(img, data['boxes'].numpy(), color=(0,0,255))
    img = plot_bbox(img, boxes, color=(255,255,255))

    cv2.imshow("",img)
    cv2.waitKey()



