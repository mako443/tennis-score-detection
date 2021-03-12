import torch
import numpy as np
import cv2
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(metrics, save_path, show_plot=False, size=(16,10), y_top_limit=None):
    rows = int(np.round(np.sqrt(len(metrics))))
    cols = int(np.ceil(len(metrics)/rows))

    fig = plt.figure()
    fig.set_size_inches(*size)

    for i, key in enumerate(metrics.keys()):
        plt.subplot(rows, cols, i+1)
        for k in metrics[key].keys():
            l = metrics[key][k]
            line, = plt.plot(l)
            line.set_label(f'{k:0.6}')
        plt.title(key)
        plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
        if y_top_limit is not None:
            plt.gca().set_ylim(top=y_top_limit)
        plt.legend()

    if show_plot:
        plt.show()
    else:
        plt.savefig(save_path)
        print(f'\n Plot saved as {save_path} \n')

def calc_iou(box0, box1):
    """IoU score calculation

    Args:
        box0 (1d-array): Box as [x0, y0, x1, y1]
        box1 (1d-array): Box as [x0, y0, x1, y1]
    """
    assert len(box0)==len(box1)==4

    if box1[0]>box0[2] or box1[1]>box0[3] or box0[0]>box1[2] or box0[1]>box1[3]:
        return 0.0

    inter = np.hstack((np.maximum(box0[0:2], box1[0:2]), np.minimum(box0[2:4], box1[2:4])))
    inter_area = (inter[2]-inter[0]) * (inter[3]-inter[1])
    area0 = (box0[2]-box0[0]) * (box0[3]-box0[1])
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])

    union_area = area0 + area1 - inter_area

    iou = inter_area / union_area

    if iou>1.0 or iou<0.0:
        print(iou, box0, box1)
    
    return iou

def plot_boxes(image, boxes, scores, min_score=0.1):
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().detach().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().detach().numpy()        
    if not isinstance(image, np.ndarray): #PIL image
        image = np.asarray(image)
    for box, score in zip(boxes, scores):
        if score<min_score:
            continue
        box = np.int0(box)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255,255,255), thickness=2)
        cv2.putText(image, f'{score:0.2f}', (box[0]+10, box[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), thickness=2)
    return image

def plot_bbox(img, box, color=(255,255,255)):
    if not isinstance(img, np.ndarray): #PIL image
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    box = np.int0(box)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness=2)
    return img