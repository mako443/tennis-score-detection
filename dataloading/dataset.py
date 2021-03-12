import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import json
import random
import cv2
import os.path as osp
import numpy as np

from utils import plot_bbox

W, H = 1280, 720

class ScoreBoxDataset(Dataset):
    def __init__(self, path_json, path_frames):
        self.path_frames = path_frames
        self.transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])

        with open(path_json, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        data = self.data[idx]
        image = Image.open(osp.join(self.path_frames, data['id'] + '.png'))
        bbox = data['bbox']

        boxes = torch.tensor([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=torch.float).reshape((1,4))
        labels = torch.tensor([1], dtype=torch.long) # Only one class, 0 is background
        image_id = torch.tensor([idx], dtype=torch.long)
        area = torch.tensor([bbox[2]*bbox[3]], dtype=torch.float)
        iscrowd = torch.tensor([0], dtype=torch.uint8)

        image_transformed = self.transform(image)

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}
        return {
            'images': image_transformed,
            'targets': target,
            'raw_images': image
        }

    def __len__(self):
        return len(self.data)

    def collage_fn(data):
        batch = {}
        for key in data[0].keys():
            if key == 'images':
                batch[key] = torch.stack([data[i][key] for i in range(len(data))])
            elif key in ('targets', 'raw_images'):
                batch[key] = [data[i][key] for i in range(len(data))]
            else:
                raise Exception('Unexpected key:' + key)
        return batch

class ScoreBoxCropDataset(Dataset):
    def __init__(self, path_json, path_frames):
        self.path_frames = path_frames
        self.transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])

        with open(path_json, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        data = self.data[idx]
        img = Image.open(osp.join(self.path_frames, data['id'] + '.png'))
        bbox = data['bbox'] #Convert to [x0, y0, x1, x1]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        
        x0,y0,x1,y1 = bbox
        crop_w, crop_h = 512, 256
        if x1 < W/2 and y1 < H/2:
            x, y = 0, 0
        elif x1 < W/2 and y0 > H/2:
            x, y = 0, H - crop_h
        elif x0 > W/2 and y1 < H/2:
            x, y = W - crop_w, 0    
        elif x0 > W/2 and y0 > H/2:
            x, y = W - crop_w, H - crop_h
        else:
            raise Exception('BBox is not in one of the corners!')

        img = img.crop((x, y, x+crop_w, y+crop_h))                     
        bbox[0] -= x
        bbox[1] -= y
        bbox[2] -= x
        bbox[3] -= y

        return {
            'np_images': np.asarray(img).copy(),
            'images': self.transform(img),
            'boxes': torch.tensor(bbox, dtype=torch.float)
        }

    def __len__(self):
        return len(self.data)

class ScoreDetectionDataset(Dataset):
    def __init__(self, path_json, path_frames):
        self.path_frames = path_frames
        self.transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])])

        with open(path_json, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, idx):
        data = self.data[idx]
        img = Image.open(osp.join(self.path_frames, data['id'] + '.png'))
        bbox = data['bbox'] #Convert to [x0, y0, x1, x1]
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]

        x0, y0, x1, y1 = bbox
        x0 -= 5
        y0 -= 5
        x1 += 5
        y1 += 5
        img = img.crop((x0, y0, x1, y1))                    

        return {
            'np_images': np.asarray(img).copy(),
            'images': self.transform(img),
            'data': data
        }

    def __len__(self):
        return len(self.data)        

if __name__ == "__main__":
    dataset = ScoreBoxCropDataset('./data/frames.json', './data/frames')
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)
    data = dataset[18]
    batch = next(iter(dataloader))