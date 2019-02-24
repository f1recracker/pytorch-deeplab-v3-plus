''' Berkeley Deepdrive Segmentation Dataset loader '''

import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset

def listdir(path, filter_=re.compile(r'.*')):
    ''' Enumerates full paths of files in a directory matching a filter '''
    return [os.path.join(path, f) for f in os.listdir(path) if filter_.match(f)]

class BDDSegmentationDataset(Dataset):
    ''' Dataset loader for Berkeley Deepdrive Segmentation dataset '''

    def __init__(self, path, split, transforms=None):
        assert split in ['train', 'val', 'test'], 'split must be one of: {train, val, test}'
        image_re = re.compile(r'(.*)\.jpg')
        label_re = re.compile(r'(.*)_train_id\.png')
        images = sorted(listdir(os.path.join(path, 'seg/images', split), image_re))
        labels = sorted(listdir(os.path.join(path, 'seg/labels', split), label_re))
        for (image, label) in zip(images, labels):
            assert (image_re.match(os.path.basename(image)).group(1) ==
                    label_re.match(os.path.basename(label)).group(1))
        self.images, self.labels = images, labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    @staticmethod
    def palette(seg_labels):
        color_map = {
            0: torch.FloatTensor((128, 67, 125)), # Road
            1: torch.FloatTensor((247, 48, 227)), # Sidewalk
            2: torch.FloatTensor((72, 72, 72)), # Building
            3: torch.FloatTensor((101, 103, 153)), # Wall
            4: torch.FloatTensor((190, 151, 152)), # Fence
            5: torch.FloatTensor((152, 152, 152)), # Pole
            6: torch.FloatTensor((254, 167, 56)), # Light
            7: torch.FloatTensor((221, 217, 55)), # Sign
            8: torch.FloatTensor((106, 140, 51)), # Vegetation
            9: torch.FloatTensor((146, 250, 157)), # Terrain
            10: torch.FloatTensor((65, 130, 176)), # Sky
            11: torch.FloatTensor((224, 20, 64)), # Person
            12: torch.FloatTensor((255, 0, 25)), # Rider
            13: torch.FloatTensor((0, 22, 138)), # Car
            14: torch.FloatTensor((0, 11, 70)), # Truck
            15: torch.FloatTensor((0, 63, 98)), # Bus
            16: torch.FloatTensor((0, 82, 99)), # Train
            17: torch.FloatTensor((0, 36, 224)), # Motorcycle
            18: torch.FloatTensor((121, 17, 38)), # Bicycle
        }
        seg_colors = torch.zeros_like(seg_labels).repeat(3, 1, 1).float()
        for c_id, color in color_map.items():
            seg_colors += ((seg_labels == c_id).float()[None, :, :] * color[:, None, None]) / 255.0
        return seg_colors

    def __getitem__(self, key):
        image = Image.open(self.images[key])
        label = Image.open(self.labels[key])
        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label
