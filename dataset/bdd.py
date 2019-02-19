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
    def palette(seg):
        color_map = {
            0: torch.FloatTensor((0xab, 0x47, 0xbc)) / 0xFF, # Road
            1: torch.FloatTensor((0x6a, 0x1b, 0x9a)) / 0xFF, # Sidewalk
            2: torch.FloatTensor((0x60, 0x7d, 0x8b)) / 0xFF, # Building
            3: torch.FloatTensor((0x8d, 0x6e, 0x63)) / 0xFF, # Wall
            4: torch.FloatTensor((0xcf, 0xd8, 0xdc)) / 0xFF, # Fence
            5: torch.FloatTensor((0xf5, 0xf5, 0xf5)) / 0xFF, # Pole
            6: torch.FloatTensor((0xff, 0xc4, 0x00)) / 0xFF, # Light
            7: torch.FloatTensor((0xff, 0xff, 0x00)) / 0xFF, # Sign
            8: torch.FloatTensor((0x00, 0xc8, 0x53)) / 0xFF, # Vegetation
            9: torch.FloatTensor((0xb2, 0xff, 0x59)) / 0xFF, # Terrain
            10: torch.FloatTensor((0xb2, 0xdf, 0xdb)) / 0xFF, # Sky
            11: torch.FloatTensor((0xff, 0x17, 0x44)) / 0xFF, # Person
            12: torch.FloatTensor((0xff, 0x8a, 0x80)) / 0xFF, # Rider
            13: torch.FloatTensor((0x3f, 0x51, 0xb5)) / 0xFF, # Car
            14: torch.FloatTensor((0x1a, 0x23, 0x7e)) / 0xFF, # Truck
            15: torch.FloatTensor((0xe8, 0xea, 0xf6)) / 0xFF, # Bus
            16: torch.FloatTensor((0x29, 0xb6, 0xf6)) / 0xFF, # Train
            17: torch.FloatTensor((0x00, 0x69, 0x5c)) / 0xFF, # Motorcycle
            18: torch.FloatTensor((0x4d, 0xb6, 0xac)) / 0xFF, # Bicycle
        }
        seg_color = torch.zeros_like(seg).repeat(3, 1, 1).float()
        for c_id, color in color_map.items():
            seg_color += (seg == c_id).float()[None, :, :] * color[:, None, None]
        return seg_color

    def __getitem__(self, key):
        image = Image.open(self.images[key])
        label = Image.open(self.labels[key])
        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label
