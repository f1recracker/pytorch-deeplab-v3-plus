''' Berkeley Deepdrive Segmentation Dataset loader '''

import os
import re

from PIL import Image
import torch
from torch.utils.data import Dataset

from dataset.utils import listdir

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

    def __getitem__(self, key):
        image = Image.open(self.images[key])
        label = Image.open(self.labels[key])
        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label


def bdd_palette(labels):
    ''' Applies a color palette to either a single label
        tensor or a batch of tensors '''
    assert len(labels.shape) in [2, 3], 'Invalid labels shape'

    # pylint: disable=bad-whitespace
    color_map = torch.Tensor([
        [128,  67, 125], # Road
        [247,  48, 227], # Sidewalk
        [ 72,  72,  72], # Building
        [101, 103, 153], # Wall
        [190, 151, 152], # Fence
        [152, 152, 152], # Pole
        [254, 167,  56], # Light
        [221, 217,  55], # Sign
        [106, 140,  51], # Vegetation
        [146, 250, 157], # Terrain
        [ 65, 130, 176], # Sky
        [224,  20,  64], # Person
        [255,   0,  25], # Rider
        [  0,  22, 138], # Car
        [  0,  11,  70], # Truck
        [  0,  63,  98], # Bus
        [  0,  82,  99], # Train
        [  0,  36, 224], # Motorcycle
        [121,  17,  38], # Bicycle
        [  0,   0,   0]  # Other
    ]).to(labels.device) / 255.0

    batched_input = True
    if len(labels.shape) == 2:
        batched_input = False
        labels = torch.unsqueeze(labels, 0)

    # Convert ignore index to label 20
    labels = torch.clamp(labels, 0, 20 - 1).long()

    n, h, w = labels.shape
    labels_one_hot = torch.zeros(n, 20, h, w).to(labels.device)
    labels_one_hot.scatter_(1, torch.unsqueeze(labels, 1), 1)

    color_labels = torch.einsum('nlhw,lc->nchw', labels_one_hot, color_map)

    if not batched_input:
        color_labels = torch.squeeze(color_labels, 0)

    return color_labels
