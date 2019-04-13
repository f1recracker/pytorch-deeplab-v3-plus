
import os
import re

import torch

def listdir(path, filter_=re.compile(r'.*')):
    ''' Enumerates full paths of files in a directory matching a filter '''
    return [os.path.join(path, f) for f in os.listdir(path) if filter_.match(f)]


def median_frequency_balance(dataset, num_classes=19, ignore_index=255, _eps=1e-5):
    '''
    For more details refer to Section 6.3.2 in
    https://arxiv.org/pdf/1411.4734.pdf
    '''
    frequency = torch.zeros(num_classes) + _eps
    for _, seg in dataset:
        for cid in torch.unique(seg):
            if cid == ignore_index:
                continue
            frequency[cid] += torch.sum(seg == cid)
    frequency /= torch.sum(frequency)
    return torch.median(frequency) / frequency


def mean_std(dataset):
    ''' Returns the channel means and standard deviations
        of the images in the dataset '''
    mean, std = 0.0, 0.0
    for image, _ in dataset:
        mean += image.mean(dim=(1, 2)) # CHW -> C
        std += image.view((3, -1)).std(dim=1) ** 2
    return mean / len(dataset), (std / len(dataset)) ** 0.5
