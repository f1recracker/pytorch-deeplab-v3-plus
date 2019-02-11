
''' Berkeley Deepdrive Segmentation Dataset loader '''

import os
import re
from PIL import Image
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
        return 1 # TODO fixme
        # return len(self.images)

    def __getitem__(self, key):
        image = Image.open(self.images[key])
        label = Image.open(self.labels[key]).convert(mode='L')
        if self.transforms:
            image, label = self.transforms(image, label)
        return image, label
