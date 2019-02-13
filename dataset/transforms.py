
''' Dataset transformation pipeline '''

import random
import torch
import numpy as np
import torchvision.transforms.functional as tfunc
from PIL import Image

def transforms(img, seg, size=(360, 640), hflip=True, five_crop=True,
               tensor=True, normalize=True):
    ''' BDD transforms pipeline '''
    
    if hflip and random.random() < 0.5:
        img = tfunc.hflip(img)
        seg = tfunc.hflip(seg)

    if five_crop and random.random() < 0.5:
        i = random.randint(0, 4)
        img = tfunc.five_crop(img, (size[0] // 2, size[1] // 2))[i]
        seg = tfunc.five_crop(seg, (size[0] // 2, size[1] // 2))[i]

    img = tfunc.resize(img, size, interpolation=Image.NEAREST)
    seg = tfunc.resize(seg, size, interpolation=Image.NEAREST)

    if tensor:
        img = tfunc.to_tensor(img)
        seg = torch.LongTensor(np.array(seg))

    return img, seg
