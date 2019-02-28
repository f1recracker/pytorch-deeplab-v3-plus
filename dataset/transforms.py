
''' Dataset transformation pipeline '''

import random
import torch
import numpy as np
import torchvision.transforms.functional as tfunc
from PIL import Image

def transforms(img, seg, size=(360, 640), hflip=True, five_crop=True,
               rotate=True, tensor=True, normalize=True):
    ''' BDD transforms pipeline '''

    if hflip and random.random() < 0.5:
        img = tfunc.hflip(img)
        seg = tfunc.hflip(seg)

    if five_crop and random.random() < 0.5:
        i = random.randint(0, 4)
        resize = lambda x, scale: tuple(int(i * scale) for i in x)[::-1]
        img = tfunc.five_crop(img, resize(img.size, 0.8))[i]
        seg = tfunc.five_crop(seg, resize(seg.size, 0.8))[i]

    if rotate and random.random() < 0.5:
        angle = random.randrange(-30, 30)
        mask = Image.new('1', seg.size, (1,))

        img = tfunc.rotate(img, angle)
        seg = tfunc.rotate(seg, angle)
        mask = tfunc.rotate(mask, angle)

        white = Image.new('L', seg.size, (0xFF))
        seg = Image.composite(seg, white, mask=mask)

    img = tfunc.resize(img, size, interpolation=Image.NEAREST)
    seg = tfunc.resize(seg, size, interpolation=Image.NEAREST)

    if tensor:
        img = tfunc.to_tensor(img)
        seg = torch.LongTensor(np.array(seg))

    return img, seg
