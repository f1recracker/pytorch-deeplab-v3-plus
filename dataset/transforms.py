
''' Dataset transformation pipeline '''

import random
import torch
import numpy as np
import torchvision.transforms.functional as tfunc
from PIL import Image

def transforms(img, seg, size=(360, 640), augment=True, hflip_prob=0.5,
               five_crop_prob=0.5, five_crop_scale=0.6,
               rotate_prob=0.5, max_rotate=30.0,
               tensor_output=True,
               normalize_mean=torch.Tensor([0.3518, 0.3932, 0.4011]),
               normalize_std=torch.Tensor([0.2363, 0.2494, 0.2611]),
               _ignore_index=255):
    ''' BDD transforms pipeline '''

    if augment and random.random() < hflip_prob:
        img = tfunc.hflip(img)
        seg = tfunc.hflip(seg)

    if augment and random.random() < five_crop_prob:
        i = random.randint(0, 4)
        resize = lambda x, scale: tuple(int(i * scale) for i in x)[::-1]
        img = tfunc.five_crop(img, resize(img.size, five_crop_scale))[i]
        seg = tfunc.five_crop(seg, resize(seg.size, five_crop_scale))[i]

    if augment and random.random() < rotate_prob:
        angle = random.randrange(-max_rotate, max_rotate)
        # mask to track rotation and ignore newly added pixels
        mask = Image.new('1', seg.size, (1,))

        img = tfunc.rotate(img, angle)
        seg = tfunc.rotate(seg, angle)
        mask = tfunc.rotate(mask, angle)

        white = Image.new('L', seg.size, (_ignore_index))
        seg = Image.composite(seg, white, mask=mask)

    img = tfunc.resize(img, size, interpolation=Image.NEAREST)
    seg = tfunc.resize(seg, size, interpolation=Image.NEAREST)

    if tensor_output:
        img = tfunc.to_tensor(img)
        img = (img - normalize_mean[:, None, None]) / normalize_std[:, None, None]
        seg = torch.LongTensor(np.array(seg))

    return img, seg
