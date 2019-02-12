
''' Dataset transformation pipeline '''

import random
import torchvision.transforms.functional as tfunc

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

    img = tfunc.resize(img, size, interpolation=PIL.Image.NEAREST)
    seg = tfunc.resize(seg, size, interpolation=PIL.Image.NEAREST)
    # seg = tfunc.to_grayscale(seg)

    if tensor:
        img = tfunc.to_tensor(img)
        seg = tfunc.to_tensor(seg).squeeze().long()

    # if normalize:
    #     img = tfunc.normalize(img,
    #                           mean=(0.36350803, 0.36781886, 0.37393862),
    #                           std=(0.26235075, 0.24659232, 0.24531917))

    return img, seg
