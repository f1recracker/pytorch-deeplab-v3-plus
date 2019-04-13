
import os

import cv2
import numpy as np
import torch

from model import DeepLab
from model.backbone import Xception
from dataset import BDDSegmentationDataset, transforms, bdd_palette
from metrics import pixel_accuracy, mean_iou

from dataset import listdir

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    bdd_val = BDDSegmentationDataset('bdd100k', 'val', transforms=transforms)
    val_loader = torch.utils.data.DataLoader(
        bdd_val, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    num_classes = 19
    model = DeepLab(Xception(output_stride=16), num_classes=num_classes)

    while True:

        latest_run = sorted(listdir('train'), reverse=True)[0] + '/checkpoints'
        latest_epoch = sorted(listdir(latest_run), reverse=True)[0]
        state = torch.load(latest_epoch, map_location='cpu')
        model.load_state_dict(state['model'])

        miou, pix_acc = [], []
        for batch, (x, y) in enumerate(val_loader):
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)

            # Metrics
            miou.append(mean_iou(y_pred, y, num_classes))
            pix_acc.append(pixel_accuracy(y_pred, y, num_classes))

            print("mean_iou", miou[-1], "pixel_acc", pix_acc[-1], end="\r")

            y, y_pred = bdd_palette(y), bdd_palette(y_pred)

            to_img = lambda t: np.moveaxis(t.cpu().numpy(), 1, -1)[0] # NCHW -> HWC
            x, y, y_pred = to_img(x), to_img(y), to_img(y_pred)
            
            mean = np.array([[[0.3518, 0.3932, 0.4011]]])
            std = np.array([[[0.2363, 0.2494, 0.2611]]])
            x = (x * std + mean).astype('float32')

            cv2.imshow('x', cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
            cv2.imshow('y', cv2.cvtColor(y, cv2.COLOR_BGR2RGB))
            cv2.imshow('y_pred', cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB))
            cv2.imshow('x + y', cv2.cvtColor((x + y) / 2, cv2.COLOR_BGR2RGB))
            cv2.imshow('x + y_pred', cv2.cvtColor((x + y_pred) / 2, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        print("-- Batch stats --")
        print("miou", sum(miou) / len(miou), "acc", sum(pix_acc) / len(pix_acc))
