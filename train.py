# pylint: disable=W0221,C0414,C0103

import functools
import math
import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from model import DeepLab, Xception
from dataset import BDDSegmentationDataset

if __name__ == '__main__':

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    def transforms(img, seg, size=(1280, 720), hflip=True, five_crop=True):
        ''' BDD transforms pipeline '''
        import random
        import torchvision.transforms.functional as tfunc

        if hflip and random.random() < 0.5:
            img = tfunc.hflip(img)
            seg = tfunc.hflip(seg)

        if five_crop and random.random() < 0.5:
            i = random.randint(0, 4)
            img = tfunc.five_crop(img, (img.size[0] // 2, img.size[1] // 2))[i]
            seg = tfunc.five_crop(seg, (seg.size[0] // 2, seg.size[1] // 2))[i]

        img = tfunc.resize(img, size)
        seg = tfunc.resize(seg, size)
        seg = tfunc.to_grayscale(seg)

        img = tfunc.to_tensor(img)
        seg = tfunc.to_tensor(seg).long()
        return img, seg

    bdd_train = BDDSegmentationDataset('bdd100k', 'train', transforms=transforms)
    train_loader = torch.utils.data.DataLoader(
        bdd_train, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    val_transforms = functools.partial(transforms, hflip=False, five_crop=False)
    bdd_val = BDDSegmentationDataset('bdd100k', 'val', transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(
        bdd_val, batch_size=1, num_workers=4)

    num_classes = 19
    model = DeepLab(Xception(output_stride=16), num_classes=num_classes)
    if torch.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    lr_init = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=4e-5)

    max_epochs = 100000
    lr_update = lambda epoch: lr_init * math.pow(1 - epoch / max_epochs, 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_update)

    writer = SummaryWriter()
    # writer.add_graph(model, torch.rand(1, 3, 1280, 720), True)

    def mean_iou(y_pred, y, eps=1e-6):
        ''' Evaluates mean IoU between prediction and gt '''
        _, num_classes = y_pred.shape
        y_pred = torch.argmax(y_pred, dim=1)

        miou = 0.0
        for i in range(num_classes):
            intersect = torch.sum((y_pred == i) * (y == i) > 0).float()
            union = torch.sum((y_pred == i) + (y == i) > 0).float()
            miou += intersect / (union + eps)
        return miou / num_classes

    for epoch in range(1, max_epochs + 1):
        scheduler.step()

        train_loss, train_mIoU = 0.0, 0.0
        for batch, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.view(-1, num_classes), y.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mIoU += mean_iou(y_pred.view(-1, num_classes), y.view(-1))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        val_loss, val_mIoU = 0.0, 0.0
        for val_batch, (x, y) in enumerate(val_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred.view(-1, num_classes), y.view(-1))

                val_mIoU += mean_iou(y_pred.view(-1, num_classes), y.view(-1))
                val_loss += loss.item()

        writer.add_scalar('train/loss', train_loss / len(train_loader.dataset), epoch)
        writer.add_scalar('train/mIoU', train_mIoU / len(train_loader.dataset), epoch)
        writer.add_scalar('val/loss', val_loss / len(val_loader.dataset), epoch)
        writer.add_scalar('val/mIoU', val_mIoU / len(val_loader.dataset), epoch)

        state = {}
        state['epoch'] = epoch
        state['model'] = model.state_dict()
        state['optimizer'] = optimizer.state_dict()
        torch.save(state, 'checkpoints/epoch-%d.pth' % epoch)
