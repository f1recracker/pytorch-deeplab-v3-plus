# pylint: disable=W0221,C0414,C0103

import functools
import os

# import apex
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from model import DeepLab
from model.backbone import Xception
from dataset import BDDSegmentationDataset, transforms

if __name__ == '__main__':

    # amp_handle = apex.amp.init(enabled=True)

    if not os.path.exists('train'):
        os.mkdir('train')
        os.mkdir('train/checkpoints')

    from datetime import datetime
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    writer = SummaryWriter(log_dir=f'train/tensorboard/sess_{time_now}')

    bdd_train = BDDSegmentationDataset('bdd100k', 'train', transforms=transforms)
    train_loader = torch.utils.data.DataLoader(
        bdd_train, batch_size=4, shuffle=True, num_workers=1, pin_memory=True)

    val_transforms = functools.partial(transforms, hflip=False, five_crop=False)
    bdd_val = BDDSegmentationDataset('bdd100k', 'val', transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(
        bdd_val, batch_size=4, num_workers=1, pin_memory=True)

    num_classes = 19
    model = DeepLab(Xception(output_stride=16), num_classes=num_classes)
    if torch.cuda.is_available():
        model = model.cuda()

    def init_weights(model):
        ''' Initializes weights using kaiming normal for conv and identity for batch norm '''
        for module in model.modules():
            if isinstance(module, torch.nn.modules.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif isinstance(module, torch.nn.modules.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0.0)

    init_weights(model)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=4e-5)

    max_epochs = 250
    lr_update = lambda epoch: (1 - epoch / max_epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_update)

    # writer.add_graph(model, torch.rand(1, 3, 1280, 720), True)

    def mean_iou(y_pred, y, eps=1e-8, logits_dim=1):
        ''' Evaluates mean IoU between prediction and ground truth '''
        num_classes = y_pred.shape[logits_dim]
        y_pred = torch.argmax(y_pred, dim=logits_dim)

        miou = 0.0
        for i in range(num_classes):
            intersect = torch.sum((y_pred == i) & (y == i)).float()
            union = torch.sum((y_pred == i) | (y == i)).float()
            miou += (intersect + eps) / (union + eps)
        return miou / num_classes

    def pixel_accuracy(y_pred, y, logits_dim=1):
        ''' Evaluates pixel accuracy between prediction and ground truth '''
        y_pred = torch.argmax(y_pred, dim=logits_dim)
        return torch.sum(y == y_pred).float() / y.nelement()

    for epoch in range(1, max_epochs + 1):
        scheduler.step()

        train_pix_acc = 0.0
        train_loss, train_mIoU = 0.0, 0.0
        for batch, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            y_pred = model(x)

            loss = criterion(y_pred, y)
            loss.backward()
            # TODO for mixed precision training
            # with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mIoU += mean_iou(y_pred, y)
            train_pix_acc += pixel_accuracy(y_pred, y)

            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

        val_pix_acc = 0.0
        val_loss, val_mIoU = 0.0, 0.0
        for val_batch, (x, y) in enumerate(val_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            with torch.no_grad():
                y_pred = model(x)
                loss = criterion(y_pred, y)

            val_loss += loss.item()
            val_mIoU += mean_iou(y_pred, y)
            val_pix_acc += pixel_accuracy(y_pred, y)

        writer.add_scalar('Train/loss', train_loss / len(train_loader), epoch)
        writer.add_scalar('Train/mIoU', train_mIoU / len(train_loader), epoch)
        writer.add_scalar('Train/accuracy', train_pix_acc / len(train_loader), epoch)

        writer.add_scalar('Validation/loss', val_loss / len(val_loader), epoch)
        writer.add_scalar('Validation/mIoU', val_mIoU / len(val_loader), epoch)
        writer.add_scalar('Validation/accuracy', val_pix_acc / len(val_loader), epoch)

        state = {}
        state['epoch'] = epoch
        state['model'] = model.state_dict()
        state['optimizer'] = optimizer.state_dict()
        torch.save(state, 'train/checkpoints/epoch-%d.pth' % epoch)
