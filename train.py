# pylint: disable=invalid-name,redefined-outer-name

import functools
import os
import pickle

import apex
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from model import DeepLab
from model.backbone import Xception
from dataset import BDDSegmentationDataset, transforms, median_frequency_balance
from metrics import pixel_accuracy, mean_iou

if __name__ == '__main__':

    amp_handle = apex.amp.init(enabled=False)

    if not os.path.exists('train'):
        os.mkdir('train')
        os.mkdir('train/checkpoints')

    from datetime import datetime
    time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    writer = SummaryWriter(log_dir=f'train/tensorboard/sess_{time_now}')

    batch_size = 8
    bdd_train = BDDSegmentationDataset('bdd100k', 'train', transforms=transforms)
    train_loader = torch.utils.data.DataLoader(
        bdd_train, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_transforms = functools.partial(transforms, hflip=False, five_crop=False)
    bdd_val = BDDSegmentationDataset('bdd100k', 'val', transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(
        bdd_val, batch_size=batch_size, num_workers=1, pin_memory=True)

    num_classes = 19
    model = DeepLab(Xception(output_stride=16), num_classes=num_classes)
    if torch.cuda.is_available():
        model = model.cuda()

    if not os.path.exists('train/class_weights.pkl'):
        class_weights = median_frequency_balance(bdd_train)
        pickle.dump(class_weights, open('train/class_weights.pkl', 'wb'))

    class_weights = pickle.load(open('train/class_weights.pkl', 'rb'))

    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-5)

    max_epochs = 500
    lr_update = lambda epoch: (1 - epoch / max_epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_update)

    # writer.add_graph(model, torch.rand(1, 3, 1280, 720), True)

    for epoch in range(1, max_epochs + 1):
        scheduler.step()
        writer.add_scalar('lr', scheduler.get_lr()[0], epoch)

        train_pix_acc = 0.0
        train_loss, train_mIoU = 0.0, 0.0
        for batch, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            y_pred = model(x)

            loss = criterion(y_pred, y)
            with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mIoU += mean_iou(y_pred, y)
            train_pix_acc += pixel_accuracy(y_pred, y)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Run validation loop
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
