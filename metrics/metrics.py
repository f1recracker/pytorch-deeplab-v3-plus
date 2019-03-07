
import torch

def mean_iou(y_pred, y, logits_dim=1, ignore_index=255, eps=1e-8):
    ''' Evaluates mean IoU between prediction and ground truth '''
    y_pred = torch.argmax(y_pred, dim=logits_dim)
    classes = set(torch.unique(torch.cat((y_pred, y))))
    classes.discard(ignore_index)
    mask = (y != ignore_index)

    miou = 0.0
    for i in classes:
        intersect = torch.sum((y_pred[mask] == i) & (y[mask] == i)).float()
        union = torch.sum((y_pred[mask] == i) | (y[mask] == i)).float()
        miou += (intersect + eps) / (union + eps)
    return (miou + eps) / (len(classes) + eps)

def pixel_accuracy(y_pred, y, logits_dim=1, ignore_index=255):
    ''' Evaluates pixel accuracy between prediction and ground truth '''
    y_pred = torch.argmax(y_pred, dim=logits_dim)
    mask = (y != ignore_index)
    return torch.sum(y[mask] == y_pred[mask]).float() / torch.sum(mask).float()
