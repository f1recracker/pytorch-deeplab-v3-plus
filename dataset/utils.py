
import torch

def median_frequency_balance(dataset, num_classes=19, ignore_index=255, _eps=1e-5):
    '''
    For more details refer to Section 6.3.2 in
    https://arxiv.org/pdf/1411.4734.pdf
    '''
    frequency = torch.zeros(num_classes) + _eps
    for _, seg in dataset:
        for cid in torch.unique(seg):
            if cid == ignore_index:
                continue
            frequency[cid] += torch.sum(seg == cid)
    frequency /= torch.sum(frequency)
    return torch.median(frequency) / frequency
