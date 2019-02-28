
import torch

from model import DeepLab
from model.backbone import Xception

# state = torch.load('epoch-158.pth', map_location='cpu')

# model = DeepLab(Xception(output_stride=16), num_classes=19)
# model.load_state_dict(state['model'])

# import random
from dataset import BDDSegmentationDataset, transforms
# import torchvision.transforms.functional as tfunc

dataset = BDDSegmentationDataset('bdd100k', 'train', transforms)
x, y = dataset[0]

print(torch.unique(y))
# y_pred = model.forward(tfunc.to_tensor(x).unsqueeze(0)).detach()
# print(y_pred.shape)

# y_pred = torch.argmax(y_pred, dim=1).squeeze()
# y_pred, _ = torch.broadcast_tensors(y_pred, x)

# print(torch.unique(y_pred))

# x = tfunc.to_pil_image(x)
# x.show()

# y = tfunc.to_pil_image(y)
# y.show()

# y_pred = tfunc.to_pil_image(y_pred)
# y_pred.show()
