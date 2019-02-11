
import torch.nn as nn

class BackboneModule(nn.Module):
    ''' Base class for all DeepLab feature extractor backbones '''

    def __init__(self, output_stride, out_channels, low_out_channels):
        super().__init__()
        self.output_stride = output_stride
        self.out_channels = out_channels
        self.low_out_channels = low_out_channels

    def forward(self, x):
        raise NotImplementedError
