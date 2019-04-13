# pylint: disable=arguments-differ, too-many-arguments

''' Extensions to standard torch.nn primitives '''

import torch
import torch.nn as nn

class Identity(nn.Module):
    ''' Identity transformation operation '''

    def forward(self, x):
        return x


class SeparableConv2d(nn.Module):
    ''' Depthwise separable 2D convolution '''

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True,
                 _depthwise_conv=nn.Conv2d, _pointwise_conv=nn.Conv2d):
        super().__init__()
        self.depthwise_conv = _depthwise_conv(
            in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, stride=stride, padding=padding,
            dilation=dilation, bias=bias)
        self.pointwise_conv = _pointwise_conv(
            in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class Conv2dLayer(nn.Module):
    ''' 2D convolution with batch norm and ReLU '''

    def __init__(self, in_channels, out_channels, kernel_size,
                 groups=1, stride=1, padding=0, dilation=1, bias=True,
                 batchnorm_opts={'eps': 1e-3, 'momentum': 3e-4}):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      groups=groups, stride=stride, padding=padding,
                      dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, **batchnorm_opts),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class SeparableConv2dLayer(SeparableConv2d):
    ''' Depthwise separable 2D convolution with batchnorm and ReLU '''

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation, bias=bias,
                         _depthwise_conv=Conv2dLayer, _pointwise_conv=Conv2dLayer)


class SkipBlock(nn.Module):
    ''' Container for modules with a skip connection, followed by an aggregator
        (default aggregator: torch.cat) '''

    def __init__(self, main_path, skip_path=Identity(), aggregator=torch.cat):
        super().__init__()
        self.main_path = main_path
        self.skip_path = skip_path
        self.aggregator = aggregator

    def forward(self, x):
        return self.aggregator(self.main_path(x), self.skip_path(x))
