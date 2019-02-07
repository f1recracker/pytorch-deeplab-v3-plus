# pylint: disable=W0221,C0414,C0103

from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as nn_func

class Conv2d(nn.Module):
    ''' 2D convolution with ReLU and batch norm '''

    def __init__(self, in_channels, out_channels, kernel_size,
                 groups=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      groups=groups, stride=stride, padding=padding,
                      dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=3e-4),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class SeparableConv2d(nn.Module):
    ''' Depthwise separable 2D convolution with ReLU and batch norm '''

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.depthwise_conv = Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, stride=stride, padding=padding,
            dilation=dilation, bias=bias)
        self.pointwise_conv = Conv2d(
            in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class SkipBlock(nn.Module):
    ''' Container for a module with a skip connection '''

    class Identity(nn.Module):
        ''' Identity transformation operation '''

        def forward(self, x):
            return x

    def __init__(self, main_path, skip_path=Identity(), reduce_op=torch.cat):
        super().__init__()
        self.main_path = main_path
        self.skip_path = skip_path
        self.reduce_op = reduce_op

    def forward(self, x):
        return self.reduce_op(self.main_path(x), self.skip_path(x))

class Xception(nn.Module):

    class EntryFlowBlock(SkipBlock):

        def __init__(self, in_channels, out_channels, dilation=1, atrous=False):
            # If atrous mode increase dilation, padding and reset stride
            out_dilation = 2 * dilation if atrous else 1 * dilation
            last_padding = 2 * dilation if atrous else 1 * dilation
            stride = 1 if atrous else 2
            super().__init__(
                main_path=nn.Sequential(
                    SeparableConv2d(in_channels, out_channels, kernel_size=3,
                                    dilation=dilation, padding=1 * dilation),
                    SeparableConv2d(out_channels, out_channels, kernel_size=3,
                                    dilation=dilation, padding=1 * dilation),
                    SeparableConv2d(out_channels, out_channels, kernel_size=3,
                                    dilation=out_dilation, stride=stride, padding=last_padding)),
                skip_path=Conv2d(in_channels, out_channels, kernel_size=1,
                                 dilation=out_dilation, stride=stride),
                reduce_op=torch.add)

    class MiddleFlowBlock(SkipBlock):

        def __init__(self, in_channels=728, out_channels=728, dilation=1):
            super().__init__(
                main_path=nn.Sequential(
                    SeparableConv2d(in_channels, out_channels, kernel_size=3,
                                    dilation=dilation, padding=1 * dilation),
                    SeparableConv2d(out_channels, out_channels, kernel_size=3,
                                    dilation=dilation, padding=1 * dilation),
                    SeparableConv2d(out_channels, out_channels, kernel_size=3,
                                    dilation=dilation, padding=1 * dilation)),
                reduce_op=torch.add)

    class ExitFlowBlock(SkipBlock):

        def __init__(self, in_channels=728, out_channels=1024, dilation=1, atrous=False):
            # If atrous mode increase dilation, padding and reset stride
            out_dilation = 2 * dilation if atrous else 1 * dilation
            last_padding = 2 * dilation if atrous else 1 * dilation
            stride = 1 if atrous else 2
            super().__init__(
                main_path=nn.Sequential(
                    SeparableConv2d(in_channels, in_channels, kernel_size=3,
                                    dilation=dilation, padding=1 * dilation),
                    SeparableConv2d(in_channels, out_channels, kernel_size=3,
                                    dilation=dilation, padding=1 * dilation),
                    SeparableConv2d(out_channels, out_channels, kernel_size=3,
                                    dilation=out_dilation, stride=stride, padding=last_padding)),
                skip_path=Conv2d(in_channels, out_channels, kernel_size=1,
                                 dilation=out_dilation, stride=stride),
                reduce_op=torch.add)

    def __init__(self, output_stride=16):
        super().__init__()

        self.output_stride = output_stride
        self.out_channels = 2048

        if output_stride == 16:
            self.low_out_channels = 128
        elif output_stride == 8:
            self.low_out_channels = 64

        # Adjust dilation rates to control output_stride
        add_opts = defaultdict(lambda: {})
        if output_stride == 16:
            add_opts['exit_flow_block_0'] = {'atrous': True}
            add_opts['exit_flow_block_1'] = {'dilation':2, 'padding': 2}
        elif output_stride == 8:
            add_opts['entry_flow_block_3'] = {'atrous': True}
            add_opts['middle_flow_block'] = {'dilation': 2}
            add_opts['exit_flow_block_0'] = {'dilation': 2, 'atrous': True}
            add_opts['exit_flow_block_1'] = {'dilation': 4, 'padding': 4}
        else:
            raise NotImplementedError('Invalid output_stride; Supported values: {8, 16}')

        entry_flow = nn.Sequential(OrderedDict([
            ('block_0', nn.Sequential(
                Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                Conv2d(32, 64, kernel_size=3, padding=1))),
            ('block_1', Xception.EntryFlowBlock(64, 128, **add_opts['entry_flow_block_1'])),
            ('block_2', Xception.EntryFlowBlock(128, 256, **add_opts['entry_flow_block_2'])),
            ('block_3', Xception.EntryFlowBlock(256, 728, **add_opts['entry_flow_block_3'])),
        ]))

        # [DeepLabV3+ specific] Split-entry flow sequence to extract
        # low-level features needed by decoder (at output_stride // 4)
        if output_stride == 16:
            self.entry_flow = nn.ModuleList([entry_flow[0:2], entry_flow[2:4]])
        elif output_stride == 8:
            self.entry_flow = nn.ModuleList([entry_flow[0:1], entry_flow[1:4]])
        else:
            raise NotImplementedError('Invalid output_stride; Supported values: {8, 16}')

        self.middle_flow = nn.Sequential(OrderedDict([
            ('block_0', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_1', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_2', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_3', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_4', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_5', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_6', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_7', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_8', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_9', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_10', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_11', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_12', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_13', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_14', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
            ('block_15', Xception.MiddleFlowBlock(**add_opts['middle_flow_block'])),
        ]))

        self.exit_flow = nn.Sequential(OrderedDict([
            ('block_0', Xception.ExitFlowBlock(**add_opts['exit_flow_block_0'])),
            ('block_1', nn.Sequential(
                SeparableConv2d(1024, 1536, kernel_size=3, **add_opts['exit_flow_block_1']),
                SeparableConv2d(1536, 1536, kernel_size=3, **add_opts['exit_flow_block_1']),
                SeparableConv2d(1536, 2048, kernel_size=3, **add_opts['exit_flow_block_1'])))
        ]))

    def forward(self, x):
        x_low = self.entry_flow[0](x)
        x = self.entry_flow[1](x_low)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x, x_low

class DeepLab(nn.Module):

    class ASPP(nn.Module):

        def __init__(self, in_channels, output_stride=16):
            super().__init__()

            if output_stride == 16:
                dilation_factor = 1
            elif output_stride == 8:
                dilation_factor = 2
            else:
                raise NotImplementedError('Invalid output_stride; Supported values: {8, 16}')

            self.aspp = nn.ModuleList([
                Conv2d(in_channels, 256, kernel_size=1, dilation=1),
                Conv2d(in_channels, 256, kernel_size=3,
                       dilation=6 * dilation_factor, padding=6 * dilation_factor),
                Conv2d(in_channels, 256, kernel_size=3,
                       dilation=12 * dilation_factor, padding=12 * dilation_factor),
                Conv2d(in_channels, 256, kernel_size=3,
                       dilation=18 * dilation_factor, padding=18 * dilation_factor)])

            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Conv2d(in_channels, 256, kernel_size=1))

            self.conv = Conv2d(256 * 5, 256, kernel_size=1)

        def forward(self, x):
            x_aspp = [aspp(x) for aspp in self.aspp]
            x_pool = self.global_avg_pool(x)
            x_pool = nn_func.interpolate(x_pool, size=x.shape[2:4])
            feats = torch.cat((*x_aspp, x_pool), dim=1)
            feats = self.conv(feats)
            return feats

    class Decoder(nn.Module):

        def __init__(self, low_in_channels, output_stride, num_classes):
            super().__init__()

            if output_stride == 16:
                self.logit_upsample = 4
            elif output_stride == 8:
                self.logit_upsample = 2
            else:
                raise NotImplementedError('Invalid output_stride; Supported values: {8, 16}')

            self.conv_low = Conv2d(low_in_channels, 48, kernel_size=1)
            self.conv_logit = Conv2d(48 + 256, num_classes, kernel_size=3, padding=1)

        def forward(self, feats, low_feats):
            low_feats = self.conv_low(low_feats)
            feats = nn_func.interpolate(feats, size=low_feats.shape[2:4], mode='bilinear',
                                        align_corners=True)
            feats = torch.cat([feats, low_feats], dim=1)
            logits = self.conv_logit(feats)
            logits = nn_func.interpolate(logits, scale_factor=self.logit_upsample,
                                         mode='bilinear', align_corners=True)
            return logits

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.aspp = DeepLab.ASPP(in_channels=backbone.out_channels,
                                 output_stride=backbone.output_stride)
        self.decoder = DeepLab.Decoder(low_in_channels=backbone.low_out_channels,
                                       output_stride=backbone.output_stride,
                                       num_classes=num_classes)

    def forward(self, x):
        x, x_low = self.backbone(x)
        x = self.aspp(x)
        logits = self.decoder(x, x_low)
        return logits

if __name__ == '__main__':

    def test_out_shapes(model, in_shape, out_shape):
        x = torch.rand(*in_shape)
        if next(model.parameters()).is_cuda:
            x = x.cuda()
        with torch.no_grad():
            y = model.forward(x)
        assert y.shape == out_shape, 'Output size mismatch!'

    def fps(model, in_shape):
        x = torch.rand(*in_shape)
        if next(model.parameters()).is_cuda:
            x = x.cuda()

        import timeit
        with torch.no_grad():
            duration = timeit.timeit(lambda: model.forward(x), number=100)
        return in_shape[0] * 100 / duration

    deeplab = DeepLab(Xception(output_stride=16), num_classes=20)
    test_out_shapes(deeplab, (1, 3, 1280, 720), (1, 20, 1280, 720))
    test_out_shapes(deeplab, (1, 3, 640, 360), (1, 20, 640, 360))

    deeplab = DeepLab(Xception(output_stride=8), num_classes=20)
    test_out_shapes(deeplab, (1, 3, 1280, 720), (1, 20, 1280, 720))
    test_out_shapes(deeplab, (1, 3, 640, 360), (1, 20, 640, 360))

    deeplab = DeepLab(Xception(output_stride=16), num_classes=20)
    print('FPS (Xception, out_stride=16, size=(1280, 720))', fps(deeplab, (1, 3, 1280, 720)))
    print('FPS (Xception, out_stride=16, size=(640, 360))', fps(deeplab, (1, 3, 640, 360)))

    deeplab = DeepLab(Xception(output_stride=8), num_classes=20)
    print('FPS (Xception, out_stride=8, size=(1280, 720))', fps(deeplab, (1, 3, 1280, 720)))
    print('FPS (Xception, out_stride=8, size=(640, 360))', fps(deeplab, (1, 3, 640, 360)))
