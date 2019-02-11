# pylint: disable=W0221,C0414

from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn

import model.nn_ext as nn_ext
from model.backbone import BackboneModule

class Xception(BackboneModule):
    ''' Xception feature extractor backbone '''

    class EntryFlowBlock(nn_ext.SkipBlock):
        ''' Xception entry flow block '''

        def __init__(self, in_channels, out_channels, dilation=1, atrous=False):
            # If atrous mode increase dilation, padding and reset stride
            out_dilation = 2 * dilation if atrous else 1 * dilation
            last_padding = 2 * dilation if atrous else 1 * dilation
            stride = 1 if atrous else 2
            super().__init__(
                main_path=nn.Sequential(
                    nn_ext.SeparableConv2dLayer(in_channels, out_channels, kernel_size=3,
                                                dilation=dilation, padding=1 * dilation),
                    nn_ext.SeparableConv2dLayer(out_channels, out_channels, kernel_size=3,
                                                dilation=dilation, padding=1 * dilation),
                    nn_ext.SeparableConv2dLayer(out_channels, out_channels, kernel_size=3,
                                                dilation=out_dilation, stride=stride,
                                                padding=last_padding)),
                skip_path=nn_ext.Conv2dLayer(in_channels, out_channels, kernel_size=1,
                                             dilation=out_dilation, stride=stride),
                aggregator=torch.add)

    class MiddleFlowBlock(nn_ext.SkipBlock):
        ''' Xception middle flow block '''

        def __init__(self, in_channels=728, out_channels=728, dilation=1):
            super().__init__(
                main_path=nn.Sequential(
                    nn_ext.SeparableConv2dLayer(in_channels, out_channels, kernel_size=3,
                                                dilation=dilation, padding=1 * dilation),
                    nn_ext.SeparableConv2dLayer(out_channels, out_channels, kernel_size=3,
                                                dilation=dilation, padding=1 * dilation),
                    nn_ext.SeparableConv2dLayer(out_channels, out_channels, kernel_size=3,
                                                dilation=dilation, padding=1 * dilation)),
                aggregator=torch.add)

    class ExitFlowBlock(nn_ext.SkipBlock):
        ''' Xception exit flow block '''

        def __init__(self, in_channels=728, out_channels=1024, dilation=1, atrous=False):
            # If atrous mode increase dilation, padding and reset stride
            out_dilation = 2 * dilation if atrous else 1 * dilation
            last_padding = 2 * dilation if atrous else 1 * dilation
            stride = 1 if atrous else 2
            super().__init__(
                main_path=nn.Sequential(
                    nn_ext.SeparableConv2dLayer(in_channels, in_channels, kernel_size=3,
                                                dilation=dilation, padding=1 * dilation),
                    nn_ext.SeparableConv2dLayer(in_channels, out_channels, kernel_size=3,
                                                dilation=dilation, padding=1 * dilation),
                    nn_ext.SeparableConv2dLayer(out_channels, out_channels, kernel_size=3,
                                                dilation=out_dilation, stride=stride,
                                                padding=last_padding)),
                skip_path=nn_ext.Conv2dLayer(in_channels, out_channels, kernel_size=1,
                                             dilation=out_dilation, stride=stride),
                aggregator=torch.add)

    def __init__(self, output_stride=16):

        if output_stride not in {8, 16}:
            raise ValueError('Invalid output_stride; Supported values: {8, 16}')
        low_out_channels = 128 if output_stride == 16 else 64
        super().__init__(output_stride=16, out_channels=2048, low_out_channels=low_out_channels)

        # Adjust dilation rates to control output_stride
        opts = defaultdict(lambda: {})
        if output_stride == 16:
            opts['exit_flow_block_0'] = {'atrous': True}
            opts['exit_flow_block_1'] = {'dilation':2, 'padding': 2}
        elif output_stride == 8:
            opts['entry_flow_block_3'] = {'atrous': True}
            opts['middle_flow_block'] = {'dilation': 2}
            opts['exit_flow_block_0'] = {'dilation': 2, 'atrous': True}
            opts['exit_flow_block_1'] = {'dilation': 4, 'padding': 4}
        else:
            raise ValueError('Invalid output_stride; Supported values: {8, 16}')

        entry_flow = nn.Sequential(OrderedDict([
            ('block_0', nn.Sequential(
                nn_ext.Conv2dLayer(3, 32, kernel_size=3, stride=2, padding=1),
                nn_ext.Conv2dLayer(32, 64, kernel_size=3, padding=1))),
            ('block_1', Xception.EntryFlowBlock(64, 128, **opts['entry_flow_block_1'])),
            ('block_2', Xception.EntryFlowBlock(128, 256, **opts['entry_flow_block_2'])),
            ('block_3', Xception.EntryFlowBlock(256, 728, **opts['entry_flow_block_3'])),
        ]))

        # [DeepLabV3+ specific] Split-entry flow sequence to extract
        # low-level features needed by decoder (at output_stride // 4)
        if output_stride == 16:
            self.entry_flow = nn.ModuleList([entry_flow[0:2], entry_flow[2:4]])
        elif output_stride == 8:
            self.entry_flow = nn.ModuleList([entry_flow[0:1], entry_flow[1:4]])
        else:
            raise ValueError('Invalid output_stride; Supported values: {8, 16}')

        self.middle_flow = nn.Sequential(OrderedDict([
            ('block_0', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_1', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_2', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_3', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_4', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_5', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_6', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_7', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_8', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_9', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_10', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_11', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_12', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_13', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_14', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
            ('block_15', Xception.MiddleFlowBlock(**opts['middle_flow_block'])),
        ]))

        self.exit_flow = nn.Sequential(OrderedDict([
            ('block_0', Xception.ExitFlowBlock(**opts['exit_flow_block_0'])),
            ('block_1', nn.Sequential(
                nn_ext.SeparableConv2dLayer(1024, 1536, kernel_size=3, **opts['exit_flow_block_1']),
                nn_ext.SeparableConv2dLayer(1536, 1536, kernel_size=3, **opts['exit_flow_block_1']),
                nn_ext.SeparableConv2dLayer(1536, 2048, kernel_size=3, **opts['exit_flow_block_1'])))
        ]))

    def forward(self, x):
        x_low = self.entry_flow[0](x)
        x = self.entry_flow[1](x_low)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        return x, x_low
