# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.model import BaseModule

# from mmdet.models.utils import CSPLayer
from mmdet.models.layers.csp_layer import CSPLayer
from mmdet3d.registry import MODELS

@MODELS.register_module()
class BIFPN(BaseModule):
    """Path Aggregation Network used in YOLOX.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels,
        num_csp_blocks=3,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        conv_cfg=None,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        init_cfg=dict(type='Kaiming',
                      layer='Conv2d',
                      a=math.sqrt(5),
                      distribution='uniform',
                      mode='fan_in',
                      nonlinearity='leaky_relu'),
        bidirectional=True,
        out_layer_num=4,
        multiscale=True,
        upsample_strides=(1, 2, 4),
    ):
        super(BIFPN, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.bidirectional = bidirectional
        self.out_layer_num = out_layer_num
        self.multiscale = multiscale
        if not self.multiscale:  # output single scale
            self.upsample_list = nn.ModuleList()
            for stride in upsample_strides:
                self.upsample_list.append(nn.Upsample(scale_factor=stride, mode='nearest'))

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(in_channels[idx],
                           in_channels[idx - 1],
                           1,
                           conv_cfg=conv_cfg,
                           norm_cfg=norm_cfg,
                           act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(in_channels[idx - 1] * 2,
                         in_channels[idx - 1],
                         expand_ratio=1.0,
                         num_blocks=num_csp_blocks,
                         add_identity=False,
                         use_depthwise=use_depthwise,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg))
        if bidirectional:
            # build bottom-up blocks
            self.downsamples = nn.ModuleList()
            self.bottom_up_blocks = nn.ModuleList()
            for idx in range(len(in_channels) - 1):
                self.downsamples.append(
                    conv(in_channels[idx],
                         in_channels[idx],
                         3,
                         stride=2,
                         padding=1,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg))
                self.bottom_up_blocks.append(
                    CSPLayer(in_channels[idx] * 2,
                             in_channels[idx + 1],
                             expand_ratio=1.0,
                             num_blocks=num_csp_blocks,
                             add_identity=False,
                             use_depthwise=use_depthwise,
                             conv_cfg=conv_cfg,
                             norm_cfg=norm_cfg,
                             act_cfg=act_cfg))

        # self.out_convs = nn.ModuleList()
        # for i in range(len(in_channels)):
        #     self.out_convs.append(
        #         ConvModule(
        #             in_channels[i],
        #             out_channels,
        #             1,
        #             conv_cfg=conv_cfg,
        #             norm_cfg=norm_cfg,
        #             act_cfg=act_cfg))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)
            outs.insert(0, inner_out)
        if self.bidirectional:
            # bottom-up path
            outs = [inner_outs[0]]
            for idx in range(len(self.in_channels) - 1):
                feat_low = outs[-1]
                feat_height = inner_outs[idx + 1]
                downsample_feat = self.downsamples[idx](feat_low)
                out = self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_height], 1))
                outs.append(out)

        # # out convs
        # for idx, conv in enumerate(self.out_convs):
        #     outs[idx] = conv(outs[idx])
        if self.multiscale:
            return list(tuple(outs)[:self.out_layer_num])
        else:
            outs = [upsample_layer(outs[i]) for i, upsample_layer in enumerate(self.upsample_list)]
            return [torch.cat(outs, dim=1)]

