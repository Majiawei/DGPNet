import warnings

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, caffe2_xavier_init, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS

from mmseg.ops import resize
from mmseg.models.decode_heads.aspp_head import ASPPModule
import torch


class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg, ratio=2, align_corners=False):
        super(upconv, self).__init__()
        self.align_corners = align_corners
        self.conv = ConvModule(in_channels, out_channels, kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ratio = ratio

    def forward(self, x):
        up_x = resize(x, scale_factor=self.ratio, mode='bilinear', align_corners=self.align_corners)
        out = self.conv(up_x)
        return out

@NECKS.register_module()
class DepthDecoder(nn.Module):
    def __init__(self,
                 in_channels=[64, 256, 512, 1024], # [64, 256, 512, 1024]
                 channels=512,
                 dilations=[1, 6, 12, 18],
                 conv_cfg=None,
                 # norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 align_corners=False):
        super(DepthDecoder, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg
        self.align_corners=align_corners
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        ##############################aspp
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels[3],
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels[3],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        ######################level5
        self.conv5_1 = ConvModule(self.channels, self.channels//4, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv5_2 = ConvModule(self.channels // 4, self.channels // 8, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv5_3 = ConvModule(self.channels // 8, self.channels // 16, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv5_4 = ConvModule(self.channels // 16, self.channels // 32, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv5_5 = ConvModule(self.channels // 32, 1, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.up_5 = upconv(self.channels,self.channels,self.conv_cfg,self.norm_cfg,self.act_cfg,2,self.align_corners)
        #########################leval4
        self.conv4_1 = ConvModule(2*self.in_channels[2],self.in_channels[1],3, padding=1,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        self.conv4_2 = ConvModule(self.in_channels[1], self.channels // 8, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv4_3 = ConvModule(self.channels // 8, self.channels // 16, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv4_4 = ConvModule(self.channels // 16, 1, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.up_4 = upconv(self.in_channels[1], self.in_channels[1], self.conv_cfg, self.norm_cfg, self.act_cfg, 2,
                           self.align_corners)
        ########################level3
        self.conv3_1 = ConvModule(2*self.in_channels[1],self.in_channels[0],3, padding=1,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        self.conv3_2 = ConvModule(self.in_channels[0], self.channels // 32, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv3_3 = ConvModule(self.channels // 32, 1, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.up_3 = upconv(self.in_channels[0], self.in_channels[0], self.conv_cfg, self.norm_cfg, self.act_cfg, 2,
                           self.align_corners)
        ########################level2
        self.conv2_1 = ConvModule(2 * self.in_channels[0], self.channels // 32, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv2_2 = ConvModule(self.channels // 32, 1, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.up_2 = upconv(self.channels // 32, self.channels // 32, self.conv_cfg, self.norm_cfg, self.act_cfg, 2,
                           self.align_corners)
        ########################level2
        self.conv1_1 = ConvModule(self.channels // 32, 1, 3, padding=1, conv_cfg=self.conv_cfg,
                                  norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def aspp_mod(self, x):
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        return output
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs): # [64/2,256/4,512/8,1024/16]
        """Forward function."""
        # x = self._transform_inputs(inputs)
        # P5
        aspp_feat = self.aspp_mod(inputs[3]) # 512/16
        lvl5_out = torch.sigmoid(self.conv5_5(self.conv5_4(self.conv5_3(self.conv5_2(self.conv5_1(aspp_feat))))))
        lvl5_up = resize(lvl5_out,scale_factor=2,mode='bilinear',align_corners=self.align_corners)
        #P4
        lvl4_in = torch.cat([self.up_5(aspp_feat),inputs[2]],dim=1)
        lvl4_mid = self.conv4_1(lvl4_in)
        lvl4_out = torch.sigmoid(self.conv4_4(self.conv4_3(self.conv4_2(lvl4_mid))))
        lvl4_up = resize(lvl4_out+lvl5_up,scale_factor=2,mode='bilinear',align_corners=self.align_corners)
        #P3
        lvl3_in = torch.cat([self.up_4(lvl4_mid),inputs[1]],dim=1)
        lvl3_mid = self.conv3_1(lvl3_in)
        lvl3_out = torch.sigmoid(self.conv3_3(self.conv3_2(lvl3_mid)))
        lvl3_up = resize(lvl3_out+lvl4_up,scale_factor=2,mode='bilinear',align_corners=self.align_corners)
        # P2
        lvl2_in = torch.cat([self.up_3(lvl3_mid), inputs[0]], dim=1)
        lvl2_mid = self.conv2_1(lvl2_in)
        lvl2_out = torch.sigmoid(self.conv2_2(lvl2_mid))
        lvl2_up = resize(lvl2_out + lvl3_up, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        # P1
        lvl1_in = self.up_2(lvl2_mid)
        lvl1_out = torch.sigmoid(self.conv1_1(lvl1_in))
        final_depth = torch.sigmoid(lvl1_out+lvl2_up)

        return final_depth,[lvl1_out,lvl2_out,lvl3_out,lvl4_out,lvl5_out]
