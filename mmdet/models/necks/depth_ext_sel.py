import warnings

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, caffe2_xavier_init, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS

from mmseg.ops import resize
from mmseg.models.decode_heads.aspp_head import ASPPModule
import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, constant_init, kaiming_init
from mmseg.ops import resize
from mmcv.cnn.bricks import NonLocal2d

class ASPPMod(nn.Module):
    def __init__(self, in_channels=256, channels=256, dilations=(1, 6, 12, 18), norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU')):
        super(ASPPMod, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = None
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = False
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            self.dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(self.dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        """Forward function."""
        aspp_outs=[]
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

class depth_feat_ext(nn.Module):
    def __init__(self, in_channels=256, base_channels=256, out_channels=256, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU')):
        super(depth_feat_ext, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # self.conv_block1 = nn.Sequential(
        #     ConvModule(self.in_channels, self.base_channels // 2, 3, stride=2, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        #     ConvModule(self.base_channels // 2, self.base_channels // 2, 3, stride=1, padding=1,
        #                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        #     ConvModule(self.base_channels // 2, self.base_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        # )
        self.conv_block1 = nn.Sequential(
            ConvModule(self.in_channels, self.base_channels, 3, stride=2, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(self.base_channels, self.base_channels, 3, stride=1, padding=1,
                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(self.base_channels, self.base_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_block2 = nn.Sequential(
            ConvModule(self.base_channels, self.base_channels, 1, stride=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(self.base_channels, self.base_channels, 3, stride=1, padding=1,
                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            ConvModule(self.base_channels, self.out_channels, 1, stride=1, norm_cfg=self.norm_cfg, act_cfg=None)
        )
        self.residual = ConvModule(self.base_channels, self.out_channels, 1, stride=1, norm_cfg=self.norm_cfg,
                                   act_cfg=None)
        self.act2 = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, depth):
        x = self.conv_block1(depth) # 1/2
        x = self.maxpool(x) # 1/4

        res = x
        x = self.conv_block2(x)
        res = self.residual(res)
        x = self.act2(x + res) #1/4

        return x

@NECKS.register_module()
class DepthExtSel(nn.Module):
    def __init__(self,
                 in_feat_channels=[64, 256, 256, 256, 256],
                 num_features=128,
                 out_channels=128,
                 max_depth=80.0,
                 align_corners=False,
                 dropout_ratio=0.1,
                 dilations=(1, 2, 6, 12, 18),
                 refine_type='conv',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU')
                 ):
        super(DepthExtSel, self).__init__()
        self.in_feat_channels=in_feat_channels
        self.num_features=num_features
        self.max_depth=max_depth
        self.conv_cfg=conv_cfg
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg
        self.align_corners=align_corners
        self.refine_type=refine_type
        self.dilations = dilations
        self.out_channels=out_channels

        self.depth_decode_32 = nn.Sequential(
            ConvModule(self.in_feat_channels[4],self.num_features,3,padding=1,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=self.align_corners),
            ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
            ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
            ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
            # ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
            #            act_cfg=self.act_cfg),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
        )
        self.depth_decode_16 = nn.Sequential(
            ConvModule(self.in_feat_channels[3],self.num_features,3,padding=1,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=self.align_corners),
            ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
            ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
            # ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
            #            act_cfg=self.act_cfg),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
        )
        self.depth_decode_8 = nn.Sequential(
            ConvModule(self.in_feat_channels[2],self.num_features,3,padding=1,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=self.align_corners),
            ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
            # ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
            #            act_cfg=self.act_cfg),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
        )
        self.depth_decode_4 = nn.Sequential(
            ConvModule(self.in_feat_channels[1],self.num_features,3,padding=1,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=self.align_corners),
            # ConvModule(self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
            #            act_cfg=self.act_cfg),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
        )

        # self.depth_decode_2 = nn.Sequential(
        #     ConvModule(self.in_feat_channels[0]+self.num_features, self.num_features, 3, padding=1, norm_cfg=self.norm_cfg,
        #                act_cfg=self.act_cfg),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=self.align_corners),
        # )
        self.depth_decode_2 = ConvModule(self.in_feat_channels[0], self.num_features, 3, padding=1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.get_depth = nn.Sequential(
            nn.Conv2d(self.num_features, 1, kernel_size=1),
            nn.Sigmoid()
        )
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        ##############################depth refine & seletion
        # if self.refine_type == 'conv':
        #     self.refine = ConvModule(
        #         self.num_features,
        #         self.num_features,
        #         3,
        #         padding=1,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg
        #     )
        # elif self.refine_type == 'non_local':
        #     self.refine = NonLocal2d(
        #         self.num_features,
        #         reduction=1,
        #         use_scale=False,
        #         conv_cfg=self.conv_cfg,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg
        #     )
        # self.ext_feat = depth_feat_ext(in_channels=self.num_features, base_channels=self.num_features, out_channels=self.out_channels, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg) # 1/4
        #
        # self.daspp1 = nn.Sequential(
        #     ConvModule(in_channels=self.num_features, out_channels=self.out_channels*2, kernel_size=1, padding=0,
        #                dilation=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        #     ConvModule(in_channels=self.num_features*2, out_channels=self.out_channels, kernel_size=1, padding=0,
        #                dilation=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        # )
        # self.daspp3 = nn.Sequential(
        #     ConvModule(in_channels=self.num_features+self.out_channels, out_channels=self.out_channels*2, kernel_size=1, padding=0,conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        #     ConvModule(in_channels=self.out_channels*2, out_channels=self.out_channels, kernel_size=3,padding=3,dilation=3,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        # )
        # self.daspp6 = nn.Sequential(
        #     ConvModule(in_channels=self.num_features+self.out_channels*2, out_channels=self.out_channels*2, kernel_size=1, padding=0,conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        #     ConvModule(in_channels=self.out_channels*2, out_channels=self.out_channels, kernel_size=3,padding=6,dilation=6,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        # )
        # self.daspp12 = nn.Sequential(
        #     ConvModule(in_channels=self.num_features+self.out_channels*3, out_channels=self.out_channels*2, kernel_size=1, padding=0,conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        #     ConvModule(in_channels=self.out_channels*2, out_channels=self.out_channels, kernel_size=3,padding=12,dilation=12,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        # )
        # self.daspp18 = nn.Sequential(
        #     ConvModule(in_channels=self.num_features+self.out_channels*4, out_channels=self.out_channels*2, kernel_size=1, padding=0,conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
        #     ConvModule(in_channels=self.out_channels*2, out_channels=self.out_channels, kernel_size=3,padding=18,dilation=18,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        # )

        #################################################
        # self.daspp1 = ConvModule(in_channels=self.num_features,out_channels=self.out_channels,kernel_size=1,padding=0,dilation=1,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        # self.daspp3 = ConvModule(in_channels=self.num_features+self.out_channels,out_channels=self.out_channels,kernel_size=3,padding=3,dilation=3,conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        # self.daspp6 = ConvModule(in_channels=self.num_features + self.out_channels*2, out_channels=self.out_channels,
        #                          kernel_size=3, padding=6, dilation=6, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
        #                          act_cfg=self.act_cfg)
        # self.daspp12 = ConvModule(in_channels=self.num_features + self.out_channels*3, out_channels=self.out_channels,
        #                          kernel_size=3, padding=12, dilation=12, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
        #                          act_cfg=self.act_cfg)
        # self.daspp18 = ConvModule(in_channels=self.num_features + self.out_channels*4, out_channels=self.out_channels,
        #                          kernel_size=3, padding=18, dilation=18, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg,
        #                          act_cfg=self.act_cfg)

    def init_weights(self):
        # self.ext_feat.init_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        # for n in self.ext_feat_list:
        #     n.init_weights()

    def forward(self, inputs):
        skip0,skip1,skip2,skip3,skip4 = inputs[0],inputs[1],inputs[2],inputs[3],inputs[4]
        depth_scaled_32 = self.depth_decode_32(inputs[4]) # 1/2

        # d_32_ds = resize(depth_scaled_32,scale_factor=1/8, mode='bilinear',align_corners=self.align_corners)
        # skip3 = torch.cat([inputs[3],d_32_ds],dim=1)
        depth_scaled_16 = self.depth_decode_16(skip3) # 1/2

        # d_16_ds = resize(depth_scaled_16,scale_factor=1/4, mode='bilinear',align_corners=self.align_corners)
        # skip2 = torch.cat([inputs[2], d_16_ds], dim=1)
        depth_scaled_8 = self.depth_decode_8(skip2) # 1/2

        # d_8_ds = resize(depth_scaled_8,scale_factor=1/2, mode='bilinear',align_corners=self.align_corners)
        # skip1 = torch.cat([inputs[1], d_8_ds], dim=1)
        depth_scaled_4 = self.depth_decode_4(skip1) # 1/2

        # d_4_ds = resize(depth_scaled_4,scale_factor=1/2,mode='bilinear',align_corners=self.align_corners)
        # skip0 = torch.cat([inputs[0], depth_scaled_4], dim=1)
        depth_scaled_2 = self.depth_decode_2(skip0) # 1/2

        depth_scaled_list = [depth_scaled_2, depth_scaled_4,depth_scaled_8,depth_scaled_16,depth_scaled_32]
        depth_scaled_sum = sum(depth_scaled_list)

        if self.dropout is not None:
            depth_scaled_sum = self.dropout(depth_scaled_sum)
        final_depth = self.max_depth * self.get_depth(depth_scaled_sum)
        ###################################refine & sel
        d_outs_list=[]
        # refine_depth = depth_scaled_sum.detach() / len(inputs)
        # # refine_depth = self.refine(refine_depth)
        # # d_outs_list = []
        # # for i in range(len(inputs)):
        # #     out_size = inputs[i].size()[2:]
        # #     agg_feat = F.interpolate(refine_depth+depth_scaled_list[i].detach(), size=out_size, mode='bilinear', align_corners=self.align_corners)
        # #     ext_feat = self.ext_feat(agg_feat)
        # #     d_outs_list.append(ext_feat)
        # # d_outs_list = []
        # # for i in range(len(inputs)):
        # #     ext_feat = self.ext_feat_list[i](refine_depth)
        # #     d_outs_list.append(ext_feat)
        # # d_outs_list = self.ext_feat_list(refine_depth)
        # #################################3
        # scaled_refine_depth = self.ext_feat(refine_depth) # 1/8
        # daspp1 = self.daspp1(scaled_refine_depth)
        # concat1 = torch.cat([scaled_refine_depth, daspp1], dim=1)
        # daspp3 = self.daspp3(concat1)
        # concat2 = torch.cat([concat1, daspp3], dim=1)
        # daspp6 = self.daspp6(concat2)
        # concat3 = torch.cat([concat2, daspp6], dim=1)
        # daspp12 = self.daspp12(concat3)
        # concat4 = torch.cat([concat3, daspp12], dim=1)
        # daspp18 = self.daspp18(concat4)
        # d_outs_list = [daspp1,daspp3,daspp6,daspp12,daspp18]
        return final_depth,d_outs_list
