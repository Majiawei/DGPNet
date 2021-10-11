import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
from mmcv.cnn.bricks import NonLocal2d

from ..builder import NECKS


@NECKS.register_module()
class MFFPN(nn.Module):
    def __init__(self,
                 in_channels, # [256, 512, 1024, 2048]
                 out_channels, # 256
                 num_outs, # 5
                 start_level=0, # 1
                 end_level=-1,
                 add_extra_convs=False, # 'on_input'
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(MFFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs # False
        self.no_norm_on_lateral = no_norm_on_lateral # False
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins # 4
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level # 1
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs # 'on_input'
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            if extra_convs_on_inputs:
                # For compatibility with previous release
                # TODO: deprecate `extra_convs_on_inputs`
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.is_act = True
        if self.is_act:
            self.act_list = nn.ModuleList()
            for i in range(self.num_outs):
                self.act_list.append(nn.Sigmoid())
        # self.bn_relu = True
        # if self.bn_relu:
        #     self.bn_list = nn.ModuleList()
        #     self.relu_list = nn.ModuleList()
        #     for i in range(self.num_outs):
        #         self.bn_list.append(nn.BatchNorm2d(out_channels))
        #         self.relu_list.append(nn.ReLU(inplace=True))
        self.is_fpn = True
        if self.is_fpn:
            self.fpn_convs = nn.ModuleList()
            for i in range(2*self.num_outs):
                self.fpn_convs.append(
                    ConvModule(out_channels,out_channels,3,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)
                )
        self.fpn_channels = [512, 1024, 2048, 256, 256]
        self.fpn_channels_depth = [512, 1024, 2048, 256, 256]
        self.img_p5_to_p6 = ConvModule(self.fpn_channels[2],out_channels,3,stride=2,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)
        self.img_p6_to_p7 = ConvModule(self.fpn_channels[3],out_channels,3,stride=2,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)
        self.depth_p5_to_p6 = ConvModule(self.fpn_channels_depth[2],out_channels,3,stride=2,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)
        self.depth_p6_to_p7 = ConvModule(self.fpn_channels_depth[3],out_channels,3,stride=2,padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)

        self.img_p3_down_channel = ConvModule(self.fpn_channels[0],out_channels,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,act_cfg=act_cfg,inplace=False)
        self.img_p4_down_channel = ConvModule(self.fpn_channels[1],out_channels,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,act_cfg=act_cfg,inplace=False)
        self.img_p5_down_channel = ConvModule(self.fpn_channels[2],out_channels,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,act_cfg=act_cfg,inplace=False)

        self.depth_p3_down_channel = ConvModule(self.fpn_channels_depth[0],out_channels,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,act_cfg=act_cfg,inplace=False)
        self.depth_p4_down_channel = ConvModule(self.fpn_channels_depth[1],out_channels,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,act_cfg=act_cfg,inplace=False)
        self.depth_p5_down_channel = ConvModule(self.fpn_channels_depth[2],out_channels,1,conv_cfg=conv_cfg,norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,act_cfg=act_cfg,inplace=False)

        # Weight
        self.epsilon = 1e-4
        self.p7_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w1_relu = nn.ReLU()
        self.p6_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p7_w = ConvModule(out_channels*2,2,1,stride=1,padding=0,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)
        self.p6_w = ConvModule(out_channels*3,3,1,stride=1,padding=0,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)
        self.p5_w = ConvModule(out_channels*3,3,1,stride=1,padding=0,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)
        self.p4_w = ConvModule(out_channels*3,3,1,stride=1,padding=0,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)
        self.p3_w = ConvModule(out_channels*3,3,1,stride=1,padding=0,conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)

        self.refine_level = 2
        self.refine_type = 'non_local'
        # assert 0 <= self.refine_level < self.num_levels
        if self.refine_type == 'conv':
            self.refine = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                out_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg)
        #######################################################################33333

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def ada_fus_func(self,x,depth,layer_index):
        # if self.deformable: # False
        #     depth = self.deform_layer(depth)
        #     x = x * depth

        if self.adaptive_diated: # True
            weight = self.ada_layers_list[layer_index](x).reshape(-1, self.fpn_channels[layer_index], 1, 3)
            weight = self.ada_softmax_list[layer_index](weight)
            x = self.dynamic_local_filtering(x, depth, dilated=1) * weight[:, :, :, 0:1] \
                + self.dynamic_local_filtering(x, depth, dilated=2) * weight[:, :, :, 1:2] \
                + self.dynamic_local_filtering(x, depth, dilated=3) * weight[:, :, :, 2:3]
            x = self.ada_bn_list[layer_index](x)
            x = self.ada_relu_list[layer_index](x)
        else:
            x = self.dynamic_local_filtering(x, depth, dilated=1) + self.dynamic_local_filtering(x, depth, dilated=2) + self.dynamic_local_filtering(x, depth, dilated=3)
        return x

    def dynamic_local_filtering(self, x, depth, dilated=1):
        padding = nn.ReflectionPad2d(dilated)  # ConstantPad2d(1, 0)
        pad_depth = padding(depth)
        n, c, h, w = x.size()
        # y = torch.cat((x[:, int(c/2):, :, :], x[:, :int(c/2), :, :]), dim=1)
        # x = x + y
        y = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)
        z = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)
        x = (x + y + z) / 3
        pad_x = padding(x)
        filter = (pad_depth[:, :, dilated: dilated + h, dilated: dilated + w] * pad_x[:, :, dilated: dilated + h, dilated: dilated + w]).clone()
        for i in [-dilated, 0, dilated]:
            for j in [-dilated, 0, dilated]:
                if i != 0 or j != 0:
                    filter += (pad_depth[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w] * pad_x[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w]).clone()
        return filter / 9
    def channel_flow(self,x):
        y = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)
        z = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)
        x = (x + y + z) / 3
        return x
    
    @auto_fp16()
    def forward(self, inputs, depth):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        img_p2, img_p3, img_p4, img_p5 = inputs # 256,512,1024,2048
        depth_p2, depth_p3, depth_p4, depth_p5 = depth # 64,128,256,512

        img_p6_in = self.img_p5_to_p6(img_p5)
        img_p7_in = self.img_p6_to_p7(img_p6_in)
        depth_p6_in = self.depth_p5_to_p6(depth_p5)
        depth_p7_in = self.depth_p6_to_p7(depth_p6_in)

        img_p3_in = self.img_p3_down_channel(img_p3)
        img_p4_in = self.img_p4_down_channel(img_p4)
        img_p5_in = self.img_p5_down_channel(img_p5)
        depth_p3_in = self.depth_p3_down_channel(depth_p3)
        depth_p4_in = self.depth_p4_down_channel(depth_p4)
        depth_p5_in = self.depth_p5_down_channel(depth_p5)

        # ################################### depth feature fusion
        # # step 1: gather multi-level features by resize and average
        # depth_feats = []
        # depth_gather_size = depth_p5_in.size()[2:]
        # depth_feats.append(F.adaptive_max_pool2d(depth_p3_in, output_size=depth_gather_size))
        # depth_feats.append(F.adaptive_max_pool2d(depth_p4_in, output_size=depth_gather_size))
        # depth_feats.append(F.interpolate(depth_p5_in, size=depth_gather_size, mode='nearest'))
        # depth_feats.append(F.interpolate(depth_p6_in, size=depth_gather_size, mode='nearest'))
        # depth_feats.append(F.interpolate(depth_p7_in, size=depth_gather_size, mode='nearest'))
        # bsf = sum(depth_feats) / len(depth_feats)
        # # step 2: refine gathered features
        # if self.refine_type is not None:
        #     bsf = self.refine(bsf)
        # step 3: scatter refined features to multi-levels by a residual path
        # depth_p3_mid = F.interpolate(bsf, size=depth_p3_in.size()[2:], mode='nearest')
        # depth_p4_mid = F.interpolate(bsf, size=depth_p4_in.size()[2:], mode='nearest')
        # depth_p5_mid = F.adaptive_max_pool2d(bsf, output_size=depth_p5_in.size()[2:])
        # depth_p6_mid = F.adaptive_max_pool2d(bsf, output_size=depth_p6_in.size()[2:])
        # depth_p7_mid = F.adaptive_max_pool2d(bsf, output_size=depth_p7_in.size()[2:])

        ############################################################

        # depth internal fusion
        depth_p7_out = torch.exp(self.act_list[0](self.channel_flow(depth_p7_in)))
        depth_p6_out = torch.exp(self.act_list[1](self.channel_flow(depth_p6_in)))
        depth_p5_out = torch.exp(self.act_list[2](self.channel_flow(depth_p5_in)))
        # depth_p4_out = torch.exp(self.act_list[3](depth_p4_in+F.interpolate(depth_p5_out,size=depth_p4_in.shape[2:],**self.upsample_cfg)))
        # depth_p3_out = torch.exp(self.act_list[4](depth_p3_in+F.interpolate(depth_p4_out,size=depth_p3_in.shape[2:],**self.upsample_cfg)))
        depth_p4_out = torch.exp(self.act_list[3](self.channel_flow(depth_p4_in)))
        depth_p3_out = torch.exp(self.act_list[4](self.channel_flow(depth_p3_in)))

        # depth_p7_out = depth_p7_in
        # depth_p6_out = depth_p6_in
        # depth_p5_out = depth_p5_in
        # depth_p4_out = depth_p4_in
        # depth_p3_out = depth_p3_in

        # img depth external fusion
        #     y = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)
        #     z = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)
        #     x = (x + y + z) / 3
        p7_fus = img_p7_in * depth_p7_out
        p6_fus = img_p6_in * depth_p6_out
        p5_fus = img_p5_in * depth_p5_out
        p4_fus = img_p4_in * depth_p4_out
        p3_fus = img_p3_in * depth_p3_out

        outs = []

        # Weighted fusion
        # p7_w1 = self.p7_w1_relu(self.p7_w1)
        # weight = p7_w1 / (torch.sum(p7_w1, dim=0) + self.epsilon)
        # p7_out = self.fpn_convs[0](weight[0] * img_p7_in + weight[1] * p7_fus)

        # p6_w1 = self.p6_w1_relu(self.p6_w1)
        # weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # p6_out = self.fpn_convs[1](weight[0] * img_p6_in + weight[1] * p6_fus + weight[2] * F.interpolate(p7_out,size=img_p6_in.shape[2:],**self.upsample_cfg))

        # p5_w1 = self.p5_w1_relu(self.p5_w1)
        # weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # p5_out = self.fpn_convs[2](weight[0] * img_p5_in + weight[1] * p5_fus + weight[2] * F.interpolate(p6_out,size=img_p5_in.shape[2:],**self.upsample_cfg))

        # p4_w1 = self.p4_w1_relu(self.p4_w1)
        # weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # p4_out = self.fpn_convs[3](weight[0] * img_p4_in + weight[1] * p4_fus + weight[2] * F.interpolate(p5_out,size=img_p4_in.shape[2:],**self.upsample_cfg))

        # p3_w1 = self.p3_w1_relu(self.p3_w1)
        # weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # p3_out = self.fpn_convs[4](weight[0] * img_p3_in + weight[1] * p3_fus + weight[2] * F.interpolate(p4_out,size=img_p3_in.shape[2:],**self.upsample_cfg))

        p7_w = F.softmax(self.p7_w(torch.cat((img_p7_in,p7_fus),1)),1)
        p7_out = self.fpn_convs[0](p7_w[:,0:1,:,:] * img_p7_in + p7_w[:,1:2,:,:] * p7_fus)
        
        p6_up = F.interpolate(p7_out,size=img_p6_in.shape[2:],**self.upsample_cfg)
        p6_w = F.softmax(self.p6_w(torch.cat((img_p6_in,p6_fus,p6_up),1)),1)
        p6_out = self.fpn_convs[1](p6_w[:,0:1,:,:] * img_p6_in + p6_w[:,1:2,:,:] * p6_fus + p6_w[:,2:,:,:] * p6_up )

        p5_up = F.interpolate(p6_out,size=img_p5_in.shape[2:],**self.upsample_cfg)
        p5_w = F.softmax(self.p5_w(torch.cat((img_p5_in,p5_fus,p5_up),1)),1)
        p5_out = self.fpn_convs[2](p5_w[:,0:1,:,:] * img_p5_in + p5_w[:,1:2,:,:] * p5_fus + p5_w[:,2:,:,:] * p5_up )

        p4_up = F.interpolate(p5_out,size=img_p4_in.shape[2:],**self.upsample_cfg)
        p4_w = F.softmax(self.p4_w(torch.cat((img_p4_in,p4_fus,p4_up),1)),1)
        p4_out = self.fpn_convs[3](p4_w[:,0:1,:,:] * img_p4_in + p4_w[:,1:2,:,:] * p4_fus + p4_w[:,2:,:,:] * p4_up )

        p3_up = F.interpolate(p4_out,size=img_p3_in.shape[2:],**self.upsample_cfg)
        p3_w = F.softmax(self.p3_w(torch.cat((img_p3_in,p3_fus,p3_up),1)),1)
        p3_out = self.fpn_convs[4](p3_w[:,0:1,:,:] * img_p3_in + p3_w[:,1:2,:,:] * p3_fus + p3_w[:,2:,:,:] * p3_up )

        outs.append(p3_out)
        outs.append(p4_out)
        outs.append(p5_out)
        outs.append(p6_out)
        outs.append(p7_out)

        return tuple(outs)
