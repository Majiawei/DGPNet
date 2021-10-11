import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, xavier_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,build_anchor_generator,build_bbox_coder,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead

from mmdet.utils import depth_utils
from mmdet.utils import balancer
import torch.nn.functional as F
import warnings
import numpy as np
from mmcv.ops import DeformConv2d, batched_nms
from mmdet.models.utils import build_positional_encoding

import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt

class AdaptiveConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=3,
                 groups=1,
                 bias=False,
                 type='dilation'):
        super(AdaptiveConv, self).__init__()
        assert type in ['offset', 'dilation']
        self.adapt_type = type

        assert kernel_size == 3, 'Adaptive conv only supports kernels 3'
        if self.adapt_type == 'offset':
            assert stride == 1 and padding == 1 and groups == 1, \
                'Adaptive conv offset mode only supports padding: {1}, ' \
                f'stride: {1}, groups: {1}'
            self.conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=dilation,
                dilation=dilation)

    def init_weights(self):
        """Init weights."""
        normal_init(self.conv, std=0.01)

    def forward(self, x, offset):
        """Forward function."""
        if self.adapt_type == 'offset':
            N, _, H, W = x.shape
            assert offset is not None
            assert H * W == offset.shape[1]
            # reshape [N, NA, 18] to (N, 18, H, W)
            offset = offset.permute(0, 2, 1).reshape(N, -1, H, W)
            offset = offset.contiguous()
            x = self.conv(x, offset)
        else:
            assert offset is None
            x = self.conv(x)
        return x

@HEADS.register_module()
class RefineATSSIOUHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 ##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                 fuse_method='mul',  # cat weight(parameter dynamic,static) gate(spatial) pecat
                 ###################################two stage
                 stage1_anchor_generator=dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     strides=[8, 16, 32, 64, 128]),
                 stage1_bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     target_means=[.0, .0, .0, .0],
                     target_stds=[1.0, 1.0, 1.0, 1.0]),
                 stage1_adapt_cfg=dict(type='dilation', dilation=3),
                 stage2_adapt_cfg=dict(type='offset'),
                 loc_filter_thr=0,
                 stage1_loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=True,
                     loss_weight=1.0),
                 stage1_loss_bbox=dict(type='L1Loss', loss_weight=1.0),
                 stage1_loss_centerness=dict(
                     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 stage1_reg_decoded_bbox=False,
                 #################################################
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RefineATSSIOUHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.loss_centerness = build_loss(loss_centerness)
        ##$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.fuse_method = fuse_method
        self.ada_conv_list = nn.ModuleList()
        for i in range(5):
            self.ada_conv_list.append(
                ConvModule(self.feat_channels, self.feat_channels, 3, stride=1, padding=1, norm_cfg=self.norm_cfg)
            )
        ###########################################refine
        self.num_anchors=1
        self._init_layers()
        self.stage1_adapt_cfg=stage1_adapt_cfg
        self.stage2_adapt_cfg=stage2_adapt_cfg
        self.stage1_anchor_strides = stage1_anchor_generator['strides']
        self.stage1_anchor_scales = stage1_anchor_generator['scales']
        self.stage1_anchor_generator = build_anchor_generator(stage1_anchor_generator)
        self.stage1_bbox_coder = build_bbox_coder(stage1_bbox_coder)
        self.stage1_loss_cls = build_loss(stage1_loss_cls)
        self.stage1_loss_bbox = build_loss(stage1_loss_bbox)
        self.stage1_loss_centerness = build_loss(stage1_loss_centerness)
        self.stage1_reg_decoded_bbox = stage1_reg_decoded_bbox

        if self.train_cfg:
            self.stage1_assigner = build_assigner(self.train_cfg.stage1_assigner)
            if hasattr(self.train_cfg, 'stage1_sampler'):
                stage1_sampler_cfg = self.train_cfg.stage1_sampler
            else:
                stage1_sampler_cfg = dict(type='PseudoSampler')
            self.stage1_sampler = build_sampler(stage1_sampler_cfg, context=self)

        self.conv_loc = nn.Conv2d(self.feat_channels, self.num_anchors, 3, padding=1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 3, padding=1)
        # self.conv_centerness = nn.Conv2d(self.feat_channels, self.num_anchors * 1, 3, padding=1)
        self.stage1_scales = nn.ModuleList([Scale(1.0) for _ in self.stage1_anchor_generator.strides])

        self.relu_cls = nn.ReLU(inplace=True)
        self.relu_reg = nn.ReLU(inplace=True)
        self.relu_stage1 = nn.ReLU(inplace=True)
        self.feature_adaption_cls = AdaptiveConv(self.in_channels, self.feat_channels, **self.stage2_adapt_cfg)
        self.feature_adaption_reg = AdaptiveConv(self.in_channels, self.feat_channels, **self.stage2_adapt_cfg)
        self.stage1_feature_adaption = AdaptiveConv(self.in_channels, self.feat_channels, **self.stage1_adapt_cfg)
        self.loc_filter_thr = loc_filter_thr
        ##########################################################

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.atss_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        # self.atss_centerness = nn.Conv2d(
        #     self.feat_channels, self.num_anchors * 1, 3, padding=1)
        # self.scales = nn.ModuleList(
        #     [Scale(1.0) for _ in self.anchor_generator.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)
        normal_init(self.atss_reg, std=0.01)
        # normal_init(self.atss_centerness, std=0.01)
        ########################################
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)
        self.feature_adaption_cls.init_weights()
        self.feature_adaption_reg.init_weights()
        self.stage1_feature_adaption.init_weights()
        #############################################
        for m in self.ada_conv_list:
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward_single_stage1(self, x, scale):
        x = self.relu_stage1(self.stage1_feature_adaption(x,None))
        loc_pred = self.conv_loc(x)
        shape_pred = scale(self.conv_shape(x)).float()
        # centerness_pred = self.conv_centerness(x)
        return loc_pred, shape_pred
    def forward_single_stage2(self, x , offset):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_feat = self.relu_cls(self.feature_adaption_cls(cls_feat, offset))
        reg_feat = self.relu_reg(self.feature_adaption_reg(reg_feat, offset))

        cls_score = self.atss_cls(cls_feat)
        # we just follow atss, not apply exp in bbox_pred
        bbox_pred = self.atss_reg(reg_feat)
        # centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred

    def fuse_single(self, det_feat, depth_feat):
        if self.fuse_method == 'gate':
            return self.gate_mod(det_feat, depth_feat)
        elif self.fuse_method == 'cat':
            return self.depth_img_gate_op(torch.cat([det_feat, depth_feat], dim=1))
        elif self.fuse_method == 'add':
            return det_feat + depth_feat
        elif self.fuse_method == 'mul':
            # return det_feat * self.depth_trans_conv(depth_feat)
            return det_feat * depth_feat
        elif self.fuse_method == 'aff':
            return self.aff_mod(det_feat, depth_feat)
        else:
            return None

    def forward_common(self,
                       x1, x2,
                       img_metas
                       ):
        det_feats = x1
        depth_feats = x2
        depth_feats_res, d_res_list = depth_feats
        ##########################depth fusion
        if self.fuse_method is not None:
            fused_feats = []
            scaled_feats = []
            for i in range(len(det_feats)):
                scaled_feat = F.interpolate(d_res_list, size=det_feats[i].shape[2:], mode='bilinear',align_corners=False)
                scaled_feat = self.ada_conv_list[i](scaled_feat)
                fuse_feat = self.fuse_single(det_feats[i], scaled_feat)
                fused_feats.append(fuse_feat)
                scaled_feats.append(scaled_feat)
        else:
            fused_feats = det_feats
        ###########################two stage
        featmap_sizes = [featmap.size()[-2:] for featmap in fused_feats]
        device = fused_feats[0].device

        stage1_anchor_list, stage1_valid_flag_list, stage2_valid_flag_list = self.stage1_get_anchors(featmap_sizes, img_metas, device=device)

        stage1_cls_pred, stage1_bbox_pred = multi_apply(self.forward_single_stage1, fused_feats, self.stage1_scales)

        stage2_anchor_list = self.refine_bboxes(stage1_anchor_list, stage1_bbox_pred, img_metas)

        offset_list = self.anchor_offset(stage2_anchor_list, self.stage1_anchor_strides, featmap_sizes)

        stage2_cls_pred, stage2_bbox_pred = multi_apply(self.forward_single_stage2, fused_feats, offset_list)

        outs = (stage1_anchor_list, stage1_valid_flag_list, stage1_cls_pred, stage1_bbox_pred, stage2_anchor_list, stage2_valid_flag_list, stage2_cls_pred, stage2_bbox_pred, depth_feats_res, featmap_sizes)
        return outs


    def stage1_get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        num_imgs = len(img_metas)

        multi_level_anchors = self.stage1_anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]


        valid_flag_list = []
        valid_flag_list1 = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.stage1_anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            multi_level_flags1 = self.stage1_anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
            valid_flag_list1.append(multi_level_flags1)

        return anchor_list, valid_flag_list, valid_flag_list1
    def anchor_offset(self, anchor_list, anchor_strides, featmap_sizes):
        def _shape_offset(anchors, stride, ks=3, dilation=1):

            assert ks == 3 and dilation == 1
            pad = (ks - 1) // 2
            idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
            yy, xx = torch.meshgrid(idx, idx)  # return order matters
            xx = xx.reshape(-1)
            yy = yy.reshape(-1)
            w = (anchors[:, 2] - anchors[:, 0]) / stride
            h = (anchors[:, 3] - anchors[:, 1]) / stride
            w = w / (ks - 1) - dilation
            h = h / (ks - 1) - dilation
            offset_x = w[:, None] * xx  # (NA, ks**2)
            offset_y = h[:, None] * yy  # (NA, ks**2)
            return offset_x, offset_y

        def _ctr_offset(anchors, stride, featmap_size):
            feat_h, feat_w = featmap_size
            assert len(anchors) == feat_h * feat_w

            x = (anchors[:, 0] + anchors[:, 2]) * 0.5
            y = (anchors[:, 1] + anchors[:, 3]) * 0.5
            # compute centers on feature map
            x = x / stride
            y = y / stride
            # compute predefine centers
            xx = torch.arange(0, feat_w, device=anchors.device)
            yy = torch.arange(0, feat_h, device=anchors.device)
            yy, xx = torch.meshgrid(yy, xx)
            xx = xx.reshape(-1).type_as(x)
            yy = yy.reshape(-1).type_as(y)

            offset_x = x - xx  # (NA, )
            offset_y = y - yy  # (NA, )
            return offset_x, offset_y

        num_imgs = len(anchor_list)
        num_lvls = len(anchor_list[0])
        dtype = anchor_list[0][0].dtype
        device = anchor_list[0][0].device
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

        offset_list = []
        for i in range(num_imgs):
            mlvl_offset = []
            for lvl in range(num_lvls):
                c_offset_x, c_offset_y = _ctr_offset(anchor_list[i][lvl],
                                                     anchor_strides[lvl],
                                                     featmap_sizes[lvl])
                s_offset_x, s_offset_y = _shape_offset(anchor_list[i][lvl],
                                                       anchor_strides[lvl])

                # offset = ctr_offset + shape_offset
                offset_x = s_offset_x + c_offset_x[:, None]
                offset_y = s_offset_y + c_offset_y[:, None]

                # offset order (y0, x0, y1, x2, .., y8, x8, y9, x9)
                offset = torch.stack([offset_y, offset_x], dim=-1)
                offset = offset.reshape(offset.size(0), -1)  # [NA, 2*ks**2]
                mlvl_offset.append(offset)
            offset_list.append(torch.cat(mlvl_offset))  # [totalNA, 2*ks**2]
        offset_list = images_to_levels(offset_list, num_level_anchors)
        return offset_list

    def refine_bboxes(self, anchor_list, bbox_preds, img_metas):

        num_levels = len(bbox_preds)
        new_anchor_list = []
        for img_id in range(len(img_metas)):
            mlvl_anchors = []
            for i in range(num_levels):
                bbox_pred = bbox_preds[i][img_id].detach()
                bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
                img_shape = img_metas[img_id]['img_shape']
                bboxes = self.stage1_bbox_coder.decode(anchor_list[img_id][i],
                                                bbox_pred, img_shape)
                mlvl_anchors.append(bboxes)
            new_anchor_list.append(mlvl_anchors)
        return new_anchor_list


    def stage1_loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, cls_num_total_samples, reg_num_total_samples):
        # classification loss
        # labels = labels.reshape([cls_score.shape[0],cls_score.shape[2],cls_score.shape[3]])
        # label_weights = label_weights.reshape([cls_score.shape[0],cls_score.shape[2],cls_score.shape[3]])
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).contiguous().reshape(-1)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        # centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # classification loss
        loss_cls = self.stage1_loss_cls(cls_score, labels, label_weights, avg_factor=cls_num_total_samples)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = 1
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            # pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(
                pos_anchors, pos_bbox_targets)
            pos_decode_bbox_pred = self.stage1_bbox_coder.decode(
                pos_anchors, pos_bbox_pred)
            pos_decode_bbox_targets = self.stage1_bbox_coder.decode(
                pos_anchors, pos_bbox_targets)

            # regression loss
            loss_bbox = self.stage1_loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=centerness_targets,
                avg_factor=1.0)


        else:
            loss_bbox = bbox_pred.sum() * 0
            # loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.)

        return loss_cls, loss_bbox, centerness_targets.sum()



    def _region_targets_single(self,
                               flat_anchors,
                               valid_flags,
                               num_level_anchors,
                               gt_bboxes,
                               gt_bboxes_ignore,
                               gt_labels,
                               img_meta,
                               label_channels=1,
                               unmap_outputs=True):

        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)  # [11657]
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]  # [11657,4]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)  # [8736, 2184, 546, 147, 44]
        assign_result = self.stage1_assigner.assign(anchors, num_level_anchors_inside,
                                                    gt_bboxes, gt_bboxes_ignore,
                                                    gt_labels)

        sampling_result = self.stage1_sampler.sample(assign_result, anchors,
                                                     gt_bboxes)

        num_valid_anchors = anchors.shape[0]  # 11657
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  1,
                                  dtype=torch.long)
        # labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)


        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.stage1_reg_decoded_bbox:
                pos_bbox_targets = self.stage1_bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def region_targets(self,
                       anchor_list,
                       valid_flag_list,
                       gt_bboxes_list,
                       img_metas,
                       gt_bboxes_ignore_list=None,
                       gt_labels_list=None,
                       label_channels=1,
                       unmap_outputs=True):

        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._region_targets_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def centerness_target(self, anchors, bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        gts = self.bbox_coder.decode(anchors, bbox_targets)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        centerness = torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) *
            (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def forward_test(self,
                     x1, x2,
                     img_metas,
                     cfg=None,
                     rescale=False,
                     with_nms=True):
        ################################two stage
        (stage1_anchor_list, stage1_valid_flag_list, stage1_cls_pred, stage1_bbox_pred, stage2_anchor_list, stage2_valid_flag_list, stage2_cls_pred, stage2_bbox_pred, depth_feats_res, featmap_sizes) = self.forward_common(x1, x2, img_metas)

        """Simple forward test function."""
        # get stage1 mask
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        loc_mask_list = [] # per img per lvl
        for img_id in range(num_imgs):
            multi_level_loc_mask = []
            for i in range(num_levels):
                loc_pred = stage1_cls_pred[i][img_id] # chw
                # softmax
                # loc_pred = loc_pred.softmax(dim=0).detach()[1]
                # sigmoid
                loc_pred = loc_pred.sigmoid().detach()
                loc_mask = (1-loc_pred) >= self.loc_filter_thr
                mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_anchors)
                mask = mask.contiguous().view(-1)
                multi_level_loc_mask.append(mask)
            loc_mask_list.append(multi_level_loc_mask)

        result_list = []
        for img_id in range(num_imgs):
            cls_score_list = [
                stage2_cls_pred[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                stage2_bbox_pred[i][img_id].detach() for i in range(num_levels)
            ]
            guided_anchor_list = [
                stage2_anchor_list[img_id][i].detach() for i in range(num_levels)
            ]
            loc_mask_list = [
                loc_mask_list[img_id][i].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                guided_anchor_list,
                                                loc_mask_list, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           mlvl_masks,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors, mask in zip(cls_scores, bbox_preds,
                                                       mlvl_anchors,
                                                       mlvl_masks):
            #############per lvl
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            # reshape scores and bbox_pred
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            anchors = anchors[mask, :]
            scores = scores[mask, :]
            bbox_pred = bbox_pred[mask, :]
            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        # multi class NMS
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels


    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
