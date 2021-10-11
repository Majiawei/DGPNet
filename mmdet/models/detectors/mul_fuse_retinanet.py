import torch
import torch.nn as nn

from mmdet.core import bbox2result,multi_apply
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector

from mmcv.runner import auto_fp16
from mmcv.utils import print_log


@DETECTORS.register_module()
class MultiStageFuseRetinaNet(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 depth_backbone=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 depth_pretrained=None):
        super(MultiStageFuseRetinaNet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.depth_backbone = build_backbone(depth_backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if depth_pretrained==None:
            self.init_weights(pretrained=pretrained,depth_pretrained=pretrained)
        else:
            self.init_weights(pretrained=pretrained,depth_pretrained=depth_pretrained)

    def init_weights(self, pretrained=None,depth_pretrained=None):
        super(MultiStageFuseRetinaNet, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.depth_backbone.init_weights(pretrained=depth_pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img, depth):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        depth = self.depth_backbone(depth)
        if self.with_neck: # fpn
            x = self.neck(x, depth)
        return x
    def extract_feats(self, imgs, depths, img_metas):
        if imgs is None:
            imgs = [None] * len(img_metas)
        feats = multi_apply(self.extract_feat, imgs, depths,
                                           img_metas)
        return feats

    def forward_dummy(self, img, depth):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img, depth)
        outs = self.bbox_head(x)
        return outs

    @auto_fp16(apply_to=('img', 'depth'))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      img,
                      depth,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img,depth)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses
    
    def forward_test(self, img, depth, img_metas, **kwargs):
        imgs = img
        depths = depth
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        if num_augs == 1:
            imgs = [imgs] if imgs is None else imgs
            return self.simple_test(imgs[0],depths[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, depths, img_metas, **kwargs)

    def simple_test(self, img, depth, img_metas, rescale=False):
        x = self.extract_feat(img,depth)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, imgs, depths, img_metas, rescale=False):
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs, depths)

        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
    
    
