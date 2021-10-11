
import tempfile
from os import path as osp
from mmcv import Config, DictAction

from mmdet.datasets.builder import build_dataset

import mmcv
import numpy as np


from mmdet.apis import (async_inference_detector, inference_detector, inference_detector_depth,
                        init_detector, show_result_pyplot)

import os

if __name__ == '__main__':
    kins_class_names = ['cyclist', 'pedestrian', 'car', 'van', 'misc']
    kitti_class_names = ['Pedestrian', 'Cyclist', 'Car']

    res_path = 'atss_kins_out.pkl'
    config = 'configs/kins/atss_r50_fpn_2x_kins.py'
    checkpoint = 'work_dirs/atss_r50_fpn_2x_kins/epoch_18.pth'
    # res_path = 'ours_kins_out.pkl'
    # config = 'configs/pdfnet/test_atss_refine_iou_r50_fpn_2x_kins.py'
    # checkpoint = 'work_dirs/test_atss_refine_iou_r50_fpn_2x_kins/epoch_17.pth'
    device = 'cuda:0'
    data_config_path = 'configs/_base_/datasets/kitti_detection.py'
    cfg = Config.fromfile(data_config_path)
    kitti_dataset = build_dataset(cfg.data.val)
    results = []
    if os.path.exists(res_path):
        results = mmcv.load(res_path)
    else:
        val_txt = 'data/kitti/ImageSets/val.txt'
        img_dir = 'data/kitti/training/image_2'
        img_lists=[]
        with open(val_txt) as f:
            img_lists = f.read().splitlines()
        model = init_detector(config, checkpoint, device=device)
        for img_name in img_lists:
            print(img_name)
            img_path = os.path.join(img_dir,img_name.strip()+".png")
            result = inference_detector(model, img_path)
            temp_res = [result[1],result[0],result[2]]
            results.append(temp_res)
        mmcv.dump(results, res_path)
    kitti_dataset.evaluate(results)





