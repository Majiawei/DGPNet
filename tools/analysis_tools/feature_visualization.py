import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from mmcv import Config
from mmdet.datasets.builder import build_dataset
from mmdet.datasets.pipelines import Compose


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def retrieve_data_cfg(config_path, skip_type, cfg_options=None):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg

def draw_feature_map(features,save_dir = 'feature_map',name = "vis"):
    i=0
    # img = cv2.imread('data/kins/testing/image_2/000002.png')
    img_path = 'data/kins/training/image_2/000008.png'
    img_name = img_path.split("/")[-1][:-4]
    skip_type = ['DefaultFormatBundle', 'Normalize', 'Collect', 'RandomFlip','LoadAnnotations']
    config_kins = 'configs/kins/atss_r50_fpn_2x_kins.py'
    kins_cfg = retrieve_data_cfg(config_kins, skip_type)
    test_pip = Compose(kins_cfg.data.train.pipeline)
    # data = dict(img=img)
    data = dict(img_info=dict(filename=img_path), img_prefix=None)
    img =test_pip(data)['img']


    if isinstance(features,torch.Tensor):
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # θΏιηh,wζηζ―δ½ ζ³θ¦ζηΉεΎεΎresizeζε€ε€§ηε°Ίε―Έ
            # heatmap = cv2.resize(heatmap, (h, w))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # δΈι’θΏθ‘ε°η­εεΎθ½¬ζ’δΈΊRGBζ ΌεΌ οΌε¦ζζ³¨ιζε°±ζ―η°εΊ¦εΎ
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
    else:
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            for heatmap in heatmaps:
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # ε°η­εεΎηε€§ε°θ°ζ΄δΈΊδΈεε§εΎεηΈε
                heatmap = np.uint8(255 * (1-heatmap))  # ε°η­εεΎθ½¬ζ’δΈΊRGBζ ΌεΌ
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.5 + img*0.3
                # superimposed_img = heatmap
                # plt.imshow(superimposed_img)
                # plt.imshow(superimposed_img,cmap='plasma_r')
                # plt.show()
                # δΈι’θΏδΊζ―ε―ΉηΉεΎεΎθΏθ‘δΏε­οΌδ½Ώη¨ζΆεζΆζ³¨ι
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(save_dir,name + "_" + img_name + "_" + str(i) +'.png'), superimposed_img)
                i=i+1
