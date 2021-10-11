import os
import numpy as np
import mmcv
import cv2
import skimage
import skimage.transform
import matplotlib.pyplot as plt

# info_path = '/home/yzzc/Work/mjw20201021/Auto_dataset/BDD100k/Labels/bdd100k_labels_images_val.json'
# kitti_infos = mmcv.load(info_path)
# print(len(kitti_infos))

# img_mm = mmcv.imread("/home/yzzc/Work/mjw20201021/experiment/mmdetection3d/data/kitti/training/depth_2/000000.png","unchanged")
#########################################################################3
# with open("../data/kitti/ImageSets/train.txt","r") as f:
#     train_list = f.readlines()
# print(len(train_list))
# # train_list = mmcv.load("../data/kitti/ImageSets/train.txt")
# depth_path = '/home/yzbj10/Work/mjw_3d/datasets/kitti/depth_2'
# path_list = os.listdir(depth_path)
# sum = 0
# sum_pix = 0
# mean = 0
# std = 0
# max_list = []
# for img_name in path_list:
#     img_path = os.path.join(depth_path, img_name)
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
#     img /= 256.0
#     # img = img[:, :, np.newaxis]
#     # img = np.tile(img, (1, 1, 3))
#     sum += img.sum()
#     max_list.append(img.max())
#     sum_pix += img.shape[0]*img.shape[1]
#     print(img_name)
# mean = sum/sum_pix
# max_ = min(max_list)
# min_ = max(max_list)
# print("mean:",mean)
# print(min_,max_)
#
# sum = 0
# for img_name in train_list:
#     img_path = img_path = os.path.join(depth_path,img_name.strip()+".png")
#     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     img = img[:, :, np.newaxis]
#     img = np.tile(img, (1, 1, 3))
#     sum+=pow((img-mean),2).sum()
#     # sum_pix+=img.shape[0]*img.shape[1]
# std = np.sqrt(sum/sum_pix)
# print("std:",std)
###############################################333
# file:///home/yzzc/Work/mjw20201021/Auto_dataset/cityscapes/disparity/train/aachen/aachen_000010_000019_disparity.png
# file:///home/yzzc/Work/mjw20201021/Auto_dataset/cityscapes/leftImg8bit/train/aachen/aachen_000061_000019_leftImg8bit.png
# file:///home/yzzc/Work/mjw20201021/experiment/AdaBins/kitti_output/000021.png
img_mm = mmcv.imread("data/cityscapes/disparity/train/aachen/aachen_000061_000019_disparity.png","unchanged")
# img_mm = mmcv.imread("data/kins/training/depth_2/000008.png","unchanged")
# img_mm1 = mmcv.imread("/home/yzbj10/Work/mjw_3d/mmdetection/data/kins/training/depth_2/000008.png","unchanged")
# Image.fromarray(final).save(save_path)
# plt.imsave('depth_vis.png', img_mm1.astype('uint16'), cmap='plasma_r')
# print(1)
depth = img_mm.astype(np.float32)
depth /= 256.0
# plt.imsave('depth_vis.png', img_mm1.astype('uint16'), cmap='plasma_r')
depth[depth>40] = depth[depth>40]-30.0
depth[depth>60] = depth[depth>60]-30.0
# depth = 0.2*depth
# depth = depth-(np.exp(depth/28.5)-1)

depth = np.clip(depth,a_min=1.0,a_max=depth.max())
final = (depth*256.0).astype(np.uint16)
# final = cv2.resize(final, (1242,375)).astype('uint16')
basename = "res.png"
out_dir = 'work_dirs'
save_path = os.path.join(out_dir, basename)
# # Image.fromarray(final).save(save_path)
plt.imsave(save_path,np.log10(final),cmap='plasma_r')
# img_mm = skimage.transform.downscale_local_mean(image=depth,factors=(4,4))
#
# depth1 = img_mm1.astype(np.float32)
# depth1 /= 256.0
# img_mm1 = skimage.transform.downscale_local_mean(image=depth1,factors=(4,4))
# print(img_mm.shape)