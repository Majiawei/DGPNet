import os
from pycocotools.coco import COCO
import random

import matplotlib.pyplot as plt
import skimage.io as io
from tqdm import tqdm
import json

###################################################################
# #########################################train
# kins_img_path = "data/kins/training/image_2"
# kins_anno_path = "data/kins/annotations/update_train_2020.json"
#
# bdd_img_path = "data/bdd100k/train"
# bdd_anno_path = "data/bdd100k/anno/bdd100k_labels_images_train_coco.json"
#
# save_dir = "data/mixcar/mix_car_train.json"
#
# kins_coco = COCO(kins_anno_path)
# kins_img_ids = kins_coco.getImgIds()
# kins_img_ids_sel = random.sample(kins_img_ids,5000)
#
# # final output
# counter = 0
# label_count = 0
# attr_dict = {"categories":
#     [
#         {"supercategory": "none", "id": 1, "name": "car"},
#     ]}
# id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
# images = list()
# annotations = list()
#
# for img_id in tqdm(kins_img_ids_sel):
#     img = kins_coco.loadImgs(img_id)[0]
#     annids = kins_coco.getAnnIds(imgIds=img_id)
#     anns = kins_coco.loadAnns(annids)
#     img['id'] = counter
#     img['file_name'] = os.path.join(kins_img_path,img['file_name'])
#     car_ext = False
#     for ann in anns:
#         if ann['category_id'] == 4:
#             car_ext = True
#             annotation = dict()
#             annotation['id'] = label_count
#             annotation["image_id"] = img['id']
#             annotation["iscrowd"] = 0
#             annotation['bbox'] = ann['a_bbox']
#             annotation['area'] = ann['a_area']
#             annotation['segmentation'] = ann['a_segm']
#             annotation['category_id'] = id_dict['car']
#             label_count += 1
#             annotations.append(annotation)
#     if not car_ext:
#         # print("car not ext!")
#         continue
#     counter+=1
#     images.append(img)
# print(len(images))
# # bdd100k
# bdd_coco = COCO(bdd_anno_path)
# bdd_img_ids = bdd_coco.getImgIds()
# bdd_img_ids_sel = random.sample(bdd_img_ids,5000)
# for img_id in tqdm(bdd_img_ids_sel):
#     img = bdd_coco.loadImgs(img_id)[0]
#     annids = bdd_coco.getAnnIds(imgIds=img_id)
#     anns = bdd_coco.loadAnns(annids)
#     img['id'] = counter
#     img['file_name'] = os.path.join(bdd_img_path, img['file_name'])
#     car_ext = False
#     for ann in anns:
#         if ann['category_id'] == 2:
#             car_ext = True
#             annotation = dict()
#             annotation['id'] = label_count
#             annotation["image_id"] = img['id']
#             annotation["iscrowd"] = 0
#             annotation['bbox'] = ann['bbox']
#             annotation['area'] = ann['area']
#             annotation['segmentation'] = ann['segmentation']
#             annotation['category_id'] = id_dict['car']
#             label_count += 1
#             annotations.append(annotation)
#     if not car_ext:
#         # print("car not ext!")
#         continue
#     counter += 1
#     images.append(img)
#
# print(len(images))
# print(len(annotations))
# attr_dict["images"] = images
# attr_dict["annotations"] = annotations
#
# with open(save_dir, "w") as file:
#     json.dump(attr_dict, file)

#########################################################################################
###################################################val
bdd_img_path = "data/bdd100k/val"
bdd_anno_path = "data/bdd100k/anno/bdd100k_labels_images_val_coco.json"
save_dir = "data/mixcar/mix_car_val.json"

bdd_coco = COCO(bdd_anno_path)
bdd_img_ids = bdd_coco.getImgIds()
bdd_img_ids_sel = random.sample(bdd_img_ids,3000)

# final output
counter = 0
label_count = 0
attr_dict = {"categories":
    [
        {"supercategory": "none", "id": 1, "name": "car"},
    ]}
id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
images = list()
annotations = list()
for img_id in tqdm(bdd_img_ids_sel):
    img = bdd_coco.loadImgs(img_id)[0]
    annids = bdd_coco.getAnnIds(imgIds=img_id)
    anns = bdd_coco.loadAnns(annids)
    img['id'] = counter
    img['file_name'] = os.path.join(bdd_img_path, img['file_name'])
    car_ext = False
    for ann in anns:
        if ann['category_id'] == 2:
            car_ext = True
            annotation = dict()
            annotation['id'] = label_count
            annotation["image_id"] = img['id']
            annotation["iscrowd"] = 0
            annotation['bbox'] = ann['bbox']
            annotation['area'] = ann['area']
            annotation['segmentation'] = ann['segmentation']
            annotation['category_id'] = id_dict['car']
            label_count += 1
            annotations.append(annotation)
    if not car_ext:
        # print("car not ext!")
        continue
    counter += 1
    images.append(img)

print(len(images))
print(len(annotations))
attr_dict["images"] = images
attr_dict["annotations"] = annotations
with open(save_dir, "w") as file:
    json.dump(attr_dict, file)


