import os
from pycocotools.coco import COCO
import random

import matplotlib.pyplot as plt
import skimage.io as io
from tqdm import tqdm
import json

# img_path = "data/kins/training/image_2"
anno_path = "data/mixcar/mix_car_val.json"

coco = COCO(anno_path)
img_ids = coco.getImgIds()
img_ids.reverse()
for img_id in tqdm(img_ids):
    img = coco.loadImgs(img_id)[0]
    annids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(annids)

    I = io.imread(img["file_name"])
    plt.imshow(I)
    plt.axis('off')
    coco.showAnns(anns,draw_bbox=True)
    plt.show()


