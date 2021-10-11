import mmcv

out_json = dict()
out_json['images'] = []
out_json['categories'] = []
out_json['annotations'] = []

# anns = mmcv.load("data/kins/annotations/update_test_2020_coco.json")
# anns1 = mmcv.load("/home/yzzc/Work/mjw20201021/Auto_dataset/cityscapes/annotations/instancesonly_filtered_gtFine_test.json")
# print(1)
anns = mmcv.load("data/kins/annotations/update_test_2020.json")
out_json_name = "data/kins/annotations/update_test_2020_coco.json"
#
new_class_list = ['cyclist', 'pedestrian', 'car', 'van', 'misc']
old_class_list = ['cyclist', 'pedestrian', 'rider', 'car', 'tram', 'truck', 'van', 'misc']
imgs_info = anns['images']
anns_info = anns["annotations"]
cat_info = anns["categories"]
for cat in cat_info:
    if cat['name'] in new_class_list:
        cat['id'] = new_class_list.index(cat['name'])+1
        out_json['categories'].append(cat)
        print(cat)
ann_id=0
for ann in anns_info:
    # if ann["category_id"]-1>7 or ann["category_id"]-1<0:
    #     continue
    if old_class_list[ann["category_id"]-1] in new_class_list:
        ann["id"]=ann_id
        ann["category_id"]=new_class_list.index(old_class_list[ann["category_id"]-1])+1
        ann["bbox"]=ann["a_bbox"]
        ann["iscrowd"]=0
        ann["area"] = ann["a_area"]
        ann["segmentation"]=ann["a_segm"]
        # print(ann["bbox"])
        out_json['annotations'].append(ann)
        ann_id+=1
out_json['images']=imgs_info
# out_json['categories']=cat_info
# out_json['annotations']=anns_info
mmcv.dump(out_json, out_json_name)