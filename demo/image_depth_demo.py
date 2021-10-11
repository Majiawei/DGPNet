import asyncio
import os
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector, inference_detector_depth,
                        init_detector, show_result_pyplot)
import numpy as np
import cv2
import matplotlib.pyplot as plt

# display the pic after detecting. 2018.04.25
def showPicResult(image,xml_boxes,det_boxes):
    # cv2.namedWindow('image_detector', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image_detector', 1024,768);
    img = cv2.imread(image)
    img_h, img_w, img_c = img.shape;
    for xml_box in xml_boxes:
        x1=xml_box[0]
        y1=xml_box[1]
        x2=xml_box[2]
        y2=xml_box[3]
        # im = cv2.imread(out_img)
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2) # green
        pass
    for det_box in det_boxes:
        x1=det_box[0]
        y1=det_box[1]
        x2=det_box[2]
        y2=det_box[3]
        # im = cv2.imread(out_img)
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2) # red
        pass
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imshow('image_detector', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    in_h = yB - yA
    in_w = xB - xA

    # compute the area of intersection rectangle
    # 交集
    interArea = 0 if in_h<0 or in_w<0 else in_h*in_w

    # compute the area of both the prediction and ground-truth rectangles
    # 并集
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
    boxBArea = (boxB[2] - boxB[0] ) * (boxB[3] - boxB[1] )

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


if __name__ == '__main__':
    classes_all = ['Pedestrian', 'Cyclist', 'Car']
    classes_need = ['Car']

    in_xml = 0
    notin_xml = 0
    notin_det = 0
    pic_sum = 0
    a = 1.0
    tp = 0
    fp = 0
    fn = 0
    matchedSum = 0
    numGlobalCareGt = 0
    numGlobalCareDet = 0
    val_txt = 'data/kitti/ImageSets/val.txt'
    img_lists=[]
    with open(val_txt) as f:
        img_lists = f.readlines()

    img_dir = 'data/kitti/training/image_2'
    depth_dir = 'data/kitti/training/depth_2'
    anno_dir = 'data/kitti/training/label_2'
    config = 'configs/retina_mf_net/mtretinan_r50_fpn_2x_kitti.py'
    checkpoint = 'work_dirs/mtretinan_r50_fpn_2x_kitti/latest.pth'
    device = 'cuda:0'
    score_thr = 0.5
    model = init_detector(config, checkpoint, device=device)
    print(model)


    for img_name in img_lists:
        anno_boxes = []
        det_boxes = []
        img_path = os.path.join(img_dir,img_name.strip()+".png")
        print(img_path)
        depth_path = os.path.join(depth_dir,img_name.strip()+".png")
        anno_path = os.path.join(anno_dir,img_name.strip()+".txt")
        with open(anno_path) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line[0] in classes_need:
                    anno_boxes.append([float(line[4]),float(line[5]),float(line[6]),float(line[7])])
        # test a single image
        result = inference_detector_depth(model, img_path, depth_path)
        # show the results
        # show_result_pyplot(model, img_path, result, score_thr=score_thr)
        classes_idx = classes_all.index(classes_need[0])
        result_need = result[classes_idx]
        # print(result)
        bboxes=result_need
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        for j in range(bboxes.shape[0]):
            bbox = bboxes[j, :]
            # x1 = int(bbox[0])
            # y1 = int(bbox[1])
            # x2 = int(bbox[2])
            # y2 = int(bbox[3])
            # score = float(bbox[4])
            det_boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])
        showPicResult(img_path, anno_boxes, det_boxes)
        ################################################calculate f1-score
        detMatched = 0
        gtPols = anno_boxes
        detPols = det_boxes
        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            # 匹配标记，0未匹配，1已匹配
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            # 二重循环计算IoU矩阵
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = bb_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):

                    # 若标签和预测框均为匹配，且均不忽略，则判断IoU
                    if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0:
                        # 若IoU大于某个阈值，则匹配成功
                        if iouMat[gtNum, detNum] > 0.5:
                            # 更新匹配标记
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            # 匹配数量+1
                            detMatched += 1
                            # 增加匹配对
                            # pairs.append({'gt': gtNum, 'det': detNum})
                            # 记录成功匹配的预测框index
                            # detMatchedNums.append(detNum)

            # 计算有效框的数量)
        numGtCare = len(gtPols)
        numDetCare = len(detPols)
        # 将该图片的计数记录到全局总数中
        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

    # 计算全部图片的结果
    # 计算全部图片的召回率
    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    # 计算全部图片的准确率
    methodPrecision = 0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    # 计算全部图片的hmean（即F-Score）
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
                methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}
    print(methodMetrics)

