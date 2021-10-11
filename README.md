# DGPNet: Depth-Guided Progressive Network for Object Detection

Here is the official code address of DGPNet based on pytorch. There is only the test code for this framework, and the training code will be released after the paper is received. In addition, the code base will be further improved.

## Get Started

Please see [GETTING_STARTED.md](https://github.com/open-mmlab/mmdetection/blob/v2.11.0/docs/get_started.md) for the basic usage of MMDetection.

## Data Preparation

- You need to download the original training and test images on the [KITTI official website](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
- You need to download the [KINS annotation data](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset) (coco format) and store it in the specified location according to the cfg file.

## Train

```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with KINS dataset in 'data/kins/'
# Note: The training code of this library is temporarily not open source and will be released after the paper is accepted.

./tools/dist_train.sh configs/dgpnet/dgpnet_r50_fpn_2x_kins.py 2
```

## Inference

```python
# For single image visualization
python tools/test.py configs/dgpnet/dgpnet_r50_fpn_2x_kins.py work_dirs/dgpnet_kins.pth --show
```

```python
# For evaluation
./tools/dist_test.sh configs/dgpnet/dgpnet_r50_fpn_2x_kins.py work_dirs/dgpnet_kins.pth 2 --eval bbox
```

## Models

We provide training models of ATSS baseline and DGPNet on the KINS dataset. All models are trained on two GPUs.

Model | Multi-scale training | AP | Link
--- |:---:|:---:|:---:
atss_r50_fpn_2x_kins | No  | 34.1 | [Baidu cloud link](https://pan.baidu.com/s/1oQ4_e3gmGq5gfyp0K3_E3A)(Extraction code: 84sp)
dgpnet_r50_fpn_2x_kins | No  | 37.1 | [Baidu cloud link](https://pan.baidu.com/s/1txLY9jDpX2sO6s9ug_9rrw)(Extraction code: 1hwb)

## Acknowledgement

Thanks MMDetection team for the wonderful open source project!

## Citation

The paper is not yet open.




