import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    # config = "configs/kins/atss_r50_fpn_2x_kins.py"
    # checkpoint = "work_dirs/atss_r50_fpn_2x_kins/epoch_18.pth"
    config = 'configs/pdfnet/test_mtatss_r50_fpn_2x_kins_mul.py'
    checkpoint = 'work_dirs/test_mtatss_r50_fpn_2x_kins.mulfuse/epoch_17.pth'
    # config = 'configs/pdfnet/test_atss_refine_iou_r50_fpn_2x_kins.py'
    # checkpoint = 'work_dirs/test_atss_refine_iou_r50_fpn_2x_kins/epoch_17.pth'
    model = init_detector(config, checkpoint, device=args.device)
    # test a single image
    img_path = 'data/bdd100k/val/b1cd1e94-26dd524f.jpg'
    result = inference_detector(model, img_path)
    # show the results
    show_result_pyplot(model, img_path, result, title="ours_"+img_path.split("/")[-1][:-4], score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)