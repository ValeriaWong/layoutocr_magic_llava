from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
from mmdet.evaluation import get_classes
from mmdet.apis import init_detector, inference_detector
import mmcv
from mmcv import imshow_det_bboxes
import matplotlib.pyplot as plt
# 指定模型配置文件及权重文件路径
config_file = 'tool/mmdetection/work_dirs/screen_250_data_augment/detr_r50_8xb2-150e_250_screenshot_coco_copy.py'
checkpoint_file = 'tool/mmdetection/work_dirs/screen_250_data_augment/best_coco_bbox_mAP_epoch_149.pth'

# 通过配置和权重初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

COCO_CLASSES = [
    'avator',
    'message',
    'header',
    'other',
    'transfer',
    'file',
    'timestamp',
    'image',
    'meme',
    'comment',
    'translate',
    'nickname',
    'unpassed_messeage',
    'voice',
    'withdraw',
]

# 测试单张图片并显示结果
img = '/root/wangqun/layoutocr_magic_llava/Batch/5a0e7c5b4eacab366851146d_image_0.jpg'
result = inference_detector(model, img)
# print(result)
# 显示图片及其检测的边界框和类别
# show_result_pyplot(model, img, result, score_thr=0.3)  # 设置阈值
# img = mmcv.imread(img)
# imshow_det_bboxes(
#     img,
#     result,
#     class_names=COCO_CLASSES,
#     score_thr=0.3,
#     show=True,
#     wait_time=0,
#     out_file=None
#     labels=label
# )
# 检测结果
# 获取预测的实例数据
pred_instances = result.pred_instances

# 提取边界框、标签和得分
bboxes = pred_instances.bboxes.cpu().numpy()  # 将tensor转移到CPU并转为NumPy数组
labels = pred_instances.labels.cpu().numpy()
scores = pred_instances.scores.cpu().numpy()

# 检查边界框数组是否非空
if len(bboxes) > 0:
    # 遍历所有边界框和相应的标签与得分
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > 0.5:
            x_min, y_min, x_max, y_max = bbox
            print(f"类别 {label}, {COCO_CLASSES[label]}:")
            print(
                f"  边界框 [x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}], 得分：{score:.4f}")
    print("")

