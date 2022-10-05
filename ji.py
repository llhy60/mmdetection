import os
import sys
import json
import time
import torch
import numpy as np
from glob import glob
from typing import List
from dataclasses import dataclass


@dataclass
class CFG:
    device: str
    target: List[str]
    score_thr: float
    yolox_path: str
    config: str
    checkpoint: str

arg = CFG(device='cuda:0', 
          target=['front_no_wear', 'front_under_nose_wear', 'front_under_mouth_wear', 
                  'side_no_wear', 'side_under_nose_wear', 'side_under_mouth_wear'],
          score_thr=0.3,
          yolox_path='/project/train/src_repo/mmlab/mmdetection_llh',
          config='/project/train/src_repo/mmlab/mmdetection_llh/configs/yolox/exp1_yolox_s_mask_baseline.py',
          checkpoint='/project/train/models/exp1_yolox_s_mask_baseline/latest.pth'
         )
sys.path.insert(1, arg.yolox_path)
from mmdet.apis import (async_inference_detector, inference_detector, init_detector)

def init():
    # Initialize
    model = init_detector(arg.config, arg.checkpoint, device=arg.device)
    return model

@torch.no_grad()
def process_image(model, input_image=None, args=None, **kwargs):
    classes = list(model.CLASSES)
    bbox_result = inference_detector(model, input_image)

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]

    inds = scores > arg.score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    
    fake_result = {}

    fake_result["algorithm_data"] = {
       "is_alert": False,
       "target_count": 0,
       "target_info": []
   }
    fake_result["model_data"] = {"objects": []}
    # Process detections
    cnt = 0
    for cls, bbox in zip(labels, bboxes):
        name, bbox, conf = classes[cls], bbox[:4], float(bbox[4])
        x1, y1, x2, y2 = bbox.astype(np.int32)
        fake_result["model_data"]['objects'].append({
                "x": x1,
                "y": y1,
                "height": y2-y1,
                "width": x2-x1,
                "confidence":float(conf),
                "name":name
            }
        )
        if name in arg.target:
            cnt += 1
            fake_result["algorithm_data"]["target_info"].append({
                "x": x1,
                "y": y1,
                "height": y2-y1,
                "width": x2-x1,
                "confidence":float(conf),
                "name":name
            }
        )
    if cnt:
        fake_result["algorithm_data"]["is_alert"] = True
        fake_result["algorithm_data"]["target_count"] = cnt
    return json.dumps(fake_result, indent = 4)


if __name__ == '__main__':
    # Test API
    image_names = glob('/home/data/*/*.jpg')[:10]
    predictor = init()
    s = 0
    for image_name in image_names:
        t1 = time.time()
        res = process_image(predictor, image_name)
        print(res)
        t2 = time.time()
        s += t2 - t1
    print(1/(s/100))
