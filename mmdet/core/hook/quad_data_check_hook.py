# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import cv2
import numpy as np 
import os
from mmcv.runner.hooks import HOOKS, Hook 

@HOOKS.register_module()
class QuadDataCheckHook(Hook):
    def __init__(self, work_dir, max_check_iter=100, **kwargs): 
        super(QuadDataCheckHook, self).__init__(**kwargs) 
        self.work_dir = work_dir
        self.max_check_iter = max_check_iter 
    
    def before_train_iter(self, runner): 
        # data and iter from runner
        data_loader = runner.data_loader 
        index = runner._inner_iter 
        if index > self.max_check_iter: 
            return
        data_iter = itertools.islice(data_loader, index, index+1) 
        out_dir = self.work_dir
        if not os.path.exists(out_dir): 
            os.mkdir(out_dir)
            for data_batch in data_iter:
                for i in range(len(data_batch['img_metas'].data[0])): 
                    bboxes={}
                    img_file = data_batch['img_metas'].data[0][i]['filename'] 
                    img = data_batch['img'].data[0][i].numpy()
                    img = np.ascontiguousarray(img.transpose(1,2,0)) 
                    # denorm
                    img_norm_cfg = data_batch['img_metas'].data[0][i]['img_norm_cfg'] 
                    assert img.shape[-1] == 3
                    img = img * img_norm_cfg['std'] 
                    img = img + img_norm_cfg['mean'] 
                    img = img.astype(np.uint8, copy=True) 
                    # gt data
                    # bboxes['gt_bboxes']:[[x1,y1,x2,y2], ...] 
                    bboxes['gt_bboxes'] = data_batch['gt_bboxes'].data[0][i].numpy() 
                    bboxes['quad_gt_bboxes'] = data_batch['quad_gt_bboxes'].data[0][i].numpy() 
                    # draw and save img
                    out_file_name = os.path.join(out_dir, os.path.basename(img_file)) 
                    draw_check_result(img, bboxes, out_file_name)

def draw_check_result(img, bboxes, out_file_name): 
    """
    args:
        img(numpy): (h,w,c)
        bboxes(dict): gt_bboxes(n,4),rot_gt_bboxes(n,8) 
        out_file_name: the result name.
    """
    for key in bboxes.keys(): 
        if key == 'gt_bboxes':
            for bbox in bboxes['gt_bboxes']: 
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) 
        if key == 'quad_gt_bboxes':
            for quad_bbox in bboxes['quad_gt_bboxes']: 
                x1, y1, x2, y2, x3, y3, x4, y4 = quad_bbox
                points = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], np.int32) 
                font = cv2.FONT_HERSHEY_SIMPLEX
                for i in range(len(points)):
                    cv2.putText(img, '{:s}'.format(str(i)), tuple(points[i]), font, 0.8, (255, 0, 255), 2) 
                    cv2.polylines(img, [points], True, (0, 255, 0), 2, lineType=cv2.LINE_AA) 
    cv2.imwrite(out_file_name, img)
