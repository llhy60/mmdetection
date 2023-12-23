# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import torch 
import numpy as np
from ..transforms import bbox_cxcywh_to_xyxy 
from ..builder import BBOX_CODERS

from .base_bbox_coder import BaseBBoxCoder 

@BBOX_CODERS.register_module()
class QuadYOLOXBBoxCoder(BaseBBoxCoder): 
    """Quad BBox coden
    Args:
        target_means (Sequence[float]):denormalizing means of target for delta coordinates
        target_stds (Sequence[float]):denormalizing standard deviation of target for delta coordinates
    """

    def __init__(self):
        super(QuadYOLOXBBoxCoder, self).__init__() 
    
    def encode(self, quad_11_target, quad_gt_bbox, priors): 
        assert quad_11_target.size(0) == quad_gt_bbox.size(0) == priors.size(0) 
        assert quad_11_target.size(-1) == quad_gt_bbox.size(-1) ==8 
        pw = priors[:, 2]
        ph = priors[:, 3]
        
        xyxy = bbox_cxcywh_to_xyxy(priors) 
        
        ex_xtop1 = xyxy[:, 0]
        ex_ytop1 = xyxy[:, 1]
        ex_xtop2 = xyxy[:, 2]
        ex_ytop2 = xyxy[:, 1] 
        ex_xtop3 = xyxy[:, 2]
        ex_ytop3 = xyxy[:, 3]
        ex_xtop4 = xyxy[:, 0]
        ex_ytop4 = xyxy[:, 3]

        quad_gt_xtop1 = quad_gt_bbox[:, 0]
        quad_gt_ytop1 = quad_gt_bbox[:, 1]
        quad_gt_xtop2 = quad_gt_bbox[:, 2] 
        quad_gt_ytop2 = quad_gt_bbox[:, 3]
        quad_gt_xtop3 = quad_gt_bbox[:, 4]
        quad_gt_ytop3 = quad_gt_bbox[:, 5]
        quad_gt_xtop4 = quad_gt_bbox[:, 6]
        quad_gt_ytop4 = quad_gt_bbox[:, 7]

        quad_11_target[:, 0] = (quad_gt_xtop1 - ex_xtop1) / pw 
        quad_11_target[:, 1] = (quad_gt_ytop1 - ex_ytop1) / ph
        quad_11_target[:, 2] = (quad_gt_xtop2 - ex_xtop2) / pw
        quad_11_target[:, 3] = (quad_gt_ytop2 - ex_ytop2) / ph
        quad_11_target[:, 4] = (quad_gt_xtop3 - ex_xtop3) / pw
        quad_11_target[:, 5] = (quad_gt_ytop3 - ex_ytop3) / ph
        quad_11_target[:, 6] = (quad_gt_xtop4 - ex_xtop4) / pw 
        quad_11_target[:, 7] = (quad_gt_ytop4 - ex_ytop4) / ph
        
        return quad_11_target

    def decode(self, priors, bbox_preds):
        # bboxes, quad_pred_boxes, quad_nms_box
        xyxy = bbox_cxcywh_to_xyxy(priors) 
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2] 
        whs = bbox_preds[..., 2:4].exp() * priors[:, 2:]
        tl_x = (xys[..., 0] - whs[..., 0] / 2) 
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)
        # quad
        dxtop1 = bbox_preds[..., 4::12] 
        dytop1 = bbox_preds[..., 5::12]
        dxtop2 = bbox_preds[..., 6::12]
        dytop2 = bbox_preds[..., 7::12] 
        dxtop3 = bbox_preds[..., 8::12] 
        dytop3 = bbox_preds[..., 9::12] 
        dxtop4 = bbox_preds[..., 10::12] 
        dytop4 = bbox_preds[..., 11::12]

        pw = priors[..., 2:3]
        ph = priors[..., 3:4] 
        # xtop1
        xtop1 = xyxy[..., 0].unsqueeze(1).expand_as(dxtop1) + dxtop1 * pw 
        # ytop1
        ytop1 = xyxy[..., 1].unsqueeze(1).expand_as(dytop1) + dytop1 * ph 
        # xtop2
        xtop2 = xyxy[..., 2].unsqueeze(1).expand_as(dxtop2) + dxtop2 * pw 
        # ytop2
        ytop2 = xyxy[..., 1].unsqueeze(1).expand_as(dytop2) + dytop2 * ph
        # xtop3
        xtop3 = xyxy[..., 2].unsqueeze(1).expand_as(dxtop3) + dxtop3 * pw
        # ytop3
        ytop3 = xyxy[..., 3].unsqueeze(1).expand_as(dytop3) + dytop3 * ph
        # xtop4
        xtop4 = xyxy[..., 0].unsqueeze(1).expand_as(dxtop4) + dxtop4 * pw 
        # ytop4
        ytop4 = xyxy[..., 3].unsqueeze(1).expand_as(dytop4) + dytop4 * ph 
        # decoded_bboxes -> torch.Size([2, 8400, 4]) 
        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)

        # quad_decoded_bboxes -> torch.Size([2,8400,1,8])-→torch.Size([2,8400,8])
        quad_decoded_bboxes = torch.stack([xtop1, ytop1, xtop2, ytop2,
                                           xtop3, ytop3, xtop4, ytop4], dim=-1) 
        quad_decoded_bboxes = quad_decoded_bboxes.reshape(quad_decoded_bboxes.size(0),
                                                          quad_decoded_bboxes.size(1), -1)

        # rot_for_nms, [cx, cy, h, w, angle]
        quad_np = quad_decoded_bboxes.detach().cpu().numpy() 
        # rpoints ->(8400,4,2)
        rpoints = np.reshape(quad_np, [-1, 4, 2]) 
        rect_for_nms = []
        for i in range(rpoints.shape[0]):
            rect = cv2.minAreaRect(rpoints[i])
            rect_for_nms.append(rect[0][0]) 
            rect_for_nms.append(rect[0][1]) 
            rect_for_nms.append(rect[1][1]) # h 
            rect_for_nms.append(rect[1][0]) # w 
            rect_for_nms.append(rect[2]) # angle
 
        rect_for_nms = np.asarray(rect_for_nms, dtype=np.float32).reshape(-1, 5) 
        # x_center
        nms_cx = torch.from_numpy(rect_for_nms[..., 0::5])
        # y_center
        nms_cy = torch.from_numpy(rect_for_nms[..., 1::5])
        # h
        nms_h = torch.from_numpy(rect_for_nms[..., 2::5])
        # w
        nms_w = torch.from_numpy(rect_for_nms[..., 3::5])
        # angle
        nms_alpha = torch.from_numpy(rect_for_nms[..., 4::5])

        rot_nms_box = torch.stack([nms_cx, nms_cy, nms_h, nms_w, nms_alpha], dim=-1)
        # torch.Size([8400, 1, 5]) -> torch.size([1, 8400, 5])
        rot_nms_box = rot_nms_box.permute(1, 0, 2)

        return decoded_bboxes, quad_decoded_bboxes, rot_nms_box