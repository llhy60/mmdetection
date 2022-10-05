# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from ..builder import DETECTORS
from .quad_single_stage import QuadSingleStageDetector

@DETECTORS.register_module()
class QuadYOLOX(QuadSingleStageDetector):
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      quad_gt_bboxes=None,
                      quad_gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # Multi-scale training
        img, gt_bboxes, quad_gt_bboxes = self._preprocess(img, gt_bboxes, quad_gt_bboxes)

        losses = super(QuadYOLOX, self).forward_train(img, img_metas, gt_bboxes,
                                                      gt_labels, gt_bboxes_ignore,
                                                      quad_gt_bboxes, quad_gt_bboxes_ignore)

        # random resizing
        if (self._progress_in_iter + 1) % self._random_size_interval == 0:
            self._input_size = self._random_resize(device=img.device)
        self._progress_in_iter += 1

        return losses

    def _preprocess(self, img, gt_bboxes, quad_gt_bboxes):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(
                img,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            for gt_bbox in gt_bboxes:
                gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
                gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
            for quad_gt_bbox in quad_gt_bboxes:
                quad_gt_bbox[..., 0::2] = quad_gt_bbox[..., 0::2] * scale_x
                quad_gt_bbox[..., 1::2] = quad_gt_bbox[..., 1::2] * scale_y              
        return img, gt_bboxes, quad_gt_bboxes
