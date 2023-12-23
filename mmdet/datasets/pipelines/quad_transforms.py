# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import copy
import math
import mmcv
import numpy as np
import random

from .transforms import Resize
from mmdet.core import find_inside_bboxes
from mmdet.utils import log_img_scale
from ..builder import PIPELINES


@PIPELINES.register_module()
class QuadResize(Resize):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        if self.bbox_clip_border:
            img_shape = results['img_shape']
        # train: results.get('bbox_fields',[]) -> ['gt_bboxes_ignore', 'gt_bboxes','rot_gt_bboxes','quad_gt_bboxes'] 
        # test: results.get('bbox_fields',[]) -> []
        if len(results.get('bbox_fields', [])) > 0: 
            if results['gt_bboxes'].shape[0] > 0:
                bboxes = results['gt_bboxes'] * results['scale_factor'] 
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1]) 
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0]) 
                results['gt_bboxes'] = bboxes
            if results['gt_bboxes_ignore'].shape[0] > 0:
                bboxes = results['gt_bboxes_ignore'] * results['scale_factor'] 
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1]) 
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0]) 
                results['gt_bboxes_ignore'] = bboxes 
            if results['quad_gt_bboxes'].shape[0] > 0:
                if results['scale_factor'].shape[0] == 4:
                    scale_factor = np.tile(results['scale_factor'], 2)
                else:
                    scale_factor = results['scale_factor']
                bboxes = results['quad_gt_bboxes'] * scale_factor 
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1]) #x1,x2,x3,x4 
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0]) #y1,y2,y3,y4 
                results['quad_gt_bboxes'] = bboxes
            if results['quad_gt_bboxes_ignore'].shape[0] > 0:
                if results['scale_factor'].shape[0] == 4:
                    scale_factor = np.tile(results['scale_factor'], 2)
                else:
                    scale_factor = results['scale_factor']
                bboxes = results['quad_gt_bboxes'] * scale_factor 
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1]) #x1,x2,x3,x4 
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0]) #y1,y2,y3,y4 
                results['quad_gt_bboxes_ignore'] = bboxes


@PIPELINES.register_module()
class RotateAngle(object):
    """Apply Rotate Transformation to image (and its corresponding bbox). 
        Args:
        angletype: fixed or random.
        angle: if angletype is fixed, angle is a list like fixed angle '[90, 180, 270]', if angletype is random, angle is a number like 15.
        center (int | float | tuple[float]): Center point (w, h) of the rotation in the source image. If None,the center of the image will be used.
        prob (float): The probability for perform transformation and should be in range 0 to 1.
    """
    def __init__(self, 
                 angletype, 
                 angle, 
                 prob=0.5):
        self.angletype = angletype
        self.angle = angle
        self.prob = prob

    def _rotate_img(self, results, angle, image_center, height, width): 
        """Rotate the image.
            Args:
            results (dict): Resült dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values mean clockwise rotation. Same in mmcv.imrotate. 
            center (tuple[float], optional): Center point(w,h) of the rotation.
        """
        for key in results.get('img_fields', ['img']): 
            img = results[key].copy()
            M = cv2.getRotationMatrix2D(image_center, angle, 1) 
            radians = math.radians(angle)
            sin = math.sin(radians) 
            cos = math.cos(radians)
            bound_w = int((height*abs(sin))+(width*abs(cos))) 
            bound_h = int((height*abs(cos))+(width*abs(sin))) 
            M[0, 2] += ((bound_w/2)-image_center[0])
            M[1, 2] += ((bound_h/2)-image_center[1])
            rot_im = cv2.warpAffine(img, M, (bound_w,bound_h)) 
            results[key] = rot_im.astype(img.dtype)
            results['img_shape'] = results[key].shape

    def _rotate_bboxes(self, results, angle, image_center, height, width): 
        """Rotate the bboxes."""
        for i in range(len(results['quad_gt_bboxes'])): 
            box=[]
            box.append((results['quad_gt_bboxes'][i][0], results['quad_gt_bboxes'][i][1])) 
            box.append((results['quad_gt_bboxes'][i][2], results['quad_gt_bboxes'][i][3])) 
            box.append((results['quad_gt_bboxes'][i][4], results['quad_gt_bboxes'][i][5])) 
            box.append((results['quad_gt_bboxes'][i][6], results['quad_gt_bboxes'][i][7])) 
            new_bb = np.array(box)
            for j, coord in enumerate(box):
                M = cv2.getRotationMatrix2D(image_center, angle, 1.0) 
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int((height*sin)+(width*cos)) 
                nH = int((height*cos)+(width*sin)) 
                M[0, 2] += ((nW/2)-image_center[0]) 
                M[1, 2] += ((nH/2)-image_center[1]) 
                v = [coord[0], coord[1], 1]
                calculated = np.dot(M, v)
                new_bb[j] = (calculated[0], calculated[1])
            results['quad_gt_bboxes'][i] = new_bb.reshape(1,-1) 
            x1 = min(new_bb[:, 0])
            y1 = min(new_bb[:, 1]) 
            x2 = max(new_bb[:, 0]) 
            y2 = max(new_bb[:, 1])
            results['gt_bboxes'][i] = np.array([[x1, y1, x2, y2]])

    def __call__(self, results):
        """Call function to rotate images, bounding boxes. 
            Args:
            results (dict): Result dict from loading pipeline. 
            Returns:
            dict: Rotated results.
        """
        if np.random.rand() > self.prob: 
            return results
        height, width = results['img'].shape[:2] 
        image_center = (width/2,height/2) 

        if self.angletype == 'fixed':
            index = random.choice(self.angle) 
        elif self.angletype == 'random':
            index = np.random.randint(-self.angle, self.angle) 
        else:
            raise NotImplementedError
        self._rotate_img(results, index, image_center, height, width) 
        self._rotate_bboxes(results, index, image_center, height, width) 
        return results

@PIPELINES.register_module()
class RotQuadWarp(object):
    """Apply Warp Transformation to image (and its corresponding bbox). 
     Args:
     x_ratio (int | float): is the warp range of points x.
     y_ratio (int | float): is the warp range of points y. 
    """
    def __init__(self, x_ratio=10.0, y_ratio=10.0, prob=0.5): 
        self.x_ratio = x_ratio 
        self.y_ratio = y_ratio 
        self.prob = prob

    def _warp_img(self, results,M,h,w): 
        """Warp the image.
          Args:
          results (dict): Result dict from loading pipeline. 
        """
        for key in results.get('img_fields', ['img']):
            img = results[key].copy()
            dst=cv2.warpPerspective(img, M, (w, h)) 
            results[key] = dst.astype(img.dtype) 
            results['img_shape'] = results[key].shape

    def _warp_bboxes(self, results, M):
        """Warp the bboxes,"""
        for i in range(len(results['quad_gt_bboxes'])):
            points = np.array([[results['quad_gt_bboxes'][i][0], results['quad_gt_bboxes'][i][1]],
                               [results['quad_gt_bboxes'][i][2], results['quad_gt_bboxes'][i][3]], 
                               [results['quad_gt_bboxes'][i][4], results['quad_gt_bboxes'][i][5]],
                               [results['quad_gt_bboxes'][i][6], results['quad_gt_bboxes'][i][7]]],dtype=np.float32)
            points = points.reshape(-1, 1, 2)
            # new_points = points * M
            new_points = cv2.perspectiveTransform(points, M)
            results['quad_gt_bboxes'][i] = new_points.reshape(1, -1) 
            x1 = min(new_points.reshape(4, 2)[:, 0])
            y1 = min(new_points.reshape(4, 2)[:, 1]) 
            x2 = max(new_points.reshape(4, 2)[:, 0]) 
            y2 = max(new_points.reshape(4, 2)[:, 1])
            results['gt_bboxes'][i] = np.array([[x1, y1, x2, y2]])

    def __call__(self, results):
        """Call function to warp images, bounding boxes. 
           Args:
           results (dict): Result dict from loading pipeline. 
           Returns:
           dict: warped results.
        """
        if np.random.rand() > self.prob: 
            return results
        h, w = results['img'].shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]) #get random new points
        dx0 = w * random.random() / self.x_ratio 
        dy0 = h * random.random() / self.y_ratio 
        dx1 = w * random.random() / self.x_ratio 
        dy1 = h * random.random() / self.y_ratio 
        dx2 = w * random.random() / self.x_ratio 
        dy2 = h * random.random() / self.y_ratio 
        dx3 = w * random.random() / self.x_ratio 
        dy3 = h * random.random() / self.y_ratio 
        # 左上 左下 右下 右上
        pts1 = np.float32([[dx0, dy0], [dx3, h-dy3],[w-dx2, h-dy2],[w-dx1, dy1]]) 
        M = cv2.getPerspectiveTransform(pts, pts1)
        self._warp_img(results, M, h, w) 
        self._warp_bboxes(results, M) 
        return results

@PIPELINES.register_module()
class QuadMosaic:
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Default to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Default to True.
        pad_val (int): Pad value. Default to 114.
        prob (float): Probability of applying this transformation.
            Default to 1.0.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=0,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114,
                 prob=1.0):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
            f'got {prob}.'

        log_img_scale(img_scale, skip_square=True)
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.pad_val = pad_val
        self.prob = prob

    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """

        if np.random.uniform(0, 1) > self.prob:
            return results

        results = self._mosaic_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        indexes = [np.random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        mosaic_quad_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(
            np.random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(
            np.random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            quad_gt_bboxes_i = results_patch['quad_gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            if quad_gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                quad_gt_bboxes_i[:, 0::2] = \
                    scale_ratio_i * quad_gt_bboxes_i[:, 0::2] + padw
                quad_gt_bboxes_i[:, 1::2] = \
                    scale_ratio_i * quad_gt_bboxes_i[:, 1::2] + padh
                            
            mosaic_quad_bboxes.append(quad_gt_bboxes_i)
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_quad_bboxes = np.concatenate(mosaic_quad_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0,
                                                 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0,
                                                 2 * self.img_scale[0])

                mosaic_quad_bboxes[:, 0::2] = np.clip(mosaic_quad_bboxes[:, 0::2], 0,
                                                 2 * self.img_scale[1])
                mosaic_quad_bboxes[:, 1::2] = np.clip(mosaic_quad_bboxes[:, 1::2], 0,
                                                 2 * self.img_scale[0])

            if not self.skip_filter:
                mosaic_bboxes, mosaic_quad_bboxes, mosaic_labels = \
                    self._filter_box_candidates(mosaic_bboxes, mosaic_quad_bboxes, mosaic_labels)

        # remove outside bboxes
        inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0],
                                         2 * self.img_scale[1])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_quad_bboxes = mosaic_quad_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels
        results['quad_gt_bboxes'] = mosaic_quad_bboxes

        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, quad_bboxes, labels):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & \
                     (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], quad_bboxes[valid_inds], labels[valid_inds]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str

@PIPELINES.register_module()
class QuadMixUp:
    """MixUp data augmentation.

    .. code:: text

                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (height, width). Default: (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Default: (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Default: 0.5.
        pad_val (int): Pad value. Default: 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Default: 15.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 5.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Default: 20.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 ratio_range=(0.5, 1.5),
                 flip_ratio=0.5,
                 pad_val=114,
                 max_iters=15,
                 min_bbox_size=5,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 bbox_clip_border=True,
                 skip_filter=True):
        assert isinstance(img_scale, tuple)
        log_img_scale(img_scale, skip_square=True)
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter

    def __call__(self, results):
        """Call function to make a mixup of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mixup transformed.
        """
        results = self._mixup_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        for i in range(self.max_iters):
            index = np.random.randint(0, len(dataset))
            gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
            if len(gt_bboxes_i) != 0:
                break

        return index

    def _mixup_transform(self, results):
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        assert len(
            results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']

        jit_factor = np.random.uniform(*self.ratio_range)
        is_filp = np.random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(
                self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor),
                                          int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset:y_offset + target_h,
                                        x_offset:x_offset + target_w]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(retrieve_gt_bboxes[:, 0::2],
                                                  0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(retrieve_gt_bboxes[:, 1::2],
                                                  0, origin_h)

        if is_filp:
            retrieve_gt_bboxes[:, 0::2] = (
                origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1])
        
        # 6. adjust quad bbox
        retrieve_quad_gt_bboxes = retrieve_results['quad_gt_bboxes']
        retrieve_quad_gt_bboxes[:, 0::2] = retrieve_quad_gt_bboxes[:, 0::2] * scale_ratio 
        retrieve_quad_gt_bboxes[:, 1::2] = retrieve_quad_gt_bboxes[:, 1::2] * scale_ratio 
        if self.bbox_clip_border:
            retrieve_quad_gt_bboxes[:, 0::2] = np.clip(retrieve_quad_gt_bboxes[:, 0::2],
                                                        0, origin_w)
            retrieve_quad_gt_bboxes[:, 1::2] = np.clip(retrieve_quad_gt_bboxes[:, 1::2],
                                                        0, origin_h)
        if is_filp:
            retrieve_quad_gt_bboxes_flip = retrieve_quad_gt_bboxes.copy()
            retrieve_quad_gt_bboxes_flip[..., 0::8] = origin_w - retrieve_quad_gt_bboxes[..., 2::8] 
            retrieve_quad_gt_bboxes_flip[..., 1::8] = retrieve_quad_gt_bboxes[..., 3::8]
            retrieve_quad_gt_bboxes_flip[..., 2::8] = origin_w - retrieve_quad_gt_bboxes[..., 0::8] 
            retrieve_quad_gt_bboxes_flip[..., 3::8] = retrieve_quad_gt_bboxes[..., 1::8]
            retrieve_quad_gt_bboxes_flip[..., 4::8] = origin_w - retrieve_quad_gt_bboxes[..., 6::8] 
            retrieve_quad_gt_bboxes_flip[..., 5::8] = retrieve_quad_gt_bboxes[..., 7::8]
            retrieve_quad_gt_bboxes_flip[..., 6::8] = origin_w - retrieve_quad_gt_bboxes[..., 4::8] 
            retrieve_quad_gt_bboxes_flip[..., 7::8] = retrieve_quad_gt_bboxes[..., 5::8]
            retrieve_quad_gt_bboxes = retrieve_quad_gt_bboxes_flip

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = \
            cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        cp_retrieve_gt_bboxes[:, 1::2] = \
            cp_retrieve_gt_bboxes[:, 1::2] - y_offset
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 0::2], 0, target_w)
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(
                cp_retrieve_gt_bboxes[:, 1::2], 0, target_h)
        
        # 7. filter: quad bbox
        cp_retrieve_quad_gt_bboxes = retrieve_quad_gt_bboxes.copy() 
        cp_retrieve_quad_gt_bboxes[:, 0::2] = cp_retrieve_quad_gt_bboxes[:, 0::2] - x_offset 
        cp_retrieve_quad_gt_bboxes[:, 1::2] = cp_retrieve_quad_gt_bboxes[:, 1::2] - y_offset 
        if self.bbox_clip_border:
            cp_retrieve_quad_gt_bboxes[:, 0::2] = np.clip(
                cp_retrieve_quad_gt_bboxes[:, 0::2], 0, target_w) 
            cp_retrieve_quad_gt_bboxes[:, 1::2] = np.clip(
                cp_retrieve_quad_gt_bboxes[:, 1::2], 0, target_h)

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results['gt_labels']
        if not self.skip_filter:
            keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T,
                                                    cp_retrieve_gt_bboxes.T)

            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            cp_retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]

        mixup_gt_bboxes = np.concatenate(
            (results['gt_bboxes'], cp_retrieve_gt_bboxes), axis=0)
        mixup_quad_gt_bboxes = np.concatenate(
            (results['quad_gt_bboxes'], cp_retrieve_quad_gt_bboxes), axis=0)        
        mixup_gt_labels = np.concatenate(
            (results['gt_labels'], retrieve_gt_labels), axis=0)

        # remove outside bbox
        inside_inds = find_inside_bboxes(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_quad_gt_bboxes = mixup_quad_gt_bboxes[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['quad_gt_bboxes'] = mixup_quad_gt_bboxes
        results['gt_labels'] = mixup_gt_labels

        return results

    def _filter_box_candidates(self, bbox1, bbox2):
        """Compute candidate boxes which include following 5 things:

        bbox1 before augment, bbox2 after augment, min_bbox_size (pixels),
        min_area_ratio, max_aspect_ratio.
        """

        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size)
                & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_iters={self.max_iters}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str