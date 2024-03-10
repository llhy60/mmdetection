# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
from shapely.errors import TopologicalError 
from shapely.geometry import Polygon, MultiPoint
def quad_bbox_iou(polygon_area1, polygon_area2, eps=1e-6):
    if polygon_area1.intersects(polygon_area2):
        try:
            overlap = polygon_area1.intersection(polygon_area2).area
            union = polygon_area1.union(polygon_area2).area
            iou = overlap / np.maximum(union, eps)
        except TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    else:
        iou = 0
    return iou

def quad_bbox_ciou(polygon_area1, polygon_area2, eps=1e-6):
    polygon1 = np.array(polygon_area1.exterior)[:-1]
    print(np.array(polygon_area1.exterior))
    print(np.array(polygon_area1.exterior)[:-1])
    polygon2 = np.array(polygon_area2.exterior)[:-1]

    b1_center = np.sum(polygon1, 0) / 4
    b2_center = np.sum(polygon2, 0) / 4

    b1_w = 0.5 * (np.linalg.norm(polygon1[0] - polygon1[3]) + \
                  np.linalg.norm(polygon1[1] - polygon1[2]))
    b1_h = 0.5 * (np.linalg.norm(polygon1[0] - polygon1[1]) + \
                  np.linalg.norm(polygon1[3] - polygon1[2]))
    b2_w = 0.5 * (np.linalg.norm(polygon2[0] - polygon2[3]) + \
                  np.linalg.norm(polygon2[1] - polygon2[2]))
    b2_h = 0.5 * (np.linalg.norm(polygon2[0] - polygon2[1]) + \
                  np.linalg.norm(polygon2[3] - polygon2[2]))
    
    iou = quad_bbox_iou(polygon_area1, polygon_area2, eps)

    center_distance = np.linalg.norm(b1_center - b2_center)
    b1_mins = np.min(polygon1)
    b2_mins = np.min(polygon2)
    b1_maxs = np.max(polygon1)
    b2_maxs = np.max(polygon2)

    enclose_mins = min(b1_mins, b2_mins)
    enclose_maxs = max(b1_maxs, b2_maxs)
    intersect_maxs = min(b1_maxs, b2_maxs)
    enclose_wh = np.max(enclose_maxs - enclose_mins, 0)
    enclose_diagonal = np.linalg.norm(enclose_wh)

    ciou = iou - 1.0 * (center_distance) / (enclose_diagonal + eps)
    v = (4 / (math.pi ** 2)) * ((np.arctan(b1_w/b1_h) - np.arctan(b2_w/b2_h)) ** 2)
    alpha = v / (1.0 - iou + v)
    ciou = ciou - alpha * v
    return ciou

def quad_bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6, use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        mode (str): IOU (intersection over union) or IOF (intersection
            over foreground)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.

    Returns:
        ious (ndarray): Shape (n, k)
    """

    assert mode in ['iou', 'ciou']
    if not use_legacy_coordinate:
        extra_length = 0.
    else:
        extra_length = 1.
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True

    for i in range(bboxes1.shape[0]):
        for j in range(bboxes2.shape[0]):
            polygon_area1 = Polygon(bboxes1[i,:].reshape(4,2)).convex_hull
            polygon_area2 = Polygon(bboxes2[j,:].reshape(4,2)).convex_hull

            if mode == 'iou':
                ious[i, j] = quad_bbox_iou(polygon_area1, polygon_area2)
            elif mode == 'ciou':
                ious[i, j] = quad_bbox_ciou(polygon_area1, polygon_area2)

    if exchange:
        ious = ious.T
    return ious


if __name__ == "__main__":
    bboxes1 = np.array([[908., 215., 934., 312., 752., 355., 728., 252.],
                        [908., 215., 934., 312., 752., 355., 728., 252.],
                        [-908., 215., 934., 312., 752., 355., 728., 252.],
                        [908., 215., 934., 312., 752., 355., 728., 252.]])
    bboxes2 = np.array([[923., 308, 758, 342, 741, 262, 907, 228],
                        [923., 308, 758, 342, 741, 262, 967, 228],
                        [923., 308, 758, 342, 741, 262, 907, 228],
                        [923., 308, 758, 342, 741, 262, 907, 228]])
    res = quad_bbox_overlaps(bboxes1, bboxes2, mode='ciou', eps=1-6)
    print(res)
