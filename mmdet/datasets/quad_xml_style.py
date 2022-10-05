# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class QuadXMLDataset(CustomDataset):
    """XML dataset for detection.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
        img_subdir (str): Subdir where images are stored. Default: JPEGImages.
        ann_subdir (str): Subdir where annotations are. Default: Annotations.
    """
    def __init__(self,
                 min_size=None,
                 bbox_type='quad',
                 cache_file=None,
                 img_subdir='JPEGImages',
                 ann_subdir='Annotations',
                 **kwargs):
        assert self.CLASSES or kwargs.get(
            'classes', None), 'CLASSES in `XMLDataset` can not be None.'
        self.img_subdir = img_subdir
        self.bbox_type = bbox_type
        self.cache_file = cache_file
        self.ann_subdir = ann_subdir
        super(QuadXMLDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
    
    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        # 从cache file中读取数据
        if self.cache_file is not None and osp.exists(self.cache_file): 
            with open(self.cache_file, 'rb') as f:
                try:
                    data_infos = pickle.load(f) 
                except:
                    data_infos = pickle.load(f, encoding='bytes')
                    print('[INFO] image data is loaded from{}'.format(self.cache_file)) 
                    return data_infos

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = osp.join(self.img_subdir, f'{img_id}.jpg')
            xml_path = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, filename)
                img = Image.open(img_path)
                width, height = img.size
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        
        # 写入cache file文件
        if self.cache_file is not None:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(data_infos, f, pickle.HIGHEST_PROTOCOL)
                print('[INFO] pickle the data to {}'.format(self.cache_file))

        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without annotation."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            if self.filter_empty_gt:
                img_id = img_info['id']
                xml_path = osp.join(self.img_prefix, self.ann_subdir,
                                    f'{img_id}.xml')
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in self.CLASSES:
                        valid_inds.append(i)
                        break
            else:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml') 
        tree = ET.parse(xml_path)
        root = tree.getroot() 
        bboxes = []
        labels = []
        quad_bboxes=[]
        labels_ignore=[]
        bboxes_ignore = []
        quad_bboxes_ignore = []

        for obj in root.findall('object'):
            name = obj.find('name').text 
            if name not in self.CLASSES: 
                continue
            label = self.cat2label[name] 
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            quadbox = obj.find('quadbox')
            quadx0 = float(quadbox.find('quadx0').text)
            quady0 = float(quadbox.find('quady0').text)
            quadx1 = float(quadbox.find('quadx1').text)
            quady1 = float(quadbox.find('quady1').text)
            quadx2 = float(quadbox.find('quadx2').text)
            quady2 = float(quadbox.find('quady2').text)
            quadx3 = float(quadbox.find('quadx3').text)
            quady3 = float(quadbox.find('quady3').text)
            x1 = min(quadx0, quadx1, quadx2, quadx3) 
            y1 = min(quady0, quady1, quady2, quady3)
            x2 = max(quadx0, quadx1, quadx2, quadx3)
            y2 = max(quady0, quady1, quady2, quady3) 
            quad_bbox = [quadx0, quady0, quadx1, quady1, quadx2, quady2, quadx3, quady3]
            bbox = [x1, y1, x2, y2]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                quad_bboxes_ignore.append(quad_bbox)
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                quad_bboxes.append(quad_bbox)
                bboxes.append(bbox)
                labels.append(label)
        
        if not quad_bboxes and not bboxes:
            quad_bboxes = np.zeros((0, 8))
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            quad_bboxes = np.array(quad_bboxes, ndmin=2) - 1
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)

        if not quad_bboxes_ignore and not bboxes_ignore:
            quad_bboxes_ignore = np.zeros((0, 8))
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            quad_bboxes = np.array(quad_bboxes, ndmin=2) - 1
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            quad_bboxes=quad_bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            quad_bboxes_ignore=quad_bboxes_ignore.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.
           部分wrapper dataset需要调用
        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids