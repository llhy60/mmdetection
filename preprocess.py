import os
import argparse
from glob import glob
from os.path import join
from tqdm import tqdm
from sklearn.model_selection import train_test_split


'''
Create VOC dataset
'''


classes = ['front_wear', 'front_no_wear', 'front_under_nose_wear', 'front_under_mouth_wear', 'mask_front_wear',
           'mask_front_under_nose_wear', 'mask_front_under_mouth_wear', 'side_wear', 'side_no_wear', 'side_under_nose_wear',
           'side_under_mouth_wear', 'mask_side_wear', 'mask_side_under_nose_wear', 'mask_side_under_mouth_wear', 
           'side_back_head_wear', 'side_back_head_no_wear', 'back_head', 'strap', 'front_unknown', 'side_unknown']

class2id = {name:i for i, name in enumerate(classes)}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Merge data")
    parser.add_argument ('--root_path', required=True, default=None, type=str, help="root file path")
    args = parser.parse_args()
    return args

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge_data(data_path_list, dst_path):
    create_dir(dst_path)
    for src_path in data_path_list:
        if os.path.exists(src_path):
            _, name = os.path.split(src_path)
            os.symlink(src_path, join(dst_path, name))
    print('[INFO] Soft links in the %s has been completed! ' % (dst_path))


def catrainlist(files_path, trainval_path):
    create_dir(trainval_path)
    with open(os.path.join(trainval_path, "trainval.txt"), 'w') as f:
        with tqdm(total=len(os.listdir(files_path)), desc="ProgressBar") as pbar:
            for name in os.listdir(files_path):
                f.write(name[:-4] + '\n')
                pbar.update(1)
    print(f"[INFO] Trainval.txt has been writed! Total of data: {len(os.listdir(files_path))}")
    train_path, val_path = train_test_split(os.listdir(files_path), test_size=0.15, random_state=42)
    with open(os.path.join(trainval_path, 'train.txt'), 'w') as f:
        with tqdm(total=len(train_path), desc="ProgressBar") as pbar:
            for name in train_path:
                f.write(name[:-4] + '\n')
                pbar.update(1)
    print(f'[INFO] Train.txt has been writed! Total of data: {len(train_path)}')
    with open(os.path.join(trainval_path, 'val.txt'), 'w') as f:
        with tqdm(total=len(val_path), desc="ProgressBar") as pbar:
            for name in val_path:
                f.write(name[:-4] + '\n')
                pbar.update(1)
    print(f'[INFO] Val.txt has been writed! Total of data: {len(val_path)}')


if __name__ == "__main__":
    args = parse_args()
    root = args.root_path
    Annotations = join(root, 'VOCdevkit2007/VOC2007/Annotations')
    JPEGImages = join(root, 'VOCdevkit2007/VOC2007/JPEGImages')
    ImageSets = join(root, 'VOCdevkit2007/VOC2007/ImageSets/Main')
    xml_files = glob(join(root + '*/*.xml'))
    img_files = glob(join(root + '*/*.jpg'))
    assert len(img_files) == len(xml_files), '[INFO] Error: The number of pictures is inconsistent with the xml files!'
    merge_data(xml_files, Annotations)
    merge_data(img_files, JPEGImages)
    catrainlist(Annotations, ImageSets)