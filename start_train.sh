cd /project/train/src_repo/mmlab/mmdetection_llh

echo "Prepare environment..."
pip install openmim
mim install mmcv-full
pip install -r requirements/build.txt
pip install -v -e .

echo "Processing data..."
python preprocess.py --root_path /home/data/

echo "Start training..."
python tools/train.py configs/yolox/exp1_yolox_s_mask_baseline.py