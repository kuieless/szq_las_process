#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5


python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate.py  \
    --image_path=/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds/train/rgbs \
    --sam_feat_path=/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds/train/sam_features  \
    --output_path=zyq/1101_get_instance_mask_train_b2_old  \
    --threshold=0.001   
    