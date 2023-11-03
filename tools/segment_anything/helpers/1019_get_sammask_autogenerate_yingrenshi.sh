#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4


python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate.py  \
    --image_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs \
    --sam_feat_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/sam_features  \
    --output_path=zyq/1103_get_instance_mask_train_yingrenshi_filter  \
    --threshold=0.001   
    