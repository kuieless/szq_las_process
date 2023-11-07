#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6


python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1103_get_sammask_autogenerate_depth_filter.py  \
    --output_path=zyq/1107_get_instance_mask_train_longhua_b1_depth  \
    --threshold=0.001   \
    --dataset_path=/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds
    