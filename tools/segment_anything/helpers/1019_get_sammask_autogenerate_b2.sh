#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5


python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2.py  \
    --dataset_path=/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds   \
    --output_path=zyq/1105_get_instance_mask_train_longhua_b2  \
    --threshold=0.001   