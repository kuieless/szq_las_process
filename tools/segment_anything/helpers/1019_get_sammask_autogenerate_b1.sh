#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2.py  \
#     --dataset_path=/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds   \
#     --output_path=zyq/1107_get_instance_mask_train_longhua_b1  \
#     --threshold=0.001   


### 2023.11.10 

python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2.py  \
    --dataset_path=/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds   \
    --output_path=zyq/1110_get_instance_mask_train_b1  \
    --threshold=0   \
    --points_per_side=64