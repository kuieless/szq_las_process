#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4




# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2.py  \
#     --dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926   \
#     --output_path=zyq/1110_get_instance_mask_train_yingrenshi  \
#     --threshold=0   \
#     --points_per_side=64



python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2.py  \
    --dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926   \
    --output_path=zyq_sam_label/1114_get_instance_mask_train_yingrenshi  \
    --threshold=0.001   \
    --points_per_side=32