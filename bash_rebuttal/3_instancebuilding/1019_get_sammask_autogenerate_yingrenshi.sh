#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1




# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
#     --rgbs_path=/data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx/train/rgbs



python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2_wo_depth.py  \
    --dataset_path=/data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx   \
    --output_path=logs_rebuttal/0129_get_instance_mask_train_ib1  \
    --threshold=0.001   \
    --points_per_side=32