#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


python /data/yuqi/code/GP-NeRF-semantic/scripts/1113_debug_cluster_sam_depth_auto.py   \
    --num_points=200000   \
    --output_path='zyq/1113_b1_cluster_cross_view_all'  \
    --panoptic_dir='logs_longhua_b1/1108_longhua_b1_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all/4/eval_100000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds'
    