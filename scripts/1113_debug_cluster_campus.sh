#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


python /data/yuqi/code/GP-NeRF-semantic/scripts/1113_debug_cluster_campus_sam_depth_auto.py   \
    --num_points=200000   \
    --output_path='zyq/1113_campus_cluster_cross_view'  \
    --panoptic_dir='logs_campus/1107_campus_density_depth_hash22_instance_origin_sam_0.001_depth_crossview/12/eval_100000_1112/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Campus_new'
    