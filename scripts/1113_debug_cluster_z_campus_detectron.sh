#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


python /data/yuqi/code/GP-NeRF-semantic/scripts/1113_debug_cluster_sam_depth_auto.py   \
    --num_points=200000   \
    --output_path='zyq/1113_campus_cluster_detectron'  \
    --panoptic_dir='logs_campus/1111_campus_detectron/5/eval_100000_1112/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Campus_new'
    