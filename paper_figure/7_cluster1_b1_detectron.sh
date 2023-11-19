#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4

#b1

python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
    --num_points=200000   \
    --output_path='zyq/1117_cluster_b1/mean_detectron'  \
    --panoptic_dir='logs_instance/1111_b1_detectron/0/eval_100000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds' \
    --all_centroids='/data/yuqi/code/GP-NeRF-semantic/zyq/1115_cluster_b1/mean_detectron/mean_depth_200000_1000/test_centroids.npy' \
    --cluster_sizes=1000

