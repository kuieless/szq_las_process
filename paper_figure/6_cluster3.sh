#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6



# yingrenshi

python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
    --num_points=200000   \
    --output_path='zyq/1117_cluster_yingrenshi/mean_sam'  \
    --panoptic_dir='logs_instance/1104_yingrenshi_density_depth_hash22_instance_origin_sam_0.001/0/eval_200000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Yingrenshi_20230926' \
    --all_centroids='zyq/1115_cluster_yingrenshi/mean_sam/mean_depth_200000_1050/test_centroids.npy' \
    --cluster_sizes=1050


python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
    --num_points=200000   \
    --output_path='zyq/1117_cluster_yingrenshi/mean_cross_view'  \
    --panoptic_dir='logs_instance/1104_yingrenshi_density_depth_hash22_instance_origin_sam_0.001_depth_crossview/0/eval_200000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Yingrenshi_20230926' \
    --all_centroids='zyq/1115_cluster_yingrenshi/mean_cross_view/mean_depth_200000_1050/test_centroids.npy' \
    --cluster_sizes=1050

python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
    --num_points=200000   \
    --output_path='zyq/1117_cluster_yingrenshi/mean_detectron'  \
    --panoptic_dir='logs_instance/1111_yingrenshi_detectron/0/eval_100000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Yingrenshi_20230926' \
    --all_centroids='zyq/1115_cluster_yingrenshi/mean_detectron/mean_depth_200000_1050/test_centroids.npy' \
    --cluster_sizes=1050




    