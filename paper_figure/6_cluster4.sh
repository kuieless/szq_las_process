#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


#campus


# python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
#     --num_points=200000   \
#     --output_path='zyq/1117_cluster_campus/mean_sam'  \
#     --panoptic_dir='logs_instance/1107_campus_density_depth_hash22_instance_origin_sam_0.001/0/eval_100000/panoptic' \
#     --dataset_path='/data/yuqi/Datasets/DJI/Campus_new' \
#     --all_centroids='zyq/1114_cluster_campus/mean_sam/mean_depth_200000_1050/test_centroids.npy' \
#     --cluster_sizes=1050


# python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
#     --num_points=200000   \
#     --output_path='zyq/1117_cluster_campus/mean_cross_view_all'  \
#     --panoptic_dir='logs_instance/1113_campus_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all/0/eval_100000/panoptic' \
#     --dataset_path='/data/yuqi/Datasets/DJI/Campus_new' \
#     --all_centroids='/data/yuqi/code/GP-NeRF-semantic/zyq/1114_cluster_campus/mean_cross_view_all/mean_depth_200000_1000/test_centroids.npy' \
#     --cluster_sizes=1000

    



python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
    --num_points=200000   \
    --output_path='zyq/1117_cluster_campus/mean_detectron'  \
    --panoptic_dir='logs_instance/1111_campus_detectron/0/eval_100000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Campus_new' \
    --all_centroids='/data/yuqi/code/GP-NeRF-semantic/zyq/1114_cluster_campus/mean_detectron/mean_depth_200000_1000/test_centroids.npy' \
    --cluster_sizes=1000

    