#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4

#b1

# python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
#     --num_points=200000   \
#     --output_path='zyq/1117_cluster_b1/mean_sam'  \
#     --panoptic_dir='logs_instance/1107_longhua_b1_density_depth_hash22_instance_origin_sam_0.001/0/eval_100000/panoptic' \
#     --dataset_path='/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds' \
#     --all_centroids='/data/yuqi/code/GP-NeRF-semantic/zyq/1114_cluster_b1/mean_sam/mean_depth_200000_1050/test_centroids.npy' \
#     --cluster_sizes=1050


# python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
#     --num_points=200000   \
#     --output_path='zyq/1117_cluster_b1/mean_cross_view_all'  \
#     --panoptic_dir='logs_instance/1108_longhua_b1_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all/0/eval_100000/panoptic' \
#     --dataset_path='/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds' \
#     --all_centroids='zyq/1114_cluster_b1/mean_cross_view_all/mean_depth_200000_1000/test_centroids.npy' \
#     --cluster_sizes=1000

python /data/yuqi/code/GP-NeRF-semantic/scripts/1114_debug_cluster_sam_depth_give_center.py   \
    --num_points=200000   \
    --output_path='zyq/1117_cluster_b1/mean_detectron'  \
    --panoptic_dir='logs_instance/1111_b1_detectron/0/eval_100000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds' \
    --all_centroids='zyq/1114_cluster_b1/mean_detectron/mean_depth_200000_1000/test_centroids.npy' \
    --cluster_sizes=1000





    