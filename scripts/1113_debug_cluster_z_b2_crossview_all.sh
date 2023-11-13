#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


# python /data/yuqi/code/GP-NeRF-semantic/scripts/1113_debug_cluster_sam_depth_auto.py   \
#     --num_points=200000   \
#     --output_path='zyq/1113_b2_cluster_cross_view_all'  \
#     --panoptic_dir='logs_longhua_b2/1108_longhua_b2_density_depth_hash22_instance_origin_sam_0.001_depth_crossview_all/3/eval_100000/panoptic' \
#     --dataset_path='/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds'


# yingrenshi crossview
python /data/yuqi/code/GP-NeRF-semantic/scripts/1113_debug_cluster_sam_depth_auto.py   \
    --num_points=200000   \
    --output_path='zyq/1113_yingrenshi_cluster_cross_view'  \
    --panoptic_dir='logs_dji/1104_yingrenshi_density_depth_hash22_instance_origin_sam_0.001_depth_crossview/3/eval_200000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Yingrenshi_20230926'
    
# yingrenshi detectron
python /data/yuqi/code/GP-NeRF-semantic/scripts/1113_debug_cluster_sam_depth_auto.py   \
    --num_points=200000   \
    --output_path='zyq/1113_yingrenshi_cluster_detectron'  \
    --panoptic_dir='logs_dji/1111_yingrenshi_detectron/1/eval_100000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Yingrenshi_20230926'
    

# b1 sam
python /data/yuqi/code/GP-NeRF-semantic/scripts/1113_debug_cluster_sam_depth_auto.py   \
    --num_points=200000   \
    --output_path='zyq/1113_b1_cluster_sam'  \
    --panoptic_dir='logs_longhua_b1/1107_longhua_b1_density_depth_hash22_instance_origin_sam_0.001/2/eval_100000/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds'

# campus sam
python /data/yuqi/code/GP-NeRF-semantic/scripts/1113_debug_cluster_sam_depth_auto.py   \
    --num_points=200000   \
    --output_path='zyq/1113_campus_cluster_sam'  \
    --panoptic_dir='/data/yuqi/code/GP-NeRF-semantic/logs_campus/1107_campus_density_depth_hash22_instance_origin_sam_0.001/6/eval_100000_1112/panoptic' \
    --dataset_path='/data/yuqi/Datasets/DJI/Campus_new'
    
    