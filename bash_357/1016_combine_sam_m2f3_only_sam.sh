#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4



# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
#     --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/pred_rgb


python tools/segment_anything/helpers/combine_sam_m2f3_only_sam.py  \
    --sam_features_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/sam_features  \
    --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/pred_rgb   \
    --labels_m2f_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/mask2former_output/labels_m2f  \
    --output_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/mask2former_output/labels_m2f_only_sam



dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926
far_paths=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/mask2former_output/labels_m2f_only_sam/labels_merge
output_path=logs_dji/augument/1015_far0.3/project_to_ori_gt_only_sam
render_type=render_far0.3

/home/yuqi/anaconda3/envs/gpnerf/bin/python   /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_3d_get_2d_label_gy_filter_far_to_ori.py \
    --dataset_path=$dataset_path  \
    --exp_name='logs_357/test'  \
    --far_paths=$far_paths   \
    --output_path=$output_path \
    --render_type=$render_type
    
/home/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_compare_ori_far.py  \
    --dataset_path=$dataset_path    \
    --exp_name='logs_357/test'  \
    --project_path=$output_path/project_far_to_ori  \
    --output_path=$output_path/compare_vis

/home/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/scripts/1016_only_sam_building_label_replace.py  \
    --dataset_path=$dataset_path    \
    --exp_name=logs_357/test    \
    --only_sam_m2f_far_project_path=$output_path/project_far_to_ori    \
    --output_path=$output_path/project_far_to_ori_replace_building
    