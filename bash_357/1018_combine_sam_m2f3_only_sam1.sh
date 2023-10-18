#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5



# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
#     --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1018_far0.3/pred_rgb



# #  far 视角， 只拿sam识别出的区域
# python tools/segment_anything/helpers/combine_sam_m2f3_only_sam.py  \
#     --sam_features_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1018_far0.3/sam_features  \
#     --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1018_far0.3/pred_rgb   \
#     --labels_m2f_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1018_far0.3/mask2former_output/labels_m2f  \
#     --output_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1018_far0.3/mask2former_output/labels_m2f_only_sam



dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926
far_paths=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1018_far0.3/mask2former_output/labels_m2f_only_sam/labels_merge
output_path=logs_dji/augument/1018_far0.3

render_type=render_far0.3

# # 上面得到的far （sam识别到区域的标签）投影到原始图
# /home/yuqi/anaconda3/envs/gpnerf/bin/python   /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_3d_get_2d_label_gy_filter_far_to_ori.py \
#     --dataset_path=$dataset_path  \
#     --exp_name='logs_357/test'  \
#     --far_paths=$far_paths   \
#     --output_path=$output_path/project_to_ori_gt_only_sam_filter \
#     --render_type=$render_type


# #1018加一个投影后的标签，经过sam得到清晰边缘的流程
# python tools/segment_anything/helpers/combine_sam_m2f2.py  \
#     --sam_features_path=$dataset_path/train/sam_features  \
#     --rgbs_path=$dataset_path/train/rgbs   \
#     --labels_m2f_path=$output_path/project_to_ori_gt_only_sam_filter/project_far_to_ori  \
#     --output_path=$output_path/project_to_ori_gt_only_sam_filter_sam



# # 拿投影后的label中的building覆盖原始的m2f
# /home/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/scripts/1016_1_only_sam_building_label_replace.py  \
#     --dataset_path=$dataset_path    \
#     --exp_name=logs_357/test    \
#     --only_sam_m2f_far_project_path=$output_path/project_to_ori_gt_only_sam_filter_sam/labels_merge    \
#     --output_path=$output_path/project_to_ori_gt_only_sam_filter_sam_replace_building


# # 这个是可视化，对比投影前后的效果
# /home/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_compare_ori_far.py  \
#     --dataset_path=$dataset_path    \
#     --exp_name='logs_357/test'  \
#     --project_path=$output_path/project_to_ori_gt_only_sam_filter_sam_replace_building/replace_building_label  \
#     --output_path=$output_path/compare_vis_ori_vs_replace_building


# /home/yuqi/anaconda3/envs/gpnerf/bin/python   /data/yuqi/code/GP-NeRF-semantic/scripts/1016_2_further_replace_car_tree_from_crop.py  \
#     --dataset_path=$dataset_path    \
#     --exp_name=logs_357/test  \
#     --only_sam_m2f_far_project_path=$output_path/project_to_ori_gt_only_sam_filter_sam_replace_building/replace_building_label  \
#     --crop_m2f=/data/yuqi/code/Mask2Former_a/output_useful/1016_compare_ds2/labels_m2f_final_sam/labels_merge  \
#     --output_path=$output_path/project_to_ori_gt_only_sam_filter_sam_replace_building_cartree


# 这个是可视化，对比投影前后的效果
/home/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_compare_ori_far.py  \
    --dataset_path=$dataset_path    \
    --exp_name='logs_357/test'  \
    --project_path=$output_path/project_to_ori_gt_only_sam_filter_sam_replace_building_cartree/replace_cartree_label  \
    --output_path=$output_path/compare_vis_ori_vs_replace_building_cartree

    