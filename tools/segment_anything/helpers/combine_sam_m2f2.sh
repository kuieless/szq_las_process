#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4



python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
    --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/pred_rgb


python tools/segment_anything/helpers/combine_sam_m2f2.py  \
    --sam_features_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/sam_features  \
    --rgbs_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs    \
    --labels_m2f_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/project_to_ori_gt/project_far_to_ori  \
    --output_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/project_to_ori_gt/project_far_to_ori_sam