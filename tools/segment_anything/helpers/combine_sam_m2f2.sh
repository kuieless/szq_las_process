#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5



python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
    --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/pred_rgb


python tools/segment_anything/helpers/combine_sam_m2f2.py  \
    --sam_features_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/sam_features  \
    --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/pred_rgb    \
    --labels_m2f_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/yingrenshi_panoptic_car_augment_near/labels_m2f  \
    --output_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/9/eval_200000_near/yingrenshi_merge_near