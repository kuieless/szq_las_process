#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4



# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
#     --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.3/pred_rgb


python tools/segment_anything/helpers/combine_sam_m2f3_only_sam.py  \
    --sam_features_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.5/sam_features  \
    --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.5/pred_rgb   \
    --labels_m2f_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.5/mask2former_output/labels_m2f  \
    --output_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.5/mask2former_output/labels_m2f_only_sam