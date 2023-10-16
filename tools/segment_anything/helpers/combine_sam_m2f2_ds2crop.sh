#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6




python tools/segment_anything/helpers/combine_sam_m2f2.py  \
    --sam_features_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/sam_features  \
    --rgbs_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs    \
    --labels_m2f_path=/data/yuqi/code/Mask2Former_a/output_useful/1016_compare_ds2/labels_m2f_final  \
    --output_path=/data/yuqi/code/Mask2Former_a/output_useful/1016_compare_ds2/labels_m2f_final_sam