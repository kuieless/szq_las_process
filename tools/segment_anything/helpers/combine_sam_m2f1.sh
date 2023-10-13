#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5

dataroot=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train
python tools/segment_anything/helpers/combine_sam_m2f1.py  \
    ${dataroot}   \
    /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/output_merge/yingrenshi