#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=2

dataroot=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/val
python tools/segment_anything/helpers/combine_sam_m2f.py  \
    ${dataroot}   \
    /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/save_merge
