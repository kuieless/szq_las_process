#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=3

python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   --dataset-path=/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/train











