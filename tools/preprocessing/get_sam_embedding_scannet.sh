#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


python -m preprocessing.get_image_embeddings_scannet --image_path /data/yuqi/code/panoptic-lifting/data/scannet/scene0050_02/color --save_path /data/yuqi/code/panoptic-lifting/data/scannet/scene0050_02/sam_features/





