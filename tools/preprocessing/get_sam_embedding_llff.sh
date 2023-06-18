#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


python -m preprocessing.get_image_embeddings --image_path /data/yuqi/Datasets/NeRF/nerf_llff_data/fern/images_8/ --save_path /data/yuqi/Datasets/NeRF/nerf_llff_data/fern/sam_features/






