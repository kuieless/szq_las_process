#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1

# python -m preprocessing.get_image_embeddings --image_path /data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/val/rgbs/ --save_path /data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/val/sam_features/

python -m preprocessing.get_image_embeddings --image_path /data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs/ --save_path /data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/sam_features/




