#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4

python -m preprocessing.get_image_embeddings --image_path /data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/val/rgbs/ --save_path /data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/val/sam_features/





