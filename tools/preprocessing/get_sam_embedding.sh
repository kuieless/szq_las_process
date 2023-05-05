#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

dataset1='UrbanScene3D'  #  "Mill19"  "UrbanScene3D"
dataset2='sci-art' #  "building"   "residence"  "sci-art"  "campus"

for type in train val;do
    python -m preprocessing.get_image_embeddings --image_path /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels/$type/rgbs/ --save_path /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels/$type/sam_features/
done





