#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

dataset1='Mill19'  #  "Mill19"  "UrbanScene3D"
dataset2='building' #  "building"   "residence"  "sci-art"  "campus"

for type in train val;do
        python -m preprocessing.get_image_embeddings --image_path /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels/$type/rgbs/ --save_path /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels/$type/sam_features/
        echo $dataset2-$type
done
echo start
dataset1='UrbanScene3D'  #  "Mill19"  "UrbanScene3D"
for dataset2 in residence sci-art campus; do
        for type in train val;do
                python -m preprocessing.get_image_embeddings --image_path /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels/$type/rgbs/ --save_path /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels/$type/sam_features/
                echo $dataset2-$type
        done
done
echo finish


