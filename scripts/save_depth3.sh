#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5

dataset1=Mill19  #  Mill19   UrbanScene3D
dataset2=building    #  building    sci-art   residence    campus


for dataset1 in UrbanScene3D;do
for dataset2 in campus;do
    python scripts/save_depth.py  --exp_name   ./logs_357/test   \
        --dataset_path   /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels    --config_file  configs/$dataset2.yaml      \
        --ckpt_path  /data/yuqi/code/GP-NeRF-semantic/logs_357/0504_G_geo_$dataset2/0/models/200000.pt  --dataset_type  filesystem
done
done