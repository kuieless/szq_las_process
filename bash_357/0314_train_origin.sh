#!/bin/bash
export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=7


exp_name='logs_357/0314_train_origin'


dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
# python gp_nerf/train.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-1      --exp_name  $exp_name
wgt_sem_loss=4e-2
val_interval=100000
train_iterations=100000
label_size=origin
python gp_nerf/train.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_size-1      --exp_name  $exp_name  --train_iterations   $train_iterations   --desired_chunks  15   --wgt_sem_loss   $wgt_sem_loss   --val_interval  $val_interval   --label_size   $label_size

#python gp_nerf/eval.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_size-1      --exp_name  $exp_name  --train_iterations   $train_iterations   --desired_chunks  15   --wgt_sem_loss   $wgt_sem_loss   --val_interval  $val_interval   --label_size   $label_size  --ckpt_path    /data/yuqi/code/GP-NeRF-semantic/logs_357/0314_train/1/models/100000.pt










