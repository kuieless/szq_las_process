#!/bin/bash
export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=6


exp_name='logs_357/0324_sem_grad_stop3_pretrain'


dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
val_interval=100000
train_iterations=100000
label_size=1024
label_type=
stop_semantic_grad=True
use_pano_lift=False
ckpt_path=/data/yuqi/code/GP-NeRF-private/log_357/iccv/0228-plane-color-sci-art/1/models/100000.pt
python gp_nerf/train.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --chunk_paths   /data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_size-1      --exp_name  $exp_name  --train_iterations   $train_iterations   --desired_chunks  15     --val_interval  $val_interval   --label_size   $label_size  --stop_semantic_grad  $stop_semantic_grad  --use_pano_lift  $use_pano_lift   --ckpt_path  $ckpt_path    --no_resume_ckpt_state











