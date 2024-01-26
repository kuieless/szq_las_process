#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6


dataset_path=/data/yuqi/jx_rebuttal/seq14_ds_10_val/postprocess
config_file=configs/seq14.yaml


batch_size=20480
train_iterations=200000
val_interval=20000
ckpt_interval=20000

network_type=gpnerf_nr3d     #  gpnerf   sdf
dataset_type=memory_depth_dji

use_scaling=False
sampling_mesh_guidance=False

enable_semantic=True

freeze_geo=True
label_name=m2f
separate_semantic=True
depth_dji_loss=False
wgt_depth_mse_loss=0
lr=0.01


exp_name=logs_rebuttal/0126_seq14_geo_sem_m2f
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0125_seq14_geo/4/models/40000.pt


log2_hashmap_size=22
desired_resolution=8192

python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --geo_init_method=idr   \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0  \
    --log2_hashmap_size=$log2_hashmap_size   --desired_resolution=$desired_resolution  \
    --freeze_geo=$freeze_geo  --ckpt_path=$ckpt_path  --wgt_sem_loss=1 \
    --separate_semantic=$separate_semantic   --label_name=$label_name  --num_layers_semantic_hidden=3    --semantic_layer_dim=128 \
    --lr=$lr    --balance_weight=True   --num_semantic_classes=5  

