#!/bin/bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=3

config_file=configs/instancebuilding1_jx.yaml

dataset_path=/data/yuqi/Datasets/InstanceBuilding/3D/scene1/colmap_jx_new


batch_size=10240
train_iterations=200000
val_interval=200
ckpt_interval=20000

network_type=gpnerf_nr3d     #  gpnerf   sdf
dataset_type=memory_depth_dji_instance_crossview


enable_semantic=False
use_scaling=False
sampling_mesh_guidance=False

enable_semantic=False
freeze_geo=True
label_name=gt
separate_semantic=True
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0129_ib1_geo/0/models/140000.pt

depth_dji_loss=False
wgt_depth_mse_loss=0
lr=0.01

log2_hashmap_size=22
desired_resolution=8192

instance_loss_mode=linear_assignment

enable_instance=True
instance_name=instances_mask_0.001_depth20
num_instance_classes=100
exp_name=logs_rebuttal/0130_ib1_geo_$instance_loss_mode-$instance_name


python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0 \
    --log2_hashmap_size=$log2_hashmap_size   --desired_resolution=$desired_resolution  \
    --freeze_geo=$freeze_geo  --ckpt_path=$ckpt_path  --wgt_sem_loss=1 \
    --separate_semantic=$separate_semantic   --label_name=$label_name  --num_layers_semantic_hidden=3    --semantic_layer_dim=128 \
    --use_subset=True      --lr=$lr    --balance_weight=True   --num_semantic_classes=5   \
    --enable_instance=$enable_instance   --freeze_semantic=True  --instance_name=$instance_name   \
    --instance_loss_mode=$instance_loss_mode  \
    --num_instance_classes=$num_instance_classes  --debug=True
