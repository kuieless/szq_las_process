#!/bin/bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=7

exp_name=logs_dji/0818_D_GP_mesh_depth_guidence_large

dataset_path=/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan
config_file=dji/cuhksz_ray_xml.yaml


batch_size=20480
train_iterations=2000000
val_interval=100000
ckpt_interval=100000

network_type='gpnerf'     #  gpnerf   sdf
dataset_type=memory_depth_dji  


depth_dji_loss=True
wgt_depth_mse_loss=10


depth_dji_type='mesh'   #  las  mesh
sampling_mesh_guidance=True

python gp_nerf/train.py  --exp_name  $exp_name  --config_file  $config_file   --dataset_path  $dataset_path   \
    --train_iterations   $train_iterations  --val_interval  $val_interval   --ckpt_interval   $ckpt_interval   --batch_size  $batch_size   \
    --network_type    $network_type     --dataset_type $dataset_type   \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0     \
    --depth_dji_type   $depth_dji_type   --sampling_mesh_guidance    $sampling_mesh_guidance    \
    --log2_hashmap_size  22  --desired_resolution   8192      --hash_feat_dim  4
