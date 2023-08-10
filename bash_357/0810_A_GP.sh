#!/bin/bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=5

exp_name=logs_dji/0810_A_GP

dataset_path=/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan
config_file=dji/cuhksz_ray_xml.yaml


batch_size=20480
train_iterations=200000
val_interval=10000
ckpt_interval=50000

network_type='gpnerf'     #  gpnerf   sdf
dataset_type=memory_depth_dji  


depth_dji_loss=False
wgt_depth_mse_loss=0


depth_dji_type='mesh'   #  las  mesh
sampling_mesh_guidance=False

python gp_nerf/train.py  --exp_name  $exp_name  --config_file  $config_file   --dataset_path  $dataset_path   \
    --train_iterations   $train_iterations  --val_interval  $val_interval   --ckpt_interval   $ckpt_interval   --batch_size  $batch_size   \
    --network_type    $network_type     --dataset_type $dataset_type   \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0     \
    --depth_dji_type   $depth_dji_type   --sampling_mesh_guidance    $sampling_mesh_guidance 







exp_name=logs_dji/0810_B_GP_las_depth

dataset_path=/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan
config_file=dji/cuhksz_ray_xml.yaml


batch_size=20480
train_iterations=500000
val_interval=10000
ckpt_interval=50000

network_type='gpnerf'     #  gpnerf   sdf
dataset_type=memory_depth_dji  


depth_dji_loss=True
wgt_depth_mse_loss=10
depth_dji_type='las'   #  las  mesh

sampling_mesh_guidance=False

python gp_nerf/train.py  --exp_name  $exp_name  --config_file  $config_file   --dataset_path  $dataset_path   \
    --train_iterations   $train_iterations  --val_interval  $val_interval   --ckpt_interval   $ckpt_interval   --batch_size  $batch_size   \
    --network_type    $network_type     --dataset_type $dataset_type   \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0     \
    --depth_dji_type   $depth_dji_type   --sampling_mesh_guidance    $sampling_mesh_guidance

