#!/bin/bash
# 前面已经得到了比较好的sdf（集成长方体hash）
# 1. 使用neus的alpha， 测试depth loss的效果
# 2. 使用neuralsim的alpha，看看+plane效果如何
# 3. 使用neuralsim的alpha，看看+plane效果如何, 以及加上depth loss
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4


dataset_path=/data/yuqi/Datasets/DJI/subset
config_file=dji/cuhksz_ray_xml.yaml


batch_size=10240
train_iterations=200000
val_interval=10000
ckpt_interval=50000

network_type=sdf_nr3d     #  gpnerf   sdf
dataset_type=memory_depth_dji

enable_semantic=False
use_scaling=False
sampling_mesh_guidance=True


depth_dji_loss=True
wgt_depth_mse_loss=0.01

exp_name=logs_dji/0904_1_dji_subset_sdf_nr3d_depth_$wgt_depth_mse_loss


python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0
