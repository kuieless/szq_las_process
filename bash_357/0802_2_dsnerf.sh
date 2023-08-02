#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7

exp_name='logs_357/0802_2_dsnerf'


dataset_path=/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan
config_file=dji/cuhksz_ray_xml.yaml
dataset_type=memory_depth_dji


separate_semantic=True  # True False
network_type='gpnerf'     #  gpnerf   sdf
label_name='m2f_new'      # merge    m2f

batch_size=20480
train_iterations=200000
val_interval=50000
ckpt_interval=50000

pos_xyz_dim=10

enable_semantic=False

python gp_nerf/train.py  --wandb_id  None   --exp_name  $exp_name   --enable_semantic  $enable_semantic  --pos_xyz_dim   $pos_xyz_dim  \
    --label_name   $label_name   --separate_semantic   $separate_semantic    --network_type    $network_type   --config_file  $config_file     \
    --dataset_path  $dataset_path       \
    --train_iterations   $train_iterations     --val_interval  $val_interval    --ckpt_interval   $ckpt_interval   --batch_size  $batch_size   \
    --dataset_type $dataset_type   --depth_dji_loss   True     --wgt_sigma_loss  0.0001









