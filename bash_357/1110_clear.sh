#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5
##两个放一起跑



config_file=configs/campus_new.yaml


batch_size=16384
train_iterations=100000
val_interval=20000
ckpt_interval=20000

use_subset=True
enable_semantic=True
freeze_semantic=True
freeze_geo=True
lr=0.01

enable_instance=True

instance_loss_mode=slow_fast
instance_name=instances_mask_0.001_depth
dataset_type=memory_depth_dji_instance_crossview
num_instance_classes=25

ckpt_path=logs_campus/1104_campus_density_depth_hash22_fusion1018_car2_semantic/0/models/200000.pt
exp_name=logs_campus/1110_campus_density_depth_hash22_instance_origin_sam_0.001_depth_crossview

only_train_building=False



python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  --config_file  $config_file   \
    --batch_size  $batch_size  --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type    --freeze_geo=$freeze_geo  --ckpt_path=$ckpt_path   \
    --use_subset=$use_subset      --lr=$lr       \
    --enable_instance=$enable_instance   --freeze_semantic=$freeze_semantic  --instance_name=$instance_name   \
    --instance_loss_mode=$instance_loss_mode  --num_instance_classes=$num_instance_classes \
    --only_train_building=$only_train_building



