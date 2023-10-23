#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


dataset_path=/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds
config_file=configs/longhua.yaml


batch_size=40960
train_iterations=200000
val_interval=10000
ckpt_interval=10000

network_type=gpnerf_nr3d     #  gpnerf   sdf
dataset_type=memory_depth_dji

use_scaling=False
sampling_mesh_guidance=True

enable_semantic=True

freeze_geo=True
label_name=1018_ml_fusion_0.3   # 这里标签虽然是22-23号生成的，但也取名叫这个
separate_semantic=True
ckpt_path=logs_dji/1023_longhua_b1_density_depth_hash22_fusion0.3_car2_semantic/0/models/40000.pt

# depth_dji_loss=True
# wgt_depth_mse_loss=1
lr=0.01
exp_name=logs_dji/1023_longhua_b1_density_depth_hash22_fusion0.3_car2_semantic

log2_hashmap_size=22
desired_resolution=8192

python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --log2_hashmap_size=$log2_hashmap_size   --desired_resolution=$desired_resolution  \
    --freeze_geo=$freeze_geo  --ckpt_path=$ckpt_path  --wgt_sem_loss=1 \
    --separate_semantic=$separate_semantic   --label_name=$label_name  --num_layers_semantic_hidden=3    --semantic_layer_dim=128 \
    --use_subset=True      --lr=$lr    --balance_weight=True   --num_semantic_classes=5   \
    --train_scale_factor=1  --val_scale_factor=1  
