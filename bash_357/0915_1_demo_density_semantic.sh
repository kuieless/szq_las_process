#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


dataset_path=/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan
config_file=dji/cuhksz_ray_xml.yaml


batch_size=10240
train_iterations=200000
val_interval=10000
ckpt_interval=50000

network_type=gpnerf_nr3d     #  gpnerf   sdf
dataset_type=memory_depth_dji



use_scaling=False

separate_semantic=False
enable_semantic=True
freeze_geo=True
label_name=m2f


sampling_mesh_guidance=True


ckpt_path=logs_dji/0915_1_demo_density/0/models/200000.pt

exp_name=logs_dji/0915_1_demo_density_semantic


python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --freeze_geo=$freeze_geo  --ckpt_path=$ckpt_path  --wgt_sem_loss=1 \
    --separate_semantic=$separate_semantic   --label_name=$label_name
