#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4

# 测试idr情况下，
# 4. w/o depth, w/o fine
# 5. w/o depth,        fine
# 6. depth, w/o fine
# 7. depth,        fine   
#  如果都不需要fine sampling过滤，则拿掉， 如果需要，看看depth能不能缓解

dataset_path=/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels
config_file=configs/residence.yaml


batch_size=10240
train_iterations=200000
val_interval=5000
ckpt_interval=50000

network_type=sdf_nr3d     #  gpnerf   sdf
dataset_type=memory

use_scaling=False
sampling_mesh_guidance=False

separate_semantic=True
enable_semantic=True
freeze_geo=True
ckpt_path=logs_dji/0912_residence_sdf/0/models/200000.pt
label_name=merge 

exp_name=logs_dji/0913_residence_sdf_semantic_separate


python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --freeze_geo=$freeze_geo  --ckpt_path=$ckpt_path  --wgt_sem_loss=1 \
      --separate_semantic=$separate_semantic   --label_name=$label_name
