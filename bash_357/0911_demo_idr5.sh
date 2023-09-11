#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5

# 测试idr情况下，
# 4. w/o depth, w/o fine
# 5. w/o depth,        fine
# 6. depth, w/o fine
# 7. depth,        fine   
#  如果都不需要fine sampling过滤，则拿掉， 如果需要，看看depth能不能缓解

dataset_path=/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan
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


depth_dji_loss=False
wgt_depth_mse_loss=0
fine_sample_filter=True

exp_name=logs_dji/0911_demo_idr5


python gp_nerf/train.py  --exp_name  $exp_name   --enable_semantic  $enable_semantic  \
    --network_type   $network_type   --config_file  $config_file   \
    --dataset_path  $dataset_path  --batch_size  $batch_size  \
    --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type     --use_scaling  $use_scaling  \
    --sampling_mesh_guidance   $sampling_mesh_guidance   --sdf_as_gpnerf  True  \
    --geo_init_method=idr   \
    --depth_dji_loss   $depth_dji_loss   --wgt_depth_mse_loss  $wgt_depth_mse_loss  --wgt_sigma_loss  0  \
    --fine_sample_filter $fine_sample_filter
