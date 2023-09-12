#  测试新的scaling的效果： 2023/08/08   现在把scaling也用在了hash上

#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5

# wgt_depth_mse_loss=10

# exp_name=logs_dji/0808_1_dsnerf_mse_${wgt_depth_mse_loss}_new_scale

# dataset_path=/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan
# config_file=dji/cuhksz_ray_xml.yaml
# dataset_type=memory_depth_dji


# separate_semantic=True  # True False
# network_type='gpnerf'     #  gpnerf   sdf
# label_name='m2f_new'      # merge    m2f

# batch_size=20480
# train_iterations=150000
# val_interval=10000
# ckpt_interval=50000

# pos_xyz_dim=10

# enable_semantic=False

# python gp_nerf/train.py  --wandb_id  None   --exp_name  $exp_name   --enable_semantic  $enable_semantic  --pos_xyz_dim   $pos_xyz_dim  \
#     --label_name   $label_name   --separate_semantic   $separate_semantic    --network_type    $network_type   --config_file  $config_file     \
#     --dataset_path  $dataset_path       \
#     --train_iterations   $train_iterations     --val_interval  $val_interval    --ckpt_interval   $ckpt_interval   --batch_size  $batch_size   \
#     --dataset_type $dataset_type   --depth_dji_loss   True     --wgt_sigma_loss  0    --wgt_depth_mse_loss  $wgt_depth_mse_loss





exp_name='logs_dji/0808_2_sdf_as_gpnerf'


dataset_path=/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan
config_file=dji/cuhksz_ray_xml.yaml
dataset_type=memory

separate_semantic=True  # True False
network_type='sdf'     #  gpnerf   sdf
label_name='m2f_new'      # merge    m2f

batch_size=20480
train_iterations=100000
val_interval=10000
ckpt_interval=10000

pos_xyz_dim=10

enable_semantic=False
use_scaling=True

python gp_nerf/train.py  --wandb_id  None   --exp_name  $exp_name   --enable_semantic  $enable_semantic  --pos_xyz_dim   $pos_xyz_dim  \
    --label_name   $label_name   --separate_semantic   $separate_semantic    --network_type    $network_type   --config_file  $config_file     \
    --dataset_path  $dataset_path       \
    --train_iterations   $train_iterations     --val_interval  $val_interval    --ckpt_interval   $ckpt_interval   --batch_size  $batch_size   \
    --dataset_type $dataset_type     --use_scaling  $use_scaling   --sdf_as_gpnerf   True   