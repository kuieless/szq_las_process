#!/bin/bash
export OMP_NUM_THREADS=4
gpu=${1:-6}
# exp=${2:-''}
export CUDA_VISIBLE_DEVICES=${gpu}

exp_name='logs_357/0510_train_sam_resi'    #${exp}
dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2="residence" # 'sci-art'  "building"  "rubble"  "quad"   "sci-art"  "campus"
wandb_id=None  #gpnerf_semantic   None

wandb_run_name=$exp_name
separate_semantic=True  # True False
enable_semantic=True
network_type='gpnerf'  #  gpnerf  sdf
label_name='merge_new'

sample_random_num=4096
batch_size=8            #4 #25600 # 128*128
sample_ray_num=1024     #  2304
sam_sample_each=512

train_iterations=40000
val_interval=300
ckpt_interval=300
pos_xyz_dim=10

dataset_type='sam'
sam_loss=CSLoss
freeze_geo=True
wgt_group_loss=0.1


args=""
chunk_path=/data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name-down4
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_357/0504_G3_geo_residence/0/models/200000.pt
args=${args}"--ckpt_path $ckpt_path --no_resume_ckpt_state  "


args=${args}" --add_random_rays  False  --sample_random_num  $sample_random_num --sam_loss  $sam_loss  --freeze_geo  $freeze_geo  --wgt_group_loss  $wgt_group_loss "

debug=False
# debug=True
if [[ $debug == 'True' ]]; then
    echo Deubg
    exp_name=${exp_name}_debug
    args=$args" --debug True --val_interval 10000 --logger_interval 5"
fi
echo DEBUG=${args}

python gp_nerf/train.py  --network_type   $network_type --enable_semantic  $enable_semantic  --config_file  configs/$dataset2.yaml   \
    --dataset_type $dataset_type \
    --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels --chunk_paths   $chunk_path \
    --exp_name  $exp_name  --train_iterations   $train_iterations  --val_interval  $val_interval    --ckpt_interval   $ckpt_interval  --label_name   $label_name \
    --batch_size  $batch_size    \
    --wandb_id  $wandb_id   --wandb_run_name  $wandb_run_name   \
    --sample_ray_num $sample_ray_num \
    $args 