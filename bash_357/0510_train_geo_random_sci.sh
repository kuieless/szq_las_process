#!/bin/bash
export OMP_NUM_THREADS=4
# gpu=${1:-6}
# exp=${2:-''}
export CUDA_VISIBLE_DEVICES=7  #${gpu}

exp_name='logs_357/0510_train_geo_random_sci'    #${exp}
dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2="sci-art" # 'sci-art'  "building"  "rubble"  "quad"   "sci-art"  "campus"
wandb_id=None  #gpnerf_semantic   None

wandb_run_name=$exp_name
separate_semantic=True  # True False
enable_semantic=False
network_type='gpnerf'  #  gpnerf  sdf
label_name='merge_new'

batch_size=8            #4 #25600 # 128*128
sample_ray_num=1024     #  2304
sample_random_num=4096

train_iterations=100000
val_interval=10000
ckpt_interval=10000
pos_xyz_dim=10

dataset_type='memory_depth'
wgt_depth_loss=0.0000

args=""
# chunk_path=/data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name-down4
# ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_357/0504_G3_geo_residence/0/models/200000.pt
# args=${args}"--ckpt_path $ckpt_path --no_resume_ckpt_state"


args=${args}" --normal_loss False  --depth_loss False --add_random_rays  True  --sample_random_num  $sample_random_num"

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
    --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels    \
    --exp_name  $exp_name  --train_iterations   $train_iterations  --val_interval  $val_interval    --ckpt_interval   $ckpt_interval  --label_name   $label_name \
    --batch_size  $batch_size    \
    --wandb_id  $wandb_id   --wandb_run_name  $wandb_run_name   \
    --wgt_depth_loss $wgt_depth_loss --sample_ray_num $sample_ray_num \
    $args 




exp_name='logs_357/0510_train_geo_random_campus'    #${exp}
dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2="campus" # 'sci-art'  "building"  "rubble"  "quad"   "sci-art"  "campus"
wandb_id=None  #gpnerf_semantic   None

wandb_run_name=$exp_name
separate_semantic=True  # True False
enable_semantic=False
network_type='gpnerf'  #  gpnerf  sdf
label_name='merge_new'

batch_size=8            #4 #25600 # 128*128
sample_ray_num=1024     #  2304
sample_random_num=4096

train_iterations=100000
val_interval=10000
ckpt_interval=10000
pos_xyz_dim=10

dataset_type='memory_depth'
wgt_depth_loss=0.0000

args=""
# chunk_path=/data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name-down4
# ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_357/0504_G3_geo_residence/0/models/200000.pt
# args=${args}"--ckpt_path $ckpt_path --no_resume_ckpt_state"


args=${args}" --normal_loss False  --depth_loss False --add_random_rays  True  --sample_random_num  $sample_random_num"

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
    --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels    \
    --exp_name  $exp_name  --train_iterations   $train_iterations  --val_interval  $val_interval    --ckpt_interval   $ckpt_interval  --label_name   $label_name \
    --batch_size  $batch_size    \
    --wandb_id  $wandb_id   --wandb_run_name  $wandb_run_name   \
    --wgt_depth_loss $wgt_depth_loss --sample_ray_num $sample_ray_num \
    $args 