#!/bin/bash
export OMP_NUM_THREADS=4
gpu=${1:-6}
# exp=${2:-''}
export CUDA_VISIBLE_DEVICES=${gpu}

exp_name='logs_357/0512_train_sam_online_sci'    #${exp}
dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2="sci-art" # 'sci-art'  "building"  "rubble"  "quad"   "sci-art"  "campus"

enable_semantic=True
network_type='gpnerf'  #  gpnerf  sdf
label_name='m2f_new'
dataset_type='sam'
freeze_geo=True
val_type='val'

online_sam_label=${2}    #True
remove_cluster=False

batch_size=4          #4 #25600 # 128*128
sample_ray_num=4096    #  2304
sam_sample_each=256

train_iterations=20000
val_interval=1000
ckpt_interval=1000


args=""
chunk_path=/data/yuqi/Datasets/MegaNeRF/$dataset1/${dataset2}_chunk-labels-$label_name-down4
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_357/0504_G_geo_${dataset2}/0/models/200000.pt
args=${args}" --ckpt_path $ckpt_path  --online_sam_label  $online_sam_label  --sam_sample_each  $sam_sample_each   --remove_cluster  $remove_cluster  "


group_loss=False
wgt_group_loss=0
sample_random_num=4096
args=${args}" --add_random_rays  False  --sample_random_num  $sample_random_num   --group_loss  $group_loss  --wgt_group_loss  $wgt_group_loss "


debug=False
# debug=True
if [[ $debug == 'True' ]]; then
    echo Deubg
    exp_name=${exp_name}_debug
    args=$args" --debug True --val_interval 100 --logger_interval 5"
fi
echo DEBUG=${args}


python gp_nerf/train.py  --network_type   $network_type --enable_semantic  $enable_semantic  --config_file  configs/$dataset2.yaml   \
    --dataset_type $dataset_type \
    --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-labels --chunk_paths   $chunk_path \
    --exp_name  $exp_name  --train_iterations   $train_iterations  --val_interval  $val_interval    --ckpt_interval   $ckpt_interval  --label_name   $label_name \
    --batch_size  $batch_size   --freeze_geo  $freeze_geo  --val_type  $val_type   \
    --sample_ray_num $sample_ray_num \
    $args 