#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6



# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2.py  \
#     --dataset_path=/data/yuqi/Datasets/DJI/Campus_new   \
#     --output_path=zyq/1107_get_instance_mask_train_campus  \
#     --threshold=0.001   \
#     --points_per_side=32


# ####2023.11.10
# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2.py  \
#     --dataset_path=/data/yuqi/Datasets/DJI/Campus_new   \
#     --output_path=zyq/1110_get_instance_mask_train_campus  \
#     --threshold=0   \
#     --points_per_side=32



# ####2023.11.10
# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2.py  \
#     --dataset_path=/data/yuqi/Datasets/DJI/Campus_new   \
#     --output_path=zyq/1110_get_instance_mask_train_campus  \
#     --threshold=0   \
#     --points_per_side=64



####2023.11.12  val
python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/1019_get_sammask_autogenerate_2.py  \
    --dataset_path=/data/yuqi/Datasets/DJI/Campus_new   \
    --output_path=zyq/1110_get_instance_mask_val_campus  \
    --threshold=0   \
    --points_per_side=64  \
    --eval=True