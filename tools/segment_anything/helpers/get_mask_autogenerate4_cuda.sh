#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5

# dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/


# #  visualize
# python tools/segment_anything/helpers/get_mask_autogenerate3.py   \
#     --image_path=$dataset_path/rgbs \
#     --sam_feat_path=$dataset_path/sam_features  \
#     --output_path='/data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/SAM_mask_autogenerate_yingrenshi_train'





# #  visualize
# python tools/segment_anything/helpers/get_mask_autogenerate4_cuda.py   \
#     --image_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/37/eval_200000/val_rgbs/pred_rgb \
#     --sam_feat_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/37/eval_200000/val_rgbs/sam_features  \
#     --output_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1003_yingrenshi_density_depth_hash22_semantic/37/eval_200000/val_rgbs/sam_features_vis




#  visualize
python tools/segment_anything/helpers/get_mask_autogenerate4_cuda.py   \
    --image_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs \
    --sam_feat_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/sam_features  \
    --output_path=zyq/1028_sam_vis


