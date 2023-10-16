#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5

# dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/


python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
    --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/augument/1015_far0.5/pred_rgb


# #  visualize
# python tools/segment_anything/helpers/get_mask_autogenerate3.py   \
#     --image_path=$dataset_path/rgbs \
#     --sam_feat_path=$dataset_path/sam_features  \
#     --output_path='/data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/SAM_mask_autogenerate_yingrenshi_train'









