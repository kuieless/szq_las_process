#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4




# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
#     --rgbs_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/train/rgbs



#  这个代码是用来 把m2f crop2 的图片结果和sam结合， 需要先跑完m2f crop2

python tools/segment_anything/helpers/combine_sam_m2f2.py  \
    --sam_features_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/val/sam_features  \
    --rgbs_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926/val/rgbs    \
    --labels_m2f_path=/data/yuqi/code/Mask2Former_a/output_useful/yingrenshi_crop_ds2_val/labels_m2f_final  \
    --output_path=/data/yuqi/code/Mask2Former_a/output_useful/yingrenshi_crop_ds2_val/labels_m2f_final_sam