#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7





# # 如果还没计算过sam feature  先计算
# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
#     --rgbs_path=/data/yuqi/Datasets/DJI/Longhua_block1_20231020_ds/val/rgbs


#  这个代码是用来 把m2f crop2 的图片结果和sam结合， 需要先跑完m2f crop2

python tools/segment_anything/helpers/combine_sam_m2f2.py  \
    --sam_features_path=/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds/val/sam_features  \
    --rgbs_path=/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds/val/rgbs    \
    --labels_m2f_path=/data/yuqi/code/Mask2Former_a/output/longhua_panoptic_b2_val_crop_ds2/labels_m2f_final  \
    --output_path=/data/yuqi/code/Mask2Former_a/output/longhua_panoptic_b2_val_crop_ds2/labels_m2f_final_sam