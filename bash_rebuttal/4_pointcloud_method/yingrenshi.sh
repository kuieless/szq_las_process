#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=2

label_name=1018_ml_fusion_0.3

/data/yuqi/anaconda3/envs/gpnerf/bin/python /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/1010_3d_get_2d_label_gy_filter.py    \
    --dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926  \
    --metaXml_path=/data/yuqi/Datasets/DJI/others/origin/Yingrenshi_20230926_origin/terra_point_ply/metadata.xml   \
    --output_pat=zyq_rebuttal_4/yingrenshi_$label_name \
    --load_ply_path=/data2/jxchen/jxchen_copy_1121/data/jxchen/dataset/dji/Yingrenshi/Yingrenshi_fine-registered.txt    \
    --label_name=$label_name