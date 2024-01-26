#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1

dataset_path=/data/yuqi/jx_rebuttal/seq14_ds_10_val/postprocess
val_dataset_path=$dataset_path/val
output_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/0125_seq14_geo_farview0.3_val_agu/8/eval_40000/val_rgbs
far_paths=$output_path/1_labels_m2f_only_sam/labels_merge
render_type=render_far0.3_val
eval=True


# # 原始图像进行sam feature提取   
# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
#     --rgbs_path=$val_dataset_path/rgbs



# # 先对nerf渲染的far图像进行sam feature提取   
# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
#     --rgbs_path=$output_path/pred_rgb



# #  接下来用 far的sam feature 和 m2f的building结果进行结合   
# python tools/segment_anything/helpers/combine_sam_m2f3_only_sam.py  \
#     --sam_features_path=$output_path/sam_features  \
#     --rgbs_path=$output_path/pred_rgb   \
#     --labels_m2f_path=$output_path/mask2former_output/labels_m2f  \
#     --output_path=$output_path/1_labels_m2f_only_sam






# 这里 把far图像投影回原始图
/data/yuqi/anaconda3/envs/gpnerf/bin/python   /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/seq_far_view/1015_0_3d_get_2d_label_gy_filter_far_to_ori.py \
    --dataset_path=$dataset_path  \
    --exp_name='logs_357/test'  \
    --far_paths=$far_paths   \
    --output_path=$output_path/2_project_to_ori_gt_only_sam \
    --render_type=$render_type  \
    --eval=$eval



#### campus 1104   在campus上发现效果不好，需要去掉这个 ##################
######################################################################
#  1028  加一个投影图和sam结合 （也是只用能够识别到的部分）
python tools/segment_anything/helpers/combine_sam_m2f3_only_sam.py  \
    --sam_features_path=$val_dataset_path/sam_features  \
    --rgbs_path=$val_dataset_path/rgbs   \
    --labels_m2f_path=$output_path/2_project_to_ori_gt_only_sam/project_far_to_ori  \
    --output_path=$output_path/3_merge_project_and_sam
######################################################################



###############  0126  1  #################
# 用投影回原图的building替换掉原始图的building   1028  改成了 car的部分不换掉
/data/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/seq_far_view/1016_1_only_sam_building_label_replace_nocar.py  \
    --dataset_path=$dataset_path    \
    --exp_name=logs_357/test    \
    --only_sam_m2f_far_project_path=$output_path/3_merge_project_and_sam/labels_merge    \
    --output_path=$output_path/4_replace_building  \
    --eval=$eval


# ###############  0126  2  #################
# # 用投影回原图的building替换掉原始图的building   1028  改成了 car的部分不换掉
# /data/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/seq_far_view/1016_1_only_sam_building_label_replace_nocar.py  \
#     --dataset_path=$dataset_path    \
#     --exp_name=logs_357/test    \
#     --only_sam_m2f_far_project_path=$output_path/2_project_to_ori_gt_only_sam/project_far_to_ori    \
#     --output_path=$output_path/4_replace_building  \
#     --eval=$eval



# ########  !!!!!!   NOTE  improtant:  以上得到了替换building的结果，  接下来要把car tree换掉      
#  这个代码是用来 把m2f crop2 的图片结果和sam结合， 需要先跑完m2f crop2
# python tools/segment_anything/helpers/combine_sam_m2f2.py  \
#     --sam_features_path=$val_dataset_path/sam_features  \
#     --rgbs_path=$val_dataset_path/rgbs    \
#     --labels_m2f_path=/data/yuqi/code/Mask2Former_a/output/seq14_ds_10_val_ds2/labels_m2f_final  \
#     --output_path=/data/yuqi/code/Mask2Former_a/output/seq14_ds_10_val_ds2/labels_m2f_final_sam




/data/yuqi/anaconda3/envs/gpnerf/bin/python     /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/seq_far_view/1016_2_further_replace_car_tree_from_crop.py  \
    --dataset_path=$dataset_path    \
    --exp_name=logs_357/test    \
    --only_sam_m2f_far_project_path=$output_path/4_replace_building/replace_building_label  \
    --crop_m2f=/data/yuqi/code/Mask2Former_a/output/seq14_ds_10_val_ds2/labels_m2f_final_sam/labels_merge  \
    --output_path=$output_path/5_replace_building_cartree  \
    --eval=$eval

