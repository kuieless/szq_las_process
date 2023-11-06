#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1


# # 先对nerf渲染的far图像进行sam feature提取   
# python /data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/extract_embeddings.py   \
#     --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_fusion2d_eval/pred_rgb


# #  接下来用 far的sam feature 和 m2f的building结果进行结合   
# python tools/segment_anything/helpers/combine_sam_m2f3_only_sam.py  \
#     --sam_features_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_fusion2d_eval/sam_features  \
#     --rgbs_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_fusion2d_eval/pred_rgb   \
#     --labels_m2f_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_fusion2d_eval/mask2former_output/labels_m2f  \
#     --output_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_fusion2d_eval/1_labels_m2f_only_sam




dataset_path=/data/yuqi/Datasets/DJI/Campus_new
far_paths=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1104_campus_fusion2d_eval/1_labels_m2f_only_sam/labels_merge
output_path=/data/yuqi/code/GP-NeRF-semantic/logs_campus/1103_campus_far_0.3
render_type=render_far0.3_val


# # 这里 把far图像投影回原始图
# /home/yuqi/anaconda3/envs/gpnerf/bin/python   /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_3d_get_2d_label_gy_filter_far_to_ori.py \
#     --dataset_path=$dataset_path  \
#     --exp_name='logs_357/test'  \
#     --far_paths=$far_paths   \
#     --output_path=$output_path/2_project_to_ori_gt_only_sam \
#     --render_type=$render_type  \
#     --eval=True



# #### campus 1104   在campus上发现效果不好，需要去掉这个 ##################
#######################################################################
# #  1028  加一个投影图和sam结合 （也是只用能够识别到的部分）
# python tools/segment_anything/helpers/combine_sam_m2f3_only_sam.py  \
#     --sam_features_path=$dataset_path/train/sam_features  \
#     --rgbs_path=$dataset_path/train/rgbs   \
#     --labels_m2f_path=$output_path/2_project_to_ori_gt_only_sam/project_far_to_ori  \
#     --output_path=$output_path/3_merge_project_and_sam
#######################################################################



# # 用投影回原图的building替换掉原始图的building   1028  改成了 car的部分不换掉
# /home/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/scripts/1016_1_only_sam_building_label_replace_nocar.py  \
#     --dataset_path=$dataset_path    \
#     --exp_name=logs_357/test    \
#     --only_sam_m2f_far_project_path=$output_path/2_project_to_ori_gt_only_sam/project_far_to_ori    \
#     --output_path=$output_path/4_replace_building \
#     --eval=True




# ########  !!!!!!   NOTE  improtant:  以上得到了替换building的结果，  接下来要把car tree换掉      
# 用/data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/helpers/combine_sam_m2f2_ds2crop_1022process_b1.sh


/home/yuqi/anaconda3/envs/gpnerf/bin/python     /data/yuqi/code/GP-NeRF-semantic/scripts/1016_2_further_replace_car_tree_from_crop_covercar.py  \
    --dataset_path=$dataset_path    \
    --exp_name=logs_357/test    \
    --only_sam_m2f_far_project_path=$output_path/4_replace_building_1018/replace_building_label  \
    --crop_m2f=/data/yuqi/code/Mask2Former_a/output/Campus_crop_ds2_train/labels_m2f_final_sam/labels_merge  \
    --output_path=$output_path/5_replace_building_cartree_covercar  \
    # --eval=True




# # ##  3.先注释， 最后跑
# # 做一个可视化的对比结果
# /home/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_compare_ori_far.py  \
#     --dataset_path=$dataset_path    \
#     --exp_name='logs_357/test'  \
#     --project_path=$output_path/project_far_to_ori_replace_building_car_tree/replace_cartree_label  \
#     --output_path=$output_path/compare_vis