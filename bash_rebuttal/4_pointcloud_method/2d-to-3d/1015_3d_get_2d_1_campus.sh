
###
# 这里给出了 
# 拉远视角（far）, 拿到nerf渲染的rgb，和经过m2f处理得到m2f_label后 

# 1. 将多视角标签投影到gt点云上，
# 2. 再由gt点云得到 label_pc ， 后续拿去eval的过程

# 3. 把生成的 label_pc 挪到指定位置，改名，跑 eval


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5

# #######################################################################################################

# #需要修改的值
# # far_project_m2f_paths   新的标签路径
# label_test=labels_1018_ml_fusion_0.3
# far_project_m2f_paths=/data/yuqi/Datasets/DJI/Campus_new/train/$label_test
# output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/campus_$label_test

# dataset_path=/data/yuqi/Datasets/DJI/Campus_new
# metaXml_path=/data2/jxchen/jxchen_copy_1121/data/jxchen/dataset/dji/campus/metadata.xml
# load_ply_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/gt/yuehai_gt_label_fineReg.txt


# # #  1.  3d get 2d label

# echo '1 start'
# /data/yuqi/anaconda3/envs/gpnerf/bin/python /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/1015_1_3d_get_2d_label_gy_filter_1015_farproject3.py  \
#     --dataset_path=$dataset_path  \
#     --exp_name='logs_357/test' \
#     --output_path=$output_path  \
#     --far_project_m2f_paths=$far_project_m2f_paths  \
#     --metaXml_path=$metaXml_path    \
#     --load_ply_path=$load_ply_path

# #######################################################################################################




# #需要修改的值
# # far_project_m2f_paths   新的标签路径


dataset_path=/data/yuqi/Datasets/DJI/Campus_new
metaXml_path=/data2/jxchen/jxchen_copy_1121/data/jxchen/dataset/dji/campus/metadata.xml
load_ply_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/data/gt/yuehai_gt_label_fineReg.txt


label_test=labels_m2f
far_project_m2f_paths=/data/yuqi/Datasets/DJI/Campus_new/train/$label_test
output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/campus_$label_test-new


# #  1.  3d get 2d label

echo '1 start'
/data/yuqi/anaconda3/envs/gpnerf/bin/python /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/1015_1_3d_get_2d_label_gy_filter_1015_farproject3.py  \
    --dataset_path=$dataset_path  \
    --exp_name='logs_357/test' \
    --output_path=$output_path  \
    --far_project_m2f_paths=$far_project_m2f_paths  \
    --metaXml_path=$metaXml_path    \
    --load_ply_path=$load_ply_path



#######################################################################################################









