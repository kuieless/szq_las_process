
###
# 这里给出了 
# 拉远视角（far）, 拿到nerf渲染的rgb，和经过m2f处理得到m2f_label后 

# 1. 将多视角标签投影到gt点云上，
# 2. 再由gt点云得到 label_pc ， 后续拿去eval的过程

# 3. 把生成的 label_pc 挪到指定位置，改名，跑 eval

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7



#需要修改的值
# far_project_m2f_paths   新的标签路径
dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926
output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/2d_to_3d/yingrenshi_labels_1018_ml_fusion_0.3

# 2. 上述代码在 output_path 下生成results.ply, 由下面代码处理得到  label_pc  把label投影到val视角
echo '1 end, 2 start'
/data/yuqi/anaconda3/envs/gpnerf/bin/python  /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/1015_2_project_pointcloud_label_to_image_cuda1.py   \
    --dataset_path=$dataset_path  \
    --exp_name='logs_357/test'  \
    --output_path=$output_path   \
    --load_ply_path=$output_path/results.ply





#需要修改的值
# far_project_m2f_paths   新的标签路径
output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/2d_to_3d/yingrenshi_labels_m2f

# 2. 上述代码在 output_path 下生成results.ply, 由下面代码处理得到  label_pc  把label投影到val视角
echo '1 end, 2 start'
/data/yuqi/anaconda3/envs/gpnerf/bin/python  /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/1015_2_project_pointcloud_label_to_image_cuda1.py   \
    --dataset_path=$dataset_path  \
    --exp_name='logs_357/test'  \
    --output_path=$output_path   \
    --load_ply_path=$output_path/results.ply
