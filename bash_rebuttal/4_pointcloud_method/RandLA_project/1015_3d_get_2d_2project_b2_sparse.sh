
###
# 这里给出了 
# 拉远视角（far）, 拿到nerf渲染的rgb，和经过m2f处理得到m2f_label后 

# 1. 将多视角标签投影到gt点云上，
# 2. 再由gt点云得到 label_pc ， 后续拿去eval的过程

# 3. 把生成的 label_pc 挪到指定位置，改名，跑 eval


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1

output_path=/data/yuqi/code/GP-NeRF-semantic/zyq_rebuttal_4/RandLaNet/b2_sparse
dataset_path=/data/yuqi/Datasets/DJI/Longhua_block2_20231020_ds
load_ply_path=/data/yuqi/code/Urban3D-2021-2nd/data_b2_sparse/log_test/test_preds/visualize_label_ours.ply


# 2. 上述代码在 output_path 下生成results.ply, 由下面代码处理得到  label_pc  把label投影到val视角
echo '1 end, 2 start'
/data/yuqi/anaconda3/envs/gpnerf/bin/python  /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/4_pointcloud_method/1015_2_project_pointcloud_label_to_image_cuda1.py   \
    --dataset_path=$dataset_path  \
    --exp_name='logs_357/test'  \
    --output_path=$output_path   \
    --load_ply_path=$load_ply_path



# # label_name_3d_to_2d=labels_pc_1015_farproject_ori_expand
# eval_output=$output_path/eval

# echo '2 end, need to move and copy ' $output_path/$label_name_3d_to_2d

# cp -r  $output_path/expand_labels_pc   $output_path/$label_name_3d_to_2d    
# cp -r  $output_path/$label_name_3d_to_2d    $dataset_path/val/$label_name_3d_to_2d

# echo 'eval the label'

# # 3. 将 label_pc  挪至 /data/yuqi/Datasets/DJI/Yingrenshi_20230926/val 并改名 (label_name_3d_to_2d)， 进行指标计算
# /home/yuqi/anaconda3/envs/gpnerf/bin/python   gp_nerf/eval_val_3d_to_2d.py  \
#     --dataset_path=$dataset_path \
#     --config_file=configs/yingrenshi.yaml    \
#     --exp_name=$eval_output  \
#     --label_name_3d_to_2d=$label_name_3d_to_2d


