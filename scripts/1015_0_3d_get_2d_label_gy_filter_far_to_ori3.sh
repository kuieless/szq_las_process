


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=6

dataset_path=/data/yuqi/Datasets/DJI/Yingrenshi_20230926
far_paths=logs_dji/augument/1015_far0.3/mask2former_output/labels_m2f
output_path=logs_dji/augument/1015_far0.3/project_to_ori_gt_depth
render_type=render_far0.3

/home/yuqi/anaconda3/envs/gpnerf/bin/python   /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_3d_get_2d_label_gy_filter_far_to_ori.py \
    --dataset_path=$dataset_path  \
    --exp_name='logs_357/test'  \
    --far_paths=$far_paths   \
    --output_path=$output_path \
    --render_type=$render_type
    
/home/yuqi/anaconda3/envs/gpnerf/bin/python    /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_compare_ori_far.py  \
    --dataset_path=$dataset_path    \
    --exp_name='logs_357/test'  \
    --project_path=$output_path/project_far_to_ori  \
    --output_path=$output_path/compare_vis
    