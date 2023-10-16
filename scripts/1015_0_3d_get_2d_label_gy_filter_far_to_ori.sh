


export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4



/home/yuqi/anaconda3/envs/gpnerf/bin/python   /data/yuqi/code/GP-NeRF-semantic/scripts/1015_0_3d_get_2d_label_gy_filter_far_to_ori.py \
    --dataset_path='/data/yuqi/Datasets/DJI/Yingrenshi_20230926'  \
    --exp_name='logs_357/test'  \
    --far_paths='logs_dji/augument/1015_far0.5/mask2former_output/labels_m2f'   \
    --output_path='logs_dji/augument/1015_far0.5/project_to_ori_gt' \
    --render_type='render_far0.5'
    