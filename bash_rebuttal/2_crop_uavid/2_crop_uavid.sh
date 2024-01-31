

python /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/2_crop_uavid/2_crop_uavid.py  \
    --input_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/2_seq14/0126_seq14_geo_sem_ours/0/eval_200000/val_rgbs/alpha_pred_label/000002_pred_label.jpg  \
    --output_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/2_crop_uavid/figure  \
    --custom_name=uavid_02_ours
    


python /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/2_crop_uavid/2_crop_uavid.py  \
    --input_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/2_seq14/0126_seq14_geo_sem_m2f/1/eval_200000/val_rgbs/alpha_pred_label/000002_pred_label.jpg  \
    --output_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/2_crop_uavid/figure  \
    --custom_name=uavid_02_m2f
    


python /data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/2_crop_uavid/2_crop_uavid.py  \
    --input_path=/data/yuqi/code/GP-NeRF-semantic/logs_rebuttal/2_seq14/0126_seq14_geo_sem_ours/0/eval_200000/val_rgbs/alpha_gt_label/000002_gt_label.jpg  \
    --output_path=/data/yuqi/code/GP-NeRF-semantic/bash_rebuttal/2_crop_uavid/figure  \
    --custom_name=uavid_02_gt
    