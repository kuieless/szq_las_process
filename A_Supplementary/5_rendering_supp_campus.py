import os


# #### contrastive 的对比  campus  
os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
          -i1 A_Supplementary_trajectory/campus_old/0/pred_rgb.mp4\
          -i2 A_Supplementary_trajectory/campus_old/0/pred_depth.mp4\
          -o A_Supplementary_trajectory_output/combine\
          -c s1,t2-s1vws2,l0.5:1-s1vws2r,l0.5:0")


# # #### contrastive 的对比  campus  
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 A_Supplementary_trajectory/campus_old/0/pred_label_alpha.mp4 \
#           -i2 A_Supplementary_trajectory/campus_old/0/instance.mp4\
#           -o A_Supplementary_trajectory_output/combine\
#           -c s1vs2,t12")


