import os


# # #### contrastive 的对比  campus  
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 A_Supplementary_trajectory/campus/0/pred_rgb.mp4\
#           -i2 A_Supplementary_trajectory/campus/0/pred_depth.mp4\
#           -o A_Supplementary_trajectory_output/combine\
#           -c s1,t2-s1vws2,l0.5:1-s1vws2r,l1:0.5-s1,t10")


# # #### contrastive 的对比  campus  
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 A_Supplementary_trajectory/campus/0/pred_rgb.mp4\
#           -i2 A_Supplementary_trajectory/campus/0/instance.mp4\
#           -o A_Supplementary_trajectory_output/combine\
#           -c s1,t2-s1vws2,l0.5:1-s1vws2r,l1:0.5-s1,t0.5-s1vws2,l1:10")


# # #### contrastive 的对比  campus  
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 A_Supplementary_trajectory/campus/0/instance.mp4\
#           -i2 A_Supplementary_trajectory/campus/0/pred_label_alpha.mp4\
#           -o A_Supplementary_trajectory_output/combine\
#           -c s1,t3.5-s1vws2,l1:2-s2,t2.8")



# # #### contrastive 的对比  campus  
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 A_Supplementary_trajectory_output/combine/campus1.mp4\
#           -i2 A_Supplementary_trajectory_output/combine/campus2.mp4\
#           -o A_Supplementary_trajectory_output/combine\
#           -c s1,t6-s2,t10")

# ffmpeg -f concat -i A_Supplementary_trajectory_output/campus.txt  -c copy A_Supplementary_trajectory_output/campus_output.mp4