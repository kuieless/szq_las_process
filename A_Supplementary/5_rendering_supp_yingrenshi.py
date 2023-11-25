import os


# # #### 1
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 A_Supplementary_trajectory/yingrenshi/0/pred_rgb.mp4\
#           -i2 A_Supplementary_trajectory/yingrenshi/0/pred_depth.mp4\
#           -o A_Supplementary_trajectory_output/combine\
#           -c s1,t2-s1vws2,l0.5:1-s1vws2r,l1:0.5-s1,t10\
#           ")


# # #### 2 
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 A_Supplementary_trajectory/yingrenshi/0/pred_rgb.mp4\
#           -i2 A_Supplementary_trajectory/yingrenshi/0/pred_label_alpha.mp4\
#           -o A_Supplementary_trajectory_output/combine\
#           -c s1,t2-s1vws2,l0.5:1-s1vws2r,l1:0.5-s1,t0.5-s1vws2,l1:10")


# # #### 3
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 A_Supplementary_trajectory/yingrenshi/0/pred_label_alpha.mp4\
#           -i2 A_Supplementary_trajectory/yingrenshi/0/instance.mp4\
#           -o A_Supplementary_trajectory_output/combine\
#           -c s1,t2.25-s1iws2,l1:1-s2,t6")



# # #### 1 2
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 A_Supplementary_trajectory_output/combine/yingrenshi1.mp4\
#           -i2 A_Supplementary_trajectory_output/combine/yingrenshi2.mp4\
#           -o A_Supplementary_trajectory_output/combine\
#           -c s1,t6-s2,t10")

# ffmpeg -f concat -i A_Supplementary_trajectory_output/yingrenshi.txt  -c copy A_Supplementary_trajectory_output/yingrenshi_output.mp4