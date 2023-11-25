import os



# # #### contrastive 的对比 b2  
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/b2/1107_longhua_b2_density_depth_hash22_instance_origin_sam_0001_depth_crossview.mp4 \
#           -i2 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/b2/1107_longhua_b2_density_depth_hash22_instance_origin_sam_0001.mp4 \
#           -o /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/combine3\
#           -c s2vs1,t1-s1iws2,l0.5:0,s-s1vws2r,l0.5:2.8-s1iws2,l1:0-s1iws2r,l0:0")
#         #   -c s2vs1,t1-s1iws2r,l0.5:0,s-s1vws2r,l0.5:2.82-s1iws2,l1:0.5")
#         #   -c s2vs1,t1-s1iws2,l0.5:0,s-s1vws2r,l0.5:2.82-s1iws2,l1:0.5")
#         #   -c s1vs2,t8")




# # #### contrastive 的对比  campus    1124的版本
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/campus/1113_campus_density_depth_hash22_instance_origin_sam_0001_depth_crossview_all.mp4 \
#           -i2 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/campus/1107_campus_density_depth_hash22_instance_origin_sam_0001.mp4\
#           -o /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/combine3\
#           -c s2vs1,t1-s1iws2r,l0.5:0,s-s1vws2,l0.5:2.82-s1iws2r,l1:0-s2,t2")



# #### contrastive 的对比  campus    1125 最后一段不要
os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
          -i1 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/campus/1113_campus_density_depth_hash22_instance_origin_sam_0001_depth_crossview_all.mp4 \
          -i2 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/campus/1107_campus_density_depth_hash22_instance_origin_sam_0001.mp4\
          -o /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/combine3\
          -c s2vs1,t1-s1iws2r,l0.5:0,s-s1vws2,l0.5:1.2-s1iws2r,l1:0")




# # #### linear assignment 的对比 b1  
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/b1/1109_longhua_b1_density_depth_hash22_instance_origin_sam_0001_linear_assignment_crossview_all.mp4 \
#           -i2 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/b1/1109_longhua_b1_density_depth_hash22_instance_origin_sam_0001_linear_assignment.mp4 \
#           -o /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/combine3\
#           -c s2vs1,t1-s1iws2r,l0.5:0,s-s1vws2,l0.5:2.82-s1iws2r,l1:0-s2,t2")


# # #### linear assignment 的对比 yingrenshi
# os.system("python /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/compare.py \
#           -i1 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/yingrenshi/1106_yingrenshi_density_depth_hash22_instance_origin_sam_0001_linear_assignment_depth_crossview.mp4 \
#           -i2 /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/yingrenshi/1106_yingrenshi_density_depth_hash22_instance_origin_sam_0001_linear_assignment.mp4 \
#           -o /data/yuqi/code/GP-NeRF-semantic/A_Supplementary/render_instance/MP4/combine3\
#           -c s2vs1,t1-s1iws2r,l0.5:0,s-s1vws2,l0.5:2.82-s1iws2r,l1:0-s2,t2")
