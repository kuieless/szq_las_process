
### 首先，用之前处理 gpnerf 的代码先对图像进行重命名, 划分 train / val , 得到 nerf 坐标系下的pose
0_process_dji_v8_color.py

### 然后对 las 文件进行处理，根据采集时间划分成单帧的数据，存储在一个 pickle 里
1_process_each_line_las

### 对当前拍摄图像，提取前后两帧的点云，投影得到稀疏 depth
2_convert_lidar_2_depth_color

### 使用mesh投影得到dense depth
3_render_mesh_depth



