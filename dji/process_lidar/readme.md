### 学长的代码库
### 首先，用之前处理 gpnerf 的代码先对图像进行重命名, 划分 train / val , 得到 nerf 坐标系下的pose
0_process_dji_v8_color.py

### 然后对 las 文件进行处理，根据采集时间划分成单帧的数据，存储在一个 pickle 里
1_process_each_line_las

### 对当前拍摄图像，提取前后两帧的点云，投影得到稀疏 depth
2_convert_lidar_2_depth_color

### 使用mesh投影得到dense depth
3_render_mesh_depth

### SZQ的代码
# szq_00run_batch_survey.sh
先得到json

# szq_0run_batch_process.sh
运行，和学长的第0步差不多，我舍去了划分以及nerf的逻辑

# szq_1run_batch_lidar.sh
对las进行处理，这里需要再py文件里设置一个时间窗口（我一般取80，前后40）

# szq_2run_batch_depth.sh
投影，得到npy（这个npy还是nerf里的归一化的）

# szq_302-visdepth.sh
把前面得到的npy进行尺度缩放，回归到正常比例，然后得到metric depth以及对应的可视化图

#
进行mesh的投影得到mesh深度图

#
把点云的深度图和mesh的深度图进行对比，设置一个阈值（1米或者3m），从而过滤掉遮挡的点。
