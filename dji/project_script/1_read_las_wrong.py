
from plyfile import PlyData
import numpy as np
from tqdm import tqdm

import laspy
import pyproj

# Load the LAS file
in_file = laspy.file.File('/data/yuqi/Datasets/DJI/M1.las', mode='r')

header = in_file.header

# from_proj = pyproj.Proj(in_file.header.project_id)
# to_proj = pyproj.Proj('epsg:32650')
# x,y,z = pyproj.transform(from_proj, to_proj, in_file.x, in_file.y, in_file.z)

# Access the points and their attributes
points = in_file.points
x = points['point']['X']  # X coordinates
y = points['point']['Y']  # Y coordinates
z = points['point']['Z']  # Z coordinates
intensity = points['point']['intensity']  # Intensity values
r = (points['point']['red'] / 65536.0 * 255).astype(np.uint8)
g = (points['point']['green'] / 65536.0 * 255).astype(np.uint8)
b = (points['point']['blue'] / 65536.0 * 255).astype(np.uint8)
rgb_colors = np.array(list(zip(r, g, b)))

coordinate = np.array(list(zip(x, y, z)))


# # 定义输入坐标系为 WGS84 (EPSG:4326)
# input_crs = pyproj.CRS("EPSG:4326")
# # 定义输出坐标系为 UTM (例如，UTM Zone 50N，EPSG:32650)
# output_crs = pyproj.CRS("EPSG:32650")
# # 创建坐标转换器
# transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)





np.save('dji/project_script/ply/points', coordinate)
np.save('dji/project_script/ply/colors', rgb_colors)

# Close the LAS file when you're done
in_file.close()
print('done')