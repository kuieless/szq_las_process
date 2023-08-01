
from plyfile import PlyData
import numpy as np
from tqdm import tqdm

# 读取PLY文件
print('loading')
# plydata = PlyData.read('/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/M1_cloud_merged.ply')
plydata = PlyData.read('/data/yuqi/Datasets/DJI/Mmerge_from_las.ply')

print('load ply done')
vertex_data = plydata['vertex']
points = []
colors = []



for vertex in tqdm(vertex_data):
    x, y, z = vertex[0], vertex[1], vertex[2]
    r, g, b = vertex[3], vertex[4], vertex[5]

    points.append([x, y, z])
    colors.append([r, g, b])
points = np.array(points)
colors = np.array(colors)

np.save('dji/project_script/ply/points', points)
np.save('dji/project_script/ply/colors', colors)

a = 1 