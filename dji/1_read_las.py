
from plyfile import PlyData
import numpy as np
from tqdm import tqdm

import laspy
import pyproj

# Load the LAS file
in_file = laspy.file.File('/data/yuqi/Datasets/DJI/origin/DJI_20230726_xiayuan_data/1.las', mode='r')

header = in_file.header


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

np.save('dji/ply/points', coordinate)
np.save('dji/ply/colors', rgb_colors)

# Close the LAS file when you're done
in_file.close()
print('done')