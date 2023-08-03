
from plyfile import PlyData
import numpy as np
from tqdm import tqdm

import laspy
import pyproj
import torch
import datetime
from datetime import timedelta
from datetime import datetime

# Load the LAS file
in_file = laspy.file.File('/data/yuqi/Datasets/DJI/M1.las', mode='r')

header = in_file.header

x, y, z = in_file.x, in_file.y, in_file.z
r = (in_file.red / 65536.0 * 255).astype(np.uint8)
g = (in_file.green / 65536.0 * 255).astype(np.uint8)
b = (in_file.blue / 65536.0 * 255).astype(np.uint8)
rgb_colors = np.array(list(zip(r, g, b)))
coordinate = np.array(list(zip(x, y, z)))



GPSfromUTC = (datetime(1980,1,6) - datetime(1970,1,1)).total_seconds()
curDate = datetime.utcfromtimestamp(in_file.gps_time[-1] + GPSfromUTC) 
print(curDate)
a = torch.load('/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/val/image_metadata/000000.pt')


intensity=in_file.intensity


np.save('dji/project_script/ply/points', coordinate)
np.save('dji/project_script/ply/colors', rgb_colors)

# Close the LAS file when you're done
in_file.close()
print('done')