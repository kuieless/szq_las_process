
import numpy as np

# 从txt文件中加载数据
data = np.loadtxt('/data/yuqi/Datasets/DJI/origin/Yingrenshi_whole_scene.txt')

# 将第二列的值改为相反数
data[:, 1] = -data[:, 1]

# 创建一个格式化字符串，保留整数类型
# fmt_str = ' '.join(['%s' if col.dtype == np.dtype('O') else '%d' for col in data.dtype])

# 保存修改后的数据到新的txt文件
# np.savetxt('/data/yuqi/Datasets/DJI/origin/Yingrenshi_whole_scene2.txt', data, fmt=fmt_str, delimiter=' ')



# 保存修改后的数据到新的txt文件
np.savetxt('/data/yuqi/Datasets/DJI/origin/Yingrenshi_whole_scene2.txt', data, fmt='%s', delimiter=' ')

print("数据处理完成并保存到output_data.txt文件中。")
