import numpy as np
import matplotlib.pyplot as plt
import os

# 替换数值为字符
x_labels = ['M2F', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
x_positions = np.arange(len(x_labels))  # 生成横坐标刻度位置
y_ours = [0.848, 0.921, 0.938, 0.932, 0.937, 0.928, 0.936, 0.936, 0.935, 0.935, 0.934]
y_hash = [0.709, 0.897, 0.884, 0.907, 0.897, 0.905, 0.904, 0.896, 0.903, 0.832, 0.879]

plt.figure(dpi=225, figsize=(6, 2))
# plt.figure(dpi=225, figsize=(5, 4.166667))
fontsize_label = 13
fontsize = 17
fontsize_legend = 13
markersize = 4
linewidth = 3
ours_color = 'lightcoral'
hash_color = 'lightskyblue'

plt.plot(x_positions, y_ours, color=ours_color, marker='o', label='Yingrenshi', markersize=markersize, linewidth=linewidth)
plt.plot(x_positions, y_hash, color=hash_color, marker='o', label='Yuehai-Campus', markersize=markersize, linewidth=linewidth)


# 设置横坐标刻度位置和标签
plt.xticks(x_positions, x_labels, size=fontsize, fontproperties='Times New Roman')
plt.yticks(size=fontsize, fontproperties='Times New Roman')

plt.xlabel('Altitude Offset', fontproperties='Times New Roman', size=fontsize_label)
plt.ylabel('IoU of Building', fontproperties='Times New Roman', size=fontsize_label)

plt.rcParams.update({'font.size': fontsize_legend})
plt.legend(loc='lower right', prop='Times New Roman')

# 设置横坐标两侧留有空白
plt.xlim(xmin=-0.1, xmax=len(x_labels)-0.9)

# ymax, ymin = max(max(y_ours), max(y_hash)), min(min(y_ours), min(y_hash))
plt.ylim(0.7, 0.95)

plt.savefig(os.path.join('./scripts/plot_far_view_ablation.pdf'), bbox_inches='tight')
plt.show()
plt.close()
