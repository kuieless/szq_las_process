import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['savefig.dpi'] = 600 
# 替换数值为字符
x_labels = ['0(M2F)', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
x_positions = np.arange(len(x_labels))  # 生成横坐标刻度位置
y_ours = [0.848, 0.921, 0.938, 0.932, 0.937, 0.928, 0.936, 0.936, 0.935, 0.935, 0.934]
y_hash = [0.709, 0.897, 0.884, 0.907, 0.897, 0.905, 0.904, 0.896, 0.903, 0.832, 0.879]

plt.figure(dpi=225, figsize=(5, 3))

fontsize_label = 18
fontsize = 18
fontsize_legend = 18
ticksize = 18

markersize = 7
linewidth = 3
ours_color = 'lightcoral'
hash_color = 'lightskyblue'

plt.plot(x_positions, y_ours, color=ours_color, marker='o', label='Yingrenshi', markersize=markersize, linewidth=linewidth)
plt.plot(x_positions, y_hash, color=hash_color, marker='o', label='Yuehai-Campus', markersize=markersize, linewidth=linewidth)

# plt.xlabel('Altitude Offset', , size=fontsize_label, weight='bold')
plt.ylabel('IoU of Building', size=fontsize_label, fontproperties='Times New Roman', weight='bold')

plt.rcParams.update({'font.size': fontsize_legend, 'font.weight': 'bold'})
plt.legend(loc='lower right', fontsize=fontsize_legend, prop={'family': 'Times New Roman', 'weight': 'bold'})

# 设置横坐标两侧留有空白
plt.xlim(xmin=-0.25, xmax=len(x_labels)-0.75)

# ymax, ymin = max(max(y_ours), max(y_hash)), min(min(y_ours), min(y_hash))
plt.ylim(0.67, 0.95)

# 设置横坐标刻度位置和标签
plt.xticks(x_positions[::3], x_labels[::3], fontsize=ticksize, fontproperties='Times New Roman', weight='bold')
plt.yticks(fontsize=ticksize, fontproperties='Times New Roman', weight='bold')

plt.savefig(os.path.join('./scripts/plot_far_view_ablation.pdf'), bbox_inches='tight')
plt.show()
plt.close()
