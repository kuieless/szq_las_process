import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['savefig.dpi'] = 600 
# 替换数值为字符
x_labels = ['0(w/o filter)', '5', '10', '15']
x_positions = np.arange(len(x_labels))  # 生成横坐标刻度位置
y_ours = [34.6, 35.9, 38.7, 37.9]

plt.figure(dpi=225, figsize=(5, 3))

fontsize_label = 18
fontsize = 18
fontsize_legend = 18
ticksize = 18
markersize = 7
linewidth = 3

ours_color = '#fdaa48'

plt.plot(x_positions, y_ours, color=ours_color, marker='o', label='Yingrenshi', markersize=markersize, linewidth=linewidth)

# plt.xlabel('Filter threshold', , size=fontsize_label, weight='bold')
plt.ylabel('PQ$^\mathrm{scene}$ of Building', fontproperties='Times New Roman', size=fontsize_label, weight='bold')

plt.rcParams.update({'font.size': fontsize_legend, 'font.weight': 'bold'})
plt.legend(loc='lower right', fontsize=fontsize_legend, prop={'family': 'Times New Roman', 'weight': 'bold'})

# 设置横坐标两侧留有空白
plt.xlim(xmin=-0.1, xmax=len(x_labels)-0.9)

plt.ylim(33, 39)

# 设置横坐标刻度位置和标签
plt.xticks(x_positions, x_labels, fontsize=ticksize, fontproperties='Times New Roman', weight='bold')
plt.yticks(fontsize=ticksize, fontproperties='Times New Roman', weight='bold')

plt.savefig(os.path.join('./scripts/plot_filter_ablation.pdf'), bbox_inches='tight')


plt.show()
plt.close()
