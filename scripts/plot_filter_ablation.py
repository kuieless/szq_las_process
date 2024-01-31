import numpy as np
import matplotlib.pyplot as plt
import os

# 替换数值为字符
x_labels = ['0(w/o filter)', '5', '10', '15']
x_positions = np.arange(len(x_labels))  # 生成横坐标刻度位置
y_ours = [34.6, 35.9, 38.7, 37.9]

plt.figure(dpi=225, figsize=(5, 3))
# plt.figure(dpi=225, figsize=(5, 4.166667))
# fontsize_label = 13
# fontsize = 17
# fontsize_legend = 13
fontsize_label = 18
fontsize = 18
fontsize_legend = 18
ticksize=18
markersize = 7
linewidth = 3

# ours_color = 'lightcoral'  #'orange'lightskyblue
# hash_color = 'lightskyblue'   #'mediumblue'lightcoral
ours_color = '#fdaa48'
# tensorf_color = 'lightgreen'
# mlp_color = '#a552e6' #lightpink


plt.plot(x_positions, y_ours, color=ours_color, marker='o', label='Yingrenshi', markersize=markersize, linewidth=linewidth)



# plt.xlabel('Filter threshold', , size=fontsize_label)
plt.ylabel('PQ$^\mathrm{scene}$ of building', fontproperties='Times New Roman', size=fontsize_label)

plt.rcParams.update({'font.size': fontsize_legend})
plt.legend(loc='lower right', prop='Times New Roman')

# 设置横坐标两侧留有空白
plt.xlim(xmin=-0.1, xmax=len(x_labels)-0.9)

plt.ylim(33, 39)


# 设置横坐标刻度位置和标签
# plt.xticks(x_positions[::3], x_labels[::3], fontsize=ticksize, )
plt.xticks(x_positions, x_labels, fontsize=ticksize, fontproperties='Times New Roman')
plt.yticks(fontsize=ticksize, fontproperties='Times New Roman')

plt.savefig(os.path.join('./scripts/plot_filter_ablation.pdf'), bbox_inches='tight')
plt.show()
plt.close()
