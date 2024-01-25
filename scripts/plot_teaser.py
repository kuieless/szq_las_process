import numpy as np

import matplotlib.pyplot  as plt
import os


# x = [9, 30,50,70,90]
# # y_ours = [22.45871816, 23.36258, 23.63999, 23.80732, 23.97601173]
# y_ours = [22.45871816, 23.36258, 23.63999, 23.90732, 24.08]
# y_hash = [22.38030133, 23.24145, 23.45302, 23.67686, 23.79520859]
# y_mlp = [19.94925033, 20.8887, 21.2016, 21.40524, 21.62904649]
# y_tensorf = [20.91901399, 21.84274, 22.09698, 22.31181, 22.44897087]
# y_dense = [21.268251, 21.87022, 22.15175, 22.31213, 22.42535778]

x = [M2F, Ours_0.1,Ours_0.2,Ours_0.3,Ours_0.4,Ours_0.5]
y_ours = [0.848,0.921,0.938,0.932,0.937,0.928]
y_hash = [0.709,0.897,0.884,0.907,0.897,0.905]

plt.figure(dpi=225,figsize=(5,4.166667))
fontsize_label= 17
fontsize=16
fontsize_legend=13
markersize=4
linewidth=3
ours_color = 'lightcoral'  #'orange'lightskyblue
hash_color = 'lightskyblue'   #'mediumblue'lightcoral
dense_color = '#fdaa48'
tensorf_color = 'lightgreen'
mlp_color = '#a552e6' #lightpink

#plt.title("Ablation study on {}".format(dataset), fontproperties='Times New Roman', size=fontsize)
# plt.text(9.5, 23.75, 'Mega-NeRF (one day)', color='#555555', fontsize=fontsize_legend)
plt.plot(x, y_ours,color=ours_color,marker='o',label='Yingrenshi', markersize=markersize, linewidth=linewidth)
plt.plot(x, y_hash, color=hash_color, marker='o',label='Yuehai-Campus', markersize=markersize, linewidth=linewidth)
# plt.plot(x, y_dense, color=dense_color, marker='o',label='Dense-grid', markersize=markersize, linewidth=linewidth)
# plt.plot(x, y_tensorf, color=tensorf_color, marker='o',label='TensoRF', markersize=markersize, linewidth=linewidth)
# plt.plot(x, y_mlp, color=mlp_color, marker='o',label='MLP', markersize=markersize, linewidth=linewidth)



xmax, xmin = 93, 7
ymax, ymin = 0.91, 0.70
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
# plt.hlines(24.03, xmin=xmin, xmax=xmax, linestyles='dashed', colors='slategray')

plt.xlabel('Thresholds', fontproperties='Times New Roman', size=fontsize_label)
plt.ylabel('PSNR of Building', fontproperties='Times New Roman', size=fontsize_label)

plt.xticks(size=fontsize, fontproperties='Times New Roman')
plt.xticks(x)
plt.yticks(size=fontsize, fontproperties='Times New Roman')
plt.yticks([20,21,22,23,24])

plt.rcParams.update({'font.size':fontsize_legend})
plt.legend(loc='lower right', prop='Times New Roman')
ax=plt.gca()
# plt.grid()


plt.savefig(os.path.join('./scripts/plot_teaser.pdf'), bbox_inches='tight')
plt.show()
plt.close()
