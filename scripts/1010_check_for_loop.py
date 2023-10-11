import time
import numpy as np
import pickle
img_num = 2
point_num = 160000
label_num = 5
H, W = 1024, 1024

#img_num = 1
#point_num = 16
#label_num = 5
#H, W = 4, 4

point_label_list = {}
for i in range(point_num):
    point_label_list[i] = []

point_label_list2 = {}
for i in range(point_num):
    point_label_list2[i] = []

for i in range(img_num):
    image = (np.random.rand(H, W) * point_num).astype(int) # an array, each pixel stores an index
    gt_label = (np.random.rand(H, W) * label_num).astype(int) # stores the label of the index

    t = time.time()
    count = 0
    for h in range(H):
        for w in range(W):
            point_label_list[int(image[h, w])].append(int(gt_label[h, w]))
            count += 1
    print(np.round_(time.time() - t, 3), 'sec elapsed')
    print('Count %d' % count)
    print('Point List Count %d' % len(point_label_list))

    if point_num < 20:
        print(point_label_list)


    # v2 
    image_flatten = image.flatten()
    gt_label_flatten = gt_label.flatten()
    t = time.time()
    count = 0
    for idx, label in zip(image_flatten, gt_label_flatten):
        #print(idx, label)
        point_label_list2[idx].append(label)
        #point_label_list2[int(idx)].append(int(label))
        count += 1

    if point_num < 20:
        print(point_label_list2)
    
    #for k, v in point_label_list_unique.items()
    #point_label_list2.update({image_flatten[i]: gt_label_flatten[i]})
    print(np.round_(time.time() - t, 3), 'sec elapsed')
    print('count', count)
    print('Point List Count %d' % len(point_label_list2))

    #print(point_label_list)


with open(f'/tmp/v1.pkl', 'wb') as file:
    pickle.dump(point_label_list, file)

with open(f'/tmp/v2.pkl', 'wb') as file:
    pickle.dump(point_label_list2, file)
