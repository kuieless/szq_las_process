import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import time
from tools.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import glob
import os
import tqdm
from pathlib import Path


torch.cuda.set_device(7)



def show_anns(colors, anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)
    # polygons = []
    # color = []
    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
    mask = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
    for i in range(len(sorted_anns)):
        m = anns[i]['segmentation']
        mask = mask + m[:,:,np.newaxis] * np.random.random((1,3))
        
        
        # for j in range(3):
        #     img[:,:,j] = colors[i][j]
        # mask = mask + img * m[:,:,np.newaxis] * 0.5
        # ax.imshow(np.dstack((img, m*0.35)))
    return mask


if __name__ == "__main__":
    points_per_side=16

    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)

    imgs = []
    for ext in ('*.png'):
        imgs.extend(glob.glob(os.path.join("/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/rgbs/", ext)))
    imgs.sort()
    imgs = imgs[1:]

    features = []
    for ext in ('*.npy'):
        features.extend(glob.glob(os.path.join("/data/yuqi/Datasets/MegaNeRF/UrbanScene3D/residence/residence-labels/train/sam_features/", ext)))
    features.sort()
    features = features[1:]


    colors = []
    for i in range(len(imgs)):
        colors.append(np.random.random((1,3)).tolist()[0])
    colors = np.clip(colors, 0, 0.95)

    Path(f'./tools/segment_anything/SAM_mask_autogenerate_{points_per_side}').mkdir(exist_ok=True)
    # for i in tqdm.tqdm(range(len(imgs))):
    for i in tqdm.tqdm(range(500)):

        # start = time.time()
        image = cv2.imread(imgs[i])
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feature = np.load(features[i])
        # return torch.from_numpy(feature)
            
        masks = mask_generator.generate(image1, feature)
        # print(len(masks))
        # print(masks[0].keys())
        
        mask = show_anns(colors, masks)
        
        # image_cat = np.concatenate([image, mask*255], axis=1)
        image_cat=mask*255
        # image_cat=np.ones_like(mask)*255

        cv2.imwrite(f"./tools/segment_anything/SAM_mask_autogenerate_{points_per_side}/"+imgs[i].split('/')[-1][:6]+".png", image_cat)
                    
        
        # plt.figure(figsize=(13.68, 9.125), dpi=100)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)

        # plt.imshow(image)
        # mask = show_anns(colors, masks)
        # plt.imshow(mask)
        # plt.savefig(f"./tools/segment_anything/SAM_mask_autogenerate_{points_per_side}/"+imgs[i].split('/')[-1][:6]+".png", dpi=100)
        # plt.close()
        # end = time.time()
        # print(end-start)
    print('done')