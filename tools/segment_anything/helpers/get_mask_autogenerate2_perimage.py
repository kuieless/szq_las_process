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


torch.cuda.set_device(6)

# #sam在demo中的版本
def show_anns1(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))

    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_anns2(colors, anns):
    
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))  # 255 denotes white
    
    mask = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 1))
    mask_visual = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
    visual = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
    
    temp_mask_list=[]
    save =0
    for i in range(len(sorted_anns)):
    # for i in range(12):

        m = sorted_anns[i]['segmentation']
        visual[m] = np.random.random((1,3))
        
        overlap = (m * mask[:,:,0]).astype(bool)
        if overlap.sum() / m.sum() < 0.2:
            if save==90:
                mask_visual = np.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 3))
                
            if save>=102:
                break
            save += 1
            mask[m] = i + 1
            temp_mask = torch.zeros((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 1))
            temp_mask[m] = i + 1
            temp_mask_list.append(temp_mask)
            mask_visual[m] = np.random.random((1,3))
            # if save>=90:
                # cv2.imwrite(f"./tools/segment_anything/SAM_mask_autogenerate_{file_name}/"+"mask_%04d.png" % i, mask_visual*255)

        # cv2.imwrite(f"./tools/segment_anything/SAM_mask_autogenerate_{file_name}/"+"visual_%04d.png" % i, visual*255)


    cv2.imwrite(f"./tools/segment_anything/SAM_mask_autogenerate_{file_name}/visual_final.png", visual*255)
    cv2.imwrite(f"./tools/segment_anything/SAM_mask_autogenerate_{file_name}/mask_final.png", mask_visual*255)
    # temp_mask_list=torch.cat(temp_mask_list, dim=-1)
    # np.save(f"./tools/segment_anything/000027_sam.npy", temp_mask_list.numpy())
    # labels = np.load('/data/yuqi/code/GP-NeRF-semantic/tools/segment_anything/000027_sam.npy')


    
    # for i in range(len(sorted_anns)):
    #     m = anns[i]['segmentation']
    #     mask = mask + m[:,:,np.newaxis] * np.random.random((1,3))
    return visual


if __name__ == "__main__":
    file_name='DJi'

    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32)

    imgs = []
    for ext in ('*.png'):
        imgs.extend(glob.glob(os.path.join("/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/val/rgbs/", ext)))
    imgs.sort()
    imgs = imgs[1:]


    features = []
    for ext in ('*.npy'):
        features.extend(glob.glob(os.path.join("/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan/val/sam_features/", ext)))
    features.sort()
    features = features[1:]



    colors = []
    for i in range(len(imgs)):
        colors.append(np.random.random((1,3)).tolist()[0])
    colors = np.clip(colors, 0, 0.95)

    Path(f'./tools/segment_anything/SAM_mask_autogenerate_{file_name}').mkdir(exist_ok=True)
    for i in tqdm.tqdm(range(len(imgs))):
    # for i in tqdm.tqdm(range(27, 500)):
    # for i in tqdm.tqdm(range(365, 490)):


        # start = time.time()
        image = cv2.imread(imgs[i])
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feature = np.load(features[i])
        # return torch.from_numpy(feature)
            
        masks = mask_generator.generate(image1, feature)
        # print(len(masks))
        # print(masks[0].keys())
        
        # mask = show_anns1(masks)

        mask = show_anns2(colors, masks)
        
        image_cat = np.concatenate([image, mask*255], axis=1)
        # image_cat=mask*255
        # image_cat=np.ones_like(mask)*255



        cv2.imwrite(f"./tools/segment_anything/SAM_mask_autogenerate_{file_name}/"+imgs[i].split('/')[-1][:6]+".png", image_cat)
                    
        
        # plt.figure(figsize=(13.68, 9.125), dpi=100)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)

        # plt.imshow(image)
        # mask = show_anns(colors, masks)
        # plt.imshow(mask)
        # plt.savefig(f"./tools/segment_anything/SAM_mask_autogenerate_{file_name}/"+imgs[i].split('/')[-1][:6]+".png", dpi=100)
        # plt.close()
        # end = time.time()
        # print(end-start)
    print('done')