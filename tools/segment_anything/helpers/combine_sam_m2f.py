import numpy as np
from pycocotools import mask
from PIL import Image
import torch
import matplotlib.pyplot as plt
import glob
import os
import time
import sys
from pathlib import Path
from tqdm import tqdm
import cv2
from tools.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# torch.cuda.set_device(6)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_id(sam, m2f):
    # print('mask', i)
    # sam = torch.from_numpy(sam).long().to(device)
    sam = sam.to(torch.long)
    labeled_sam = torch.where(sam==1, m2f, sam)
    
    unique, counts = torch.unique(labeled_sam, return_counts=True)
    # print('labeled', unique, counts)

    # label the whole mask with the most frequent label
    if len(unique) > 1:
        unique = unique[1:]
        counts = counts[1:]

        # add by zyq:
        # attach the labels only when the max_label_class > 10% * mask_area
        sam_mask_unique, sam_mask_counts = torch.unique(sam, return_counts=True)
        counts_total = sam_mask_counts[sam_mask_unique==1]
        couts_max_label = counts.max()
        if couts_max_label / counts_total > 0.1:
            id_max_label = unique[counts.argmax()]
        else:
            id_max_label = 0
        
        # id_max_label = unique[counts.argmax()]
        
        result = torch.where(sam==1, id_max_label, sam)
    else:
        result = torch.where(sam==1, unique[counts.argmax()], sam)

    return result
def get_mask(dict, semantics):
    if len(dict) == 0:
        return

    mask = torch.zeros_like((dict[0]['segmentation'])).to(torch.long)
    for i in range(len(dict)):
        id = dict[i]['id']
        mask = torch.where(mask==0, id, mask) # avoid overwriting
    mask = torch.where(mask==0, semantics, mask)

    # mask = np.zeros((dict[0]['segmentation'].shape[0], dict[0]['segmentation'].shape[1]))
    # for i in range(len(dict)):
    #     id = dict[i]['id']
    #     mask = np.where(mask==0, id, mask) # avoid overwriting
    # mask = np.where(mask==0, semantics, mask)
    return mask.cpu().numpy()
def custom2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]             # cluster       black
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 0, 0]           # building      red
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [192, 192, 192]       # road          grey  
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [192, 0, 192]         # car           light violet
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 128, 0]           # tree          green
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [128, 128, 0]         # vegetation    dark green
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 255, 0]         # human         yellow
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [135, 206, 250]       # sky           light blue
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 0, 128]           # water         blue
    mask_rgb[np.all(mask_convert == 9, axis=0)] = [252,230,201]         # ground        egg
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [128, 64, 128]       # mountain      dark violet

    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("zyq: wrong input argv number")
        sys.exit(1)

    root_path = sys.argv[1]
    SAM_path = os.path.join(root_path, 'sam_features')
    m2f_path = os.path.join(root_path, 'labels_m2f')
    img_path = os.path.join(root_path, 'rgbs')
    save_path_o = sys.argv[2]
    save_path = os.path.join(save_path_o, 'labels_merge')
    save_path_vis = os.path.join(save_path_o, 'labels_merge_vis')
    Path(save_path).mkdir(exist_ok=True,parents=True)
    Path(save_path_vis).mkdir(exist_ok=True)

    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32)


    imgs = []
    for ext in ('*.jpg'):
        imgs.extend(glob.glob(os.path.join(img_path, ext)))
    imgs.sort()
    imgs = imgs[1:]

    sams = []
    for ext in ('*.npy'):
        sams.extend(glob.glob(os.path.join(SAM_path, ext)))
    sams.sort()
    sams = sams[1:]

    m2fs = []
    for ext in ('*.png'):
        m2fs.extend(glob.glob(os.path.join(m2f_path, ext)))
    m2fs.sort()
    m2fs = m2fs[1:]

    # for i in range(len(sams)):
    for i in tqdm(range(len(sams))):
        # print(i)
        # start = time.time()
        # load_dict = np.load(sams[i],allow_pickle=True)
        image = cv2.imread(imgs[i])
        image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        feature = np.load(sams[i])

        m2f = Image.open(m2fs[i])
        semantics = torch.from_numpy(np.array(m2f)).long().to(device)

        load_dict = mask_generator.generate(image1, feature)

        # load_dict = torch.tensor(mask.decode(list(load_dict))).to(device)

        # for j in range(len(load_dict)):
        #     load_dict[j] = torch.tensor(mask.decode(load_dict[j])).to(device)

        dict = [{'segmentation': torch.tensor(load_dict[k]['segmentation']).to(device), 'id': get_id(torch.tensor(load_dict[k]['segmentation']).to(device), semantics)} for k in range(len(load_dict))]
        # dict = [{'segmentation': dict[i]['segmentation'], 'id': dict[i]['id'], 'color': custom2rgb(dict[i]['id'].numpy())} for i in range(len(dict))]

        # plt.figure(figsize=(12.16, 9.125), dpi=100)
        # plt.axis('off')
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # dict['id'] = dict['id'].cpu().numpy()
        _mask = get_mask(dict, semantics)
        Image.fromarray(_mask.astype(np.uint16)).save(os.path.join(save_path,sams[i].split('/')[-1][:6]+".png"))

        mask_rgb = custom2rgb(_mask)
        Image.fromarray(mask_rgb).save(os.path.join(save_path_vis,sams[i].split('/')[-1][:6]+".png"))
        # end = time.time()
        # print(end-start)
        