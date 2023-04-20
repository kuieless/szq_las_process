import numpy as np


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]           # building
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]        # road
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]           # tree
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]         # vegetation
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]          # moving car
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]         # static car
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]           # human
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]             # cluster

    # mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 0, 128]             # sky
    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def custom2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]             # cluster       black

    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 0, 0]           # building      red
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [192, 192, 192]         # road        grey  
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [192, 0, 192]         # car           light violet
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 128, 0]           # tree          green
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [128, 128, 0]         # vegetation    dark green
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 255, 0]         # human         yellow
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [135, 206, 250]       # sky           light blue
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 0, 128]           # water         blue

    mask_rgb[np.all(mask_convert == 9, axis=0)] = [252,230,201]          # ground       egg
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [128, 64, 128]        # mountain     dark violet

    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb