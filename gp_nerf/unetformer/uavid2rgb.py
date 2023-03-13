import numpy as np


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb

