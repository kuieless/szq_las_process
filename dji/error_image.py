import cv2
import numpy as np

def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def create_error_image(image1, image2):
    diff_image = cv2.absdiff(image1, image2)
    return diff_image

def convert_to_color_map(error_image):
    # Normalize the error image to the range [0, 255]
    normalized_error_image = cv2.normalize(error_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Apply color map
    color_map = cv2.applyColorMap(error_image, cv2.COLORMAP_JET)
    return color_map



image1 = cv2.imread("test/val/rgbs/000000_1_AT_ori.jpg")
image2 = cv2.imread("test/val/rgbs/000000_3_ori.jpg")

difference = mse(image1, image2)
print("MSE:", difference)

error_image = create_error_image(image1, image2)
color_map_image = convert_to_color_map(error_image)


cv2.imwrite(f"test/val/rgbs/error_image_1_3_{difference}.jpg", color_map_image)



