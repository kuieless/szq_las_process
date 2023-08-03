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
    # normalized_error_image = cv2.normalize(error_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Apply color map
    color_map = cv2.applyColorMap(error_image, cv2.COLORMAP_JET)
    return color_map


i = range(0, 300, 50)
for j in i:
    image2 = cv2.imread('dji/project_script/ply/output/'+'{0:06d}_2_project.png'.format(j))
    image1 = cv2.imread('dji/project_script/ply/output/'+'{0:06d}_1_rgbs.png'.format(j))
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0])) 
    
    mask = ((image2[:,:,0]==255) * (image2[:,:,1]==255) * (image2[:,:,2]==255))
    image2[mask]=[255,255,255]
    image1[mask]=[255,255,255]
    


    
    # image2 = image2 * (~mask[:,:,np.newaxis].repeat(3,2))
    # image1 = image1 * (~mask[:,:,np.newaxis].repeat(3,2))

    difference = mse(image1, image2)
    print("MSE:", difference)

    error_image = create_error_image(image1, image2)
    color_map_image = convert_to_color_map(error_image)
    color_map_image[mask]=[255,255,255]

    cv2.imwrite("dji/project_script/ply/output/"+'{0:06d}_3_a.jpg'.format(j), color_map_image)



