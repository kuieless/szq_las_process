import cv2
import numpy as np

import configargparse


def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err



def _get_opts():
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--output_path', default='dji/process_las/output_1/debug',type=str, required=False)
    return parser.parse_known_args()[0]

def main(hparams):

    i = range(0, 300, 50)
    for j in i:
        image2 = cv2.imread(str(hparams.output_path)+'/{0:06d}_2_project.png'.format(j))
        image1 = cv2.imread(str(hparams.output_path)+'/{0:06d}_1_rgbs.png'.format(j))
        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0])) 
        
        mask = ((image2[:,:,0]==255) * (image2[:,:,1]==255) * (image2[:,:,2]==255))
        image2[mask]=[255,255,255]
        image1[mask]=[255,255,255]
        
        # image2 = image2 * (~mask[:,:,np.newaxis].repeat(3,2))
        # image1 = image1 * (~mask[:,:,np.newaxis].repeat(3,2))

        difference = mse(image1, image2)
        print("MSE:", difference)

        error_image = cv2.absdiff(image1, image2)
        color_map_image = cv2.applyColorMap(error_image, cv2.COLORMAP_JET)
        # color_map_image[mask]=[255,255,255]

        cv2.imwrite(str(hparams.output_path)+'/{0:06d}_3_a.png'.format(j), color_map_image)


        diff = np.power((np.clip(error_image / 255, 0, 0.2)/0.2), 6).astype(np.uint8)*255
        color_map_image1 = cv2.applyColorMap(diff, cv2.COLORMAP_JET)


        # color_map_image1[mask]=[255,255,255]

        cv2.imwrite(str(hparams.output_path)+'/{0:06d}_3_b.png'.format(j), color_map_image1)
            

if __name__ == '__main__':
    main(_get_opts())


