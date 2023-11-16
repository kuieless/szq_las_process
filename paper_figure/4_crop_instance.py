import os
from PIL import Image
import configargparse

def crop_and_save_images(input_folder, output_folder, crop_coordinates, custom_name, output_quality=85):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图像文件
    supported_formats = ['.png', '.jpg', '.jpeg']
    for filename in os.listdir(input_folder):
        _, file_extension = os.path.splitext(filename.lower())

        # 检查文件扩展名是否受支持
        if file_extension in supported_formats:
            input_path = os.path.join(input_folder, filename)

            # 打开图像
            with Image.open(input_path) as img:
                # 根据提供的坐标进行裁剪
                left, upper, right, lower = crop_coordinates
                cropped_img = img.crop((left, upper, right, lower))

                # 构造新的文件名
                new_filename = f"{custom_name}_{filename}"

                # 保存裁剪后的图像并设置输出质量
                output_path = os.path.join(output_folder, new_filename)
                if file_extension == '.jpg' or file_extension == '.jpeg':
                    cropped_img.save(output_path, 'JPEG', quality=output_quality)
                else:
                    cropped_img.save(output_path)

if __name__ == "__main__":
    # 创建 Argument Parser
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    # 添加输入参数
    parser.add_argument('--input_path', required=True, help='输入文件夹路径')
    parser.add_argument('--output_path', required=True, help='输出文件夹路径')
    parser.add_argument('--custom_name', required=True, help='自定义名称前缀')



    # 解析输入参数
    args = parser.parse_args()
    if 'b1' in args.output_path or 'b2' in args.output_path:
        H, W = 1024, 1536
    else:
        H, W = 912, 1368


    crop_coordinates = (W*2+6, H+2, W*3+6, H*2+2)
    output_path = os.path.join(args.output_path, args.custom_name)
    # 调用函数进行裁剪并保存
    crop_and_save_images(args.input_path, output_path, crop_coordinates, args.custom_name, output_quality=85)
