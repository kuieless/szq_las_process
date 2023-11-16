import os
from PIL import Image
import configargparse

def find_and_save_images(input_folder, output_folder, target_digits, target_size=(300, 300), output_quality=85):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历母文件夹中的所有子文件夹
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)

        # 检查是否为文件夹
        if os.path.isdir(subfolder_path):
            # 查找指定六位数字的图片
            for filename in os.listdir(subfolder_path):
                _, file_extension = os.path.splitext(filename.lower())

                # 检查是否为目标图片
                if target_digits in filename and file_extension in ['.png', '.jpg', '.jpeg']:
                    input_path = os.path.join(subfolder_path, filename)

                    # 打开图像
                    with Image.open(input_path) as img:
                        # 调整图像大小
                        resized_img = img.resize(target_size)

                        # 构造新的文件名
                        new_filename = filename

                        # 保存调整大小后的图像并设置输出质量
                        output_path = os.path.join(output_folder, new_filename)
                        if file_extension == '.jpg' or file_extension == '.jpeg':
                            resized_img.save(output_path, 'JPEG', quality=output_quality)
                        else:
                            resized_img.save(output_path)


if __name__ == "__main__":
    # 创建 Argument Parser
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)

    # 添加输入参数
    parser.add_argument('--input_folder_path', required=True, help='母文件夹路径')
    parser.add_argument('--output_folder_path', default='/data/yuqi/code/GP-NeRF-semantic/paper_figure/4_instance/overleaf', help='输出文件夹路径')
    parser.add_argument('--target_digits', required=True, help='目标六位数字字符串')
    parser.add_argument('--target_size', nargs='+', type=int, default=[450, 300], help='目标大小，默认为[300, 300]')
    parser.add_argument('--output_quality', type=int, default=85, help='输出质量，默认为85')

    # 解析输入参数
    args = parser.parse_args()

    # 调用函数查找并保存图片
    find_and_save_images(args.input_folder_path, args.output_folder_path, args.target_digits, target_size=args.target_size, output_quality=args.output_quality)
