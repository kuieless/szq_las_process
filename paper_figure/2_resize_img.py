import os
from PIL import Image

def resize_images(input_folder, output_folder, min_short_side=300):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图片
            with Image.open(input_path) as img:
                # 计算resize的比例
                width, height = img.size
                aspect_ratio = min_short_side / min(width, height)
                new_width = int(width * aspect_ratio)
                new_height = int(height * aspect_ratio)

                # 进行resize
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)

                # 保存resize后的图片
                resized_img.save(output_path)

if __name__ == "__main__":
    # 输入文件夹路径
    input_folder_path = "/data/yuqi/code/GP-NeRF-semantic/paper_figure/3_semantic_fusion"

    # 输出文件夹路径
    output_folder_path = "/data/yuqi/code/GP-NeRF-semantic/paper_figure/3_semantic_fusion/resize"

    # 最短边设为300
    min_short_side = 300

    # 调用函数进行resize
    resize_images(input_folder_path, output_folder_path, min_short_side)
