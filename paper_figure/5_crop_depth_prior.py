from PIL import Image
import os

# 原始图片文件夹路径
input_folder = "/data/yuqi/code/GP-NeRF-semantic/paper_figure/5_dep_prior1"

# 新文件夹路径，用于保存裁剪后的图片
output_folder = "/data/yuqi/code/GP-NeRF-semantic/paper_figure/5_dep_prior1/folder"

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 裁剪区域的左上角和右下角坐标
left_top = (500, 170)
right_bottom = (1500, 820)

# 遍历原始文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # 读取原始图片
        image_path = os.path.join(input_folder, filename)
        original_image = Image.open(image_path)

        # 裁剪图片
        cropped_image = original_image.crop((left_top[0], left_top[1], right_bottom[0], right_bottom[1]))
        cropped_image = cropped_image.convert("RGB")

        # 构造输出路径
        output_path = os.path.join(output_folder, f"cropped_{filename}")
        # 保存裁剪后的图片
        cropped_image.save(output_path[:-4]+'.jpg')

print("裁剪完成，保存在", output_folder)
