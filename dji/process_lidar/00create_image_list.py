import os
import json
import argparse
from tqdm import tqdm


def create_image_list_json(image_folder_path, output_filename='image_list.json'):
    """
    遍历指定的图片文件夹及其所有子文件夹，生成一个包含图片id和相对路径的JSON文件。

    JSON 格式:
    [
      {
        "id": "文件名 (无后缀)",
        "origin_path": "子文件夹/文件名.jpg"
      },
      ...
    ]
    """
    print(f"开始扫描文件夹: {image_folder_path}")

    image_list = []
    # 支持常见的图片格式，不区分大小写
    supported_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

    # 使用 os.walk 遍历所有子文件夹
    for root, _, files in tqdm(os.walk(image_folder_path), desc="正在扫描文件"):
        for filename in files:
            # 检查文件扩展名是否为支持的图片格式
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                # 获取文件名（不含扩展名）作为 id
                file_id = os.path.splitext(filename)[0]

                # 获取从主文件夹开始的相对路径
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, image_folder_path)

                # 确保路径分隔符为 '/'
                relative_path = relative_path.replace(os.sep, '/')

                image_list.append({
                    "id": file_id,
                    "origin_path": relative_path
                })

    # 按原始路径对列表进行排序，确保顺序一致性
    image_list.sort(key=lambda x: x['origin_path'])

    # 定义输出文件的完整路径
    output_path = os.path.join(image_folder_path, output_filename)

    # 将列表写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(image_list, f, indent=2, ensure_ascii=False)

    print(f"\n成功生成 {len(image_list)} 条记录。")
    print(f"JSON 文件已保存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为图片文件夹生成一个 image_list.json 文件。")
    parser.add_argument("image_folder", type=str, help="包含原始照片的根文件夹路径。")

    args = parser.parse_args()

    if not os.path.isdir(args.image_folder):
        print(f"错误: 提供的路径 '{args.image_folder}' 不是一个有效的文件夹。")
    else:
        create_image_list_json(args.image_folder)

#python 00create_image_list.py H:\BaiduNetdiskDownload\LiDAR_Raw\SZIIT\PSZIIT018\images\survey