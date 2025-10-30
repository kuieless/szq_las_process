import os
import json
import argparse
from tqdm import tqdm
import sys

def create_image_list_json(image_folder_path, output_filepath):
    """
    遍历指定的图片文件夹及其所有子文件夹，生成一个包含图片id和相对路径的JSON文件。
    """
    print(f"开始扫描文件夹: {image_folder_path}")

    # 检查 image_folder_path 是否存在
    if not os.path.isdir(image_folder_path):
        # 将警告信息输出到标准错误流，这样Bash脚本可以更容易区分
        print(f"警告: 目标文件夹不存在或不是一个文件夹: {image_folder_path}", file=sys.stderr)
        return 0 # 返回0表示处理了0个文件

    image_list = []
    supported_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

    # os.path.basename 用于在tqdm描述中显示更简洁的名称
    scan_desc = f"扫描 {os.path.basename(image_folder_path)}"
    
    for root, _, files in tqdm(os.walk(image_folder_path), desc=scan_desc, leave=False):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in supported_extensions):
                file_id = os.path.splitext(filename)[0]
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, image_folder_path)
                relative_path = relative_path.replace(os.sep, '/')

                image_list.append({
                    "id": file_id,
                    "origin_path": relative_path
                })

    image_list.sort(key=lambda x: x['origin_path'])

    # 检查是否有找到图片
    if not image_list:
        print(f"警告: 在 {image_folder_path} 中未找到任何图片。", file=sys.stderr)
        return 0

    # --- 关键修改 ---
    # 确保输出文件的目录存在
    output_dir = os.path.dirname(output_filepath)
    # 如果 output_dir 为空字符串 (表示当前目录)，则不创建
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")
        except OSError as e:
            print(f"错误: 无法创建输出目录 {output_dir}. 错误: {e}", file=sys.stderr)
            return 0

    # 将列表写入指定的JSON文件
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(image_list, f, indent=2, ensure_ascii=False)
        
        print(f"成功生成 {len(image_list)} 条记录。")
        print(f"JSON 文件已保存至: {output_filepath}")
        return len(image_list)
    except IOError as e:
        print(f"错误: 无法写入JSON文件 {output_filepath}. 错误: {e}", file=sys.stderr)
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为图片文件夹生成一个 image_list.json 文件。")
    parser.add_argument("image_folder", type=str, help="包含原始照片的根文件夹路径。")
    # 添加 --output_file 参数，并设为必需
    parser.add_argument(
        "-o", "--output_file", 
        type=str, 
        required=True, 
        help="输出JSON文件的完整路径 (例如: /path/to/output/PSZIIT018.json)。"
    )

    args = parser.parse_args()
    
    # 调用主函数
    create_image_list_json(args.image_folder, args.output_file)