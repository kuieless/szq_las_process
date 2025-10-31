import torch
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

def find_original_images(dataset_path, original_images_base_path):
    """
    根据 image_metadata 目录反向查找 rgbs 图像对应的原始图像。
    """
    
    output_path = Path(dataset_path)
    original_base_path = Path(original_images_base_path)
    
    rgbs_dir = output_path / 'rgbs'
    image_metadata_dir = output_path / 'image_metadata'
    
    if not rgbs_dir.exists():
        print(f"❌ 错误: 找不到 'rgbs' 目录: {rgbs_dir}")
        sys.exit(1)
    if not image_metadata_dir.exists():
        print(f"❌ 错误: 找不到 'image_metadata' 目录: {image_metadata_dir}")
        print("这是否是正确的 dataset_path？")
        sys.exit(1)

    print(f"正在扫描: {rgbs_dir}")
    print(f"查找元数据: {image_metadata_dir}")
    print(f"原始图像根目录: {original_base_path}\n")
    
    # 找到所有重命名的 .jpg 图像
    renamed_images = sorted(list(rgbs_dir.glob('*.jpg')))
    
    if not renamed_images:
        print(f"⚠️ 警告: 在 {rgbs_dir} 中没有找到 .jpg 文件。")
        return

    print(f"找到了 {len(renamed_images)} 张重命名的图像。正在生成映射...\n")
    
    # (可选) 将映射保存到
    mapping_file_path = output_path / "original_to_renamed_mapping.txt"
    
    with open(mapping_file_path, "w", encoding="utf-8") as f:
        f.write("Renamed_Image_Path  <---  Full_Original_Image_Path\n")
        f.write("=" * 80 + "\n")

        # 遍历每张重命名的图像
        for renamed_img_path in tqdm(renamed_images, desc="反向映射"):
            
            # 1. 获取基础名称 (例如 "000123")
            file_stem = renamed_img_path.stem
            
            # 2. 构建对应的 .pt 文件路径
            meta_path = image_metadata_dir / f'{file_stem}.pt'
            
            if not meta_path.exists():
                print(f"警告: 找不到 {renamed_img_path} 对应的元数据文件 {meta_path}！")
                continue
                
            try:
                # 3. 加载元数据
                metadata = torch.load(meta_path, map_location='cpu')
                
                # 4. 提取 'original_image_name'
                if 'original_image_name' in metadata:
                    relative_original_path = metadata['original_image_name']
                    
                    # 5. 构建完整的原始路径
                    full_original_path = original_base_path / relative_original_path
                    
                    # 写入文件
                    f.write(f"{renamed_img_path.name}  <---  {full_original_path}\n")
                    
                else:
                    print(f"警告: 元数据 {meta_path} 中缺少 'original_image_name' 键！")
                    
            except Exception as e:
                print(f"错误: 加载或处理 {meta_path} 时出错: {e}")

    print("\n" + "=" * 30)
    print("✅ 映射完成！")
    print(f"映射报告已保存至: {mapping_file_path}")
    print("\n报告的前几行内容示例:")
    
    # 打印前 5 行作为预览
    with open(mapping_file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 6: # (2行标题 + 5行数据)
                break
            print(line.strip())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="反向查找重命名图像的原始文件")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="包含 'rgbs' 和 'image_metadata' 目录的输出路径。")
    parser.add_argument('--original_images_path', type=str, required=True,
                        help="原始脚本运行时使用的 'original_images_path' 根目录。")
    
    args = parser.parse_args()
    
    find_original_images(args.dataset_path, args.original_images_path)
'''
    # 示例命令
python szq_a_reverse_index.py \
    --dataset_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/finished/sztu \
    --original_images_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSZTU/images/survey

    '''