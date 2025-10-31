import csv
import os
import argparse
from tqdm import tqdm
import sys

def create_renamed_split_file(mapping_file_path, split_csv_path, output_file_path):
    """
    根据 'mapping.txt' 和 'split.csv' 生成重命名文件的 train/test 划分表。
    """
    
    print(f"步骤 1: 正在从 '{mapping_file_path}' 加载映射关系...")
    
    # 1. 加载映射文件 (mapping.txt)
    #    我们创建一个字典: { 原始完整路径 -> {重命名信息} }
    original_to_renamed_map = {}
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            mapping_lines = f.readlines()
            
        # 从索引 2 开始，跳过前两行标题
        for line in tqdm(mapping_lines[2:], desc="加载映射"):
            parts = line.strip().split("  <---  ")
            if len(parts) != 2:
                continue  # 跳过空行或格式不正确的行
            
            renamed_name = parts[0].strip()
            original_path = parts[1].strip()
            
            if not renamed_name or not original_path:
                continue
                
            # 提取重命名后的索引 (例如 "000000")
            renamed_index = os.path.splitext(renamed_name)[0]
            # 提取原始文件名 (例如 "008e57...JPG")
            original_filename = os.path.basename(original_path)
            
            # 存储所有需要的信息
            original_to_renamed_map[original_path] = {
                "renamed_name": renamed_name,
                "renamed_index": renamed_index,
                "original_filename": original_filename
            }
            
    except FileNotFoundError:
        print(f"❌ 错误: 映射文件未找到: {mapping_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: 读取映射文件时出错: {e}")
        sys.exit(1)

    print(f"加载了 {len(original_to_renamed_map)} 条映射关系。")

    # -----------------------------------------------------------------

    print(f"\n步骤 2: 正在处理 '{split_csv_path}' 并生成新划分表...")
    
    # 2. 处理原始的 split.csv 并写入新文件
    found_matches = 0
    missing_in_mapping = 0
    
    try:
        with open(split_csv_path, 'r', encoding='utf-8') as f_csv, \
             open(output_file_path, 'w', encoding='utf-8', newline='') as f_out:
            
            reader = csv.DictReader(f_csv)
            
            # 写入标题行
            f_out.write("renamed_image-original_filename-renamed_index-split\n")
            
            for row in tqdm(reader, desc="生成划分"):
                original_path = row.get('filepath')
                split_label = row.get('split')
                
                if not original_path or not split_label:
                    print("警告: 跳过 'filepath' 或 'split' 字段缺失的行。")
                    continue
                    
                # 核心逻辑：在映射字典中查找
                if original_path in original_to_renamed_map:
                    info = original_to_renamed_map[original_path]
                    
                    # 格式化输出: 000000.jpg-008e57...JPG-000000-train
                    output_line = (
                        f"{info['renamed_name']}-"
                        f"{info['original_filename']}-"
                        f"{info['renamed_index']}-"
                        f"{split_label}\n"
                    )
                    f_out.write(output_line)
                    found_matches += 1
                else:
                    # CSV 中的这个文件，在之前的脚本处理中被跳过了 (例如 XML 中没有)
                    missing_in_mapping += 1
                        
    except FileNotFoundError:
        print(f"❌ 错误: 划分 CSV 文件未找到: {split_csv_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ 错误: CSV 文件缺少必需的列: {e}。 (需要 'filepath' 和 'split')")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: 处理 CSV 文件时出错: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------

    print("\n" + "=" * 30)
    print("✅ 划分文件生成完毕！")
    print(f"新划分文件已保存至: {output_file_path}")
    print(f"总共写入 {found_matches} 条记录。")
    if missing_in_mapping > 0:
        print(f"⚠️ 警告: {missing_in_mapping} 条来自 CSV 的记录在映射文件中未找到，已被跳过。")
    
    # 打印新文件的前几行作为预览
    print("\n新文件内容预览:")
    try:
        with open(output_file_path, 'r', encoding='utf-8') as f_prev:
            for i, line in enumerate(f_prev):
                if i > 5: # 只显示 1 个标题行 + 5 条数据
                    break
                print(line.strip())
    except Exception:
        pass # 预览失败也没关系

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将重命名的文件映射到 train/test 划分。")
    
    parser.add_argument('--mapping_file', type=str, required=True,
                        help="您生成的 'original_to_renamed_mapping.txt' 文件的路径。")
                        
    parser.add_argument('--split_csv', type=str, required=True,
                        help="包含原始图像 train/test 划分的 CSV 文件路径。")
                        
    parser.add_argument('--output_file', type=str, default="renamed_splits.txt",
                        help="输出的新划分文件的名称 (默认: renamed_splits.txt)。")
    
    args = parser.parse_args()
    
    create_renamed_split_file(args.mapping_file, args.split_csv, args.output_file)

'''
python szq_c_merge_split.py \
    --mapping_file /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/finished/sztu/original_to_renamed_mapping.txt \
    --split_csv /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/geospatial_split.csv \
    --output_file /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/splits_sztu.txt

'''