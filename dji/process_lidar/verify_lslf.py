import json
import xml.etree.ElementTree as ET
import os
import sys

# --- 请修改以下三个文件名 ---
XML_FILE_PATH = '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plfls45-47,65-67,70-72/AT/BlocksExchangeUndistortAT.xml'     # 替换为你的 XML 文件名
JSON_FILE_PATH = '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/plfls45-47,65-67,70-72.json'   # 替换为你的 JSON 文件名
OUTPUT_JSON_PATH = '/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/lfls-new.json' # 这是将要创建的新文件名

# ------------------------------

# 调试开关：设置为 True 会打印前 5 个提取到的 ID，用于检查
PRINT_DEBUG_IDS = True 

def extract_ids_from_xml(xml_path):
    """
    从 BlocksExchange XML 文件中解析所有 <Photo> 记录，
    并从 <ImagePath> 中提取图像 ID。
    """
    print(f"正在从 {xml_path} 加载 XML 记录...")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"错误: 无法解析 XML 文件。{e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"错误: 找不到 XML 文件 {xml_path}", file=sys.stderr)
        sys.exit(1)

    xml_image_ids = set()
    first_five_ids = [] # 用于调试

    for photo in root.findall('.//Photo'):
        image_path_element = photo.find('ImagePath')
        if image_path_element is not None and image_path_element.text:
            image_path = image_path_element.text
            
            # --- 这是关键的修复 ---
            # 1. 将所有 Windows 路径分隔符 \ 替换为 /
            normalized_path = image_path.replace('\\', '/')
            
            # 2. 现在 os.path.basename 可以正确工作
            basename = os.path.basename(normalized_path)
            # -----------------------
            
            # 3. 获取不带扩展名的 ID
            image_id, _ = os.path.splitext(basename)
            
            xml_image_ids.add(image_id)
            
            if PRINT_DEBUG_IDS and len(first_five_ids) < 5:
                first_five_ids.append(image_id)

    if PRINT_DEBUG_IDS:
        print("--- [调试] 从 XML 提取的前 5 个 ID ---")
        for i, id_val in enumerate(first_five_ids):
            print(f"  XML ID {i+1}: '{id_val}'")
        print("--------------------------------------")

    print(f"从 XML 中成功加载 {len(xml_image_ids)} 条位姿 ID。")
    return xml_image_ids

def filter_json_by_xml_ids(json_path, xml_ids):
    """
    加载 JSON 文件，并根据 XML ID 集合来过滤它。
    """
    print(f"正在从 {json_path} 加载 JSON 记录...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 JSON 文件。{e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"错误: 找不到 JSON 文件 {json_path}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(json_data, list):
        print(f"错误: JSON 文件的根结构不是一个列表 (list)。", file=sys.stderr)
        sys.exit(1)

    original_count = len(json_data)
    print(f"从 JSON 中成功加载 {original_count} 条图像记录。")
    
    print("正在根据 XML ID 过滤 JSON 记录...")
    filtered_data = []
    
    json_ids_debug = [] # 用于调试
    
    for record in json_data:
        if 'id' not in record:
            print(f"警告: 发现一条没有 'id' 字段的 JSON 记录，已跳过: {record}")
            continue
        
        json_id = record['id']
        
        if PRINT_DEBUG_IDS and len(json_ids_debug) < 5:
            json_ids_debug.append(json_id)

        # 核心逻辑：只保留 ID 存在于 XML ID 集合中的记录
        if json_id in xml_ids:
            filtered_data.append(record)

    if PRINT_DEBUG_IDS:
        print("--- [调试] 从 JSON 检查的前 5 个 ID ---")
        for i, id_val in enumerate(json_ids_debug):
            print(f"  JSON ID {i+1}: '{id_val}'")
        print("-------------------------------------")

    print(f"过滤完毕。共 {len(filtered_data)} 条记录匹配成功。")
    return filtered_data, original_count

def save_new_json(data, output_path):
    """
    将过滤后的数据保存为新的 JSON 文件。
    """
    print(f"正在将过滤后的数据保存到 {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("保存成功！")
    except IOError as e:
        print(f"错误: 无法写入文件 {output_path}。{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    xml_ids = extract_ids_from_xml(XML_FILE_PATH)
    filtered_json_data, original_json_count = filter_json_by_xml_ids(JSON_FILE_PATH, xml_ids)
    save_new_json(filtered_json_data, OUTPUT_JSON_PATH)
    
    print("\n--- 任务完成 ---")
    print(f"XML 记录数:     {len(xml_ids)}")
    print(f"原始 JSON 记录数: {original_json_count}")
    print(f"新 JSON 记录数:   {len(filtered_json_data)}")
    print(f"已创建新文件: {OUTPUT_JSON_PATH}")