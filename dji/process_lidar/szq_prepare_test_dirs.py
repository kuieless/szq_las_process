import os
import json

def create_symlink_test_set():
    """
    读取 test.json 并为推理创建一个符号链接目录结构。
    """
    
    # 1. 定义文件和路径
    test_json_file = "/home/data1/szq/Megadepth/metric3D/D2/Metric3D/training/data_server_info/annotations/test.json"
    
    # 定义新测试集的根目录
    # 它将被创建在 test.json 所在目录的上一级，名为 'test_set_for_inference'
    base_output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "test_set_for_inference"
    )
    
    # 定义我们关心的源数据集名称
    # 这将帮助我们从路径中解析出 'hav', 'SMBU', 'sztu'
    source_identifiers = ["hav", "SMBU", "sztu", "lfls2", "upper"]
    
    print(f"将在以下位置创建符号链接集: {base_output_dir}")
    
    # 2. 检查 test.json 是否存在
    if not os.path.exists(test_json_file):
        print(f"错误: '{test_json_file}' 未在当前目录中找到。")
        return

    # 3. 读取 JSON 文件
    try:
        with open(test_json_file, 'r') as f:
            data = json.load(f)
        
        file_list = data.get("files", [])
        if not file_list:
            print("错误: 'test.json' 中没有找到 'files' 列表或列表为空。")
            return
            
    except Exception as e:
        print(f"读取或解析 'test.json' 时出错: {e}")
        return

    # 4. 遍历文件并创建符号链接
    symlink_count = 0
    for entry in file_list:
        original_rgb_path = entry.get("rgb")
        if not original_rgb_path:
            continue
            
        # 5. 确定源数据集 (hav, SMBU, or sztu)
        source_name = None
        for name in source_identifiers:
            if f"images_gt_{name}" in original_rgb_path:
                source_name = name
                break
        
        if not source_name:
            print(f"警告: 无法从路径中确定源: {original_rgb_path}")
            continue
            
        # 6. 创建目标目录
        target_dir = os.path.join(base_output_dir, source_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 7. 创建符号链接
        image_filename = os.path.basename(original_rgb_path)
        symlink_path = os.path.join(target_dir, image_filename)
        
        try:
            # 检查原始文件是否存在
            if not os.path.exists(original_rgb_path):
                print(f"警告: 源文件不存在，跳过: {original_rgb_path}")
                continue
                
            # 如果符号链接已存在，则跳过
            if os.path.exists(symlink_path) or os.path.lexists(symlink_path):
                continue
                
            os.symlink(original_rgb_path, symlink_path)
            symlink_count += 1
            
        except Exception as e:
            print(f"创建符号链接时出错 ( {original_rgb_path} -> {symlink_path} ): {e}")

    print("\n--- 准备完成 ---")
    print(f"总共为 {len(file_list)} 个测试条目创建了 {symlink_count} 个新的符号链接。")
    print(f"测试集现已准备好在: {base_output_dir}")

if __name__ == "__main__":
    create_symlink_test_set()