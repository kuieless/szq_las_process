# #!/bin/bash

# # --- 配置 ---
# # 设置你的Python解释器命令 (例如 python, python3)
# # 推荐使用 python3
# PYTHON_CMD="python3"

# # 你的新的Python脚本的文件名
# # **请确保这个文件名与你保存的Python脚本一致**
# PYTHON_SCRIPT_NAME="00create_image_list_v2.py"
# # --- 结束配置 ---


# # --- 脚本主逻辑 ---

# # 1. 检查参数 (需要2个参数)
# if [ $# -ne 2 ]; then
#     echo "错误: 参数数量不正确。"
#     echo "用法: $0 <父输入文件夹> <父输出文件夹>"
#     echo "示例: $0 /data/my_projects 00output"
#     exit 1
# fi

# PARENT_INPUT_DIR="$1"
# PARENT_OUTPUT_DIR="$2"

# # 2. 检查输入父文件夹是否存在
# if [ ! -d "$PARENT_INPUT_DIR" ]; then
#     echo "错误: 输入父文件夹 '$PARENT_INPUT_DIR' 不存在。"
#     exit 1
# fi

# # 3. 检查并创建输出文件夹
# if [ ! -d "$PARENT_OUTPUT_DIR" ]; then
#     echo "输出文件夹 '$PARENT_OUTPUT_DIR' 不存在，正在创建..."
#     # -p 确保如果路径中间的目录不存在也会一并创建
#     mkdir -p "$PARENT_OUTPUT_DIR"
#     if [ $? -ne 0 ]; then
#         echo "错误: 无法创建输出文件夹 '$PARENT_OUTPUT_DIR'。"
#         exit 1
#     fi
#     echo "已创建输出文件夹: $PARENT_OUTPUT_DIR"
# fi
# # 获取输出文件夹的绝对路径，这在日志中更清晰
# PARENT_OUTPUT_DIR_ABS=$(cd "$PARENT_OUTPUT_DIR" &> /dev/null && pwd)


# # 4. 检查 Python 脚本是否存在
# # 假设 Python 脚本与此 bash 脚本位于同一目录
# SCRIPT_DIR_ABS=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# PYTHON_SCRIPT_PATH="$SCRIPT_DIR_ABS/$PYTHON_SCRIPT_NAME"

# if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
#     echo "错误: Python 脚本未找到: $PYTHON_SCRIPT_PATH"
#     echo "请确保 $PYTHON_SCRIPT_NAME 与此 bash 脚本放在同一目录下。"
#     exit 1
# fi

# echo "开始批量处理..."
# echo "输入父文件夹: $PARENT_INPUT_DIR"
# echo "输出文件夹:   $PARENT_OUTPUT_DIR_ABS"
# echo "Python 脚本:  $PYTHON_SCRIPT_PATH"
# echo "======================================================"

# # 5. 查找所有一级子文件夹 (例如 PSZIIT018, PSZIIT019)
# # 使用 find 和 while read 循环，可以安全处理带空格的文件夹名
# find "$PARENT_INPUT_DIR" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r project_dir_path; do
    
#     # 获取项目文件夹的名称 (例如 "PSZIIT018")
#     project_name=$(basename "$project_dir_path")
    
#     # 定义要扫描的实际图片文件夹路径
#     # 根据你的结构: <project_dir_path>/images/survey
#     image_scan_dir="$project_dir_path/images/survey"
    
#     # 定义输出 JSON 文件的完整路径
#     # 例如: 00output/PSZIIT018.json
#     output_json_path="$PARENT_OUTPUT_DIR_ABS/$project_name.json"
    
#     echo
#     echo "--- 正在处理项目: $project_name ---"
    
#     # 检查目标扫描文件夹 (images/survey) 是否存在
#     if [ ! -d "$image_scan_dir" ]; then
#         echo "警告: 未找到图片文件夹 '$image_scan_dir'"
#         echo "--- 跳过项目: $project_name ---"
#         continue # 跳过这个循环，处理下一个项目
#     fi
    
#     # 6. 执行Python脚本
#     # $PYTHON_CMD "$PYTHON_SCRIPT_PATH" "<输入图片文件夹>" -o "<输出JSON路径>"
#     echo "执行: $PYTHON_CMD \"$PYTHON_SCRIPT_PATH\" \"$image_scan_dir\" -o \"$output_json_path\""
    
#     "$PYTHON_CMD" "$PYTHON_SCRIPT_PATH" "$image_scan_dir" -o "$output_json_path"
    
#     if [ $? -eq 0 ]; then
#         echo "--- 成功处理: $project_name ---"
#     else
#         # Python 脚本内部的 print(..., file=sys.stderr) 会在这里显示
#         echo "--- 处理失败: $project_name (Python脚本返回错误) ---"
#     fi
    
# done

# echo
# echo "======================================================"
# echo "所有项目处理完毕。"
# echo "请检查 '$PARENT_OUTPUT_DIR_ABS' 文件夹以获取 JSON 文件。"

# #./run_batch_survey.sh /data/my_projects 00output

#!/bin/bash

# --- 配置 ---
# 设置你的Python解释器命令 (例如 python, python3)
PYTHON_CMD="python3"

# 你的Python脚本的文件名
# **请确保这个文件名与你保存的Python脚本一致**
PYTHON_SCRIPT_NAME="szq_00create_image_list_v2.py"
# --- 结束配置 ---


# --- 脚本主逻辑 ---

# 1. 检查 Python 脚本是否存在
# 获取此 bash 脚本所在的绝对目录
SCRIPT_DIR_ABS=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PYTHON_SCRIPT_PATH="$SCRIPT_DIR_ABS/$PYTHON_SCRIPT_NAME"

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "错误: Python 脚本未找到: $PYTHON_SCRIPT_PATH"
    echo "请确保 $PYTHON_SCRIPT_NAME 与此 bash 脚本放在同一目录下。"
    exit 1
fi

# 2. 定义一个辅助函数来执行单个任务
# 用法: run_task <任务名称> <输入图片文件夹> <输出JSON文件>
run_task() {
    # $1, $2, $3 是传递给函数的参数
    local task_name="$1"
    local input_dir="$2"
    local output_file="$3"

    echo
    echo "--- 正在处理任务: $task_name ---"
    
    # 打印将要执行的命令，方便调试
    echo "执行: $PYTHON_CMD \"$PYTHON_SCRIPT_PATH\" \"$input_dir\" -o \"$output_file\""
    
    # 执行Python脚本
    # 我们将参数用引号括起来，以安全处理包含空格的路径
    "$PYTHON_CMD" "$PYTHON_SCRIPT_PATH" "$input_dir" -o "$output_file"
    
    # 检查Python脚本的退出状态
    if [ $? -eq 0 ]; then
        # 退出状态为 0 表示成功
        echo "--- 成功完成: $task_name ---"
    else
        # 退出状态非 0 表示失败
        # Python 脚本内部的 print(..., file=sys.stderr) 错误信息会显示在这里
        echo "--- 处理失败: $task_name (Python脚本返回错误) ---"
    fi
}

# 3. 脚本主执行区
echo "开始执行手动定义的批量任务..."
echo "使用 Python 脚本: $PYTHON_SCRIPT_PATH"
echo "======================================================"

# --- 在这里手动定义您的任务列表 ---
#
# 您可以复制并修改 "run_task" 这一行来添加任意多个任务。
# 它们将按照您书写的顺序依次执行。
#
# 用法: run_task "任务的描述性名称" "完整的输入图片路径" "完整的输出JSON路径"
#

# run_task "项目 pSMBU-all" "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSMBU-all/images/survey" "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pSMBU-all.json"

run_task "项目lfls" "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/rgbsed" "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/rgbsed.json"

# run_task "项目pSZTU" "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/pSZTU/images/survey" "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pSZTU.json"

# run_task "项目pupper" "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/pupper/images/survey" "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pupper.json"

# run_task "一个完全不同的项目" "/data/other_location/Project_X/raw_scans" "00output/other/Project_X_list.json"

# run_task "测试项目 (子文件夹)" "/data/my_projects/PSZIIT020/images/survey/subset1" "00output/PSZIIT020_subset1.json"

# run_task "下一个任务" "/path/to/input/folder" "/path/to/output/file.json"
# run_task "再下一个任务" "/another/path/input" "/another/path/output.json"


# --- 任务定义结束 ---

echo
echo "======================================================"
echo "所有手动定义的任务处理完毕。"

# #./szq_00run_batch_survey.sh 