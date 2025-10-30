# # #!/bin/bash

# # # ==============================================================================
# # # 批量处理脚本 (用于 1_process_multi_las_gpu.py)
# # #
# # # 功能:
# # # 1. 定义多个独立的数据集任务 (SBMU, SZTTI)。
# # # 2. 逐次执行每个任务。
# # # 3. 使用 'set -e' 确保任何任务失败时，脚本会立即停止。
# # # ==============================================================================

# # set -e

# # # --- 配置 ---
# # # 请将此名称修改为您保存的 Python 脚本的名称
# # PYTHON_SCRIPT_NAME="szq_1_process_each_line_las_no_split.py"


# # # ==============================================================================
# # # 任务 1: SBMU 数据集
# # # ==============================================================================
# # run_task_SBMU() {
# #     echo ""
# #     echo "================================================="
# #     echo "=== 正在开始 LiDAR 匹配任务: SBMU ==="
# #     echo "================================================="

# #     # --- SBMU 任务参数 ---
# #     # 第0步脚本的输出目录 (作为输入)
# #     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav"
    
# #     # (已修改) 包含 .las/.laz 文件块的目录
# #     LAS_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/hav"
    
# #     # 最终 .pkl 文件的输出目录 (通常与 DATASET_PATH 相同)
# #     LAS_OUTPUT_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav"
    

# #     # --- 执行 SBMU 任务 ---
# #     # 注意：已移除 --num_val
# #     # 注意：--las_path 已修改为 --las_dir
# #     python $PYTHON_SCRIPT_NAME \
# #         --dataset_path "$DATASET_PATH" \
# #         --las_dir "$LAS_DIR" \
# #         --las_output_path "$LAS_OUTPUT_PATH"

# #     echo "=== 任务 SBMU 完成 ==="
# # }



# # # ==============================================================================
# # # --- 主执行流程 ---
# # # 脚本将按顺序执行此处列出的任务
# # # ==============================================================================

# # echo "======= 批量 LiDAR 处理任务启动 ======="

# # run_task_SBMU
# # run_task_SZTTI

# # echo ""
# # echo "======= 所有批量处理任务已成功完成 ======="

# #!/bin/bash

# # ==============================================================================
# # 批量处理脚本 (用于 1_process_multi_las_gpu.py) - 已修正
# #
# # 功能:
# # 1. 定义多个独立的数据集任务 (SBMU, SZTTI)。
# # 2. 逐次执行每个任务。
# # 3. 使用 'set -e' 确保任何任务失败时，脚本会立即停止。
# # ==============================================================================

# set -e

# # --- 配置 ---
# # (请确保这里的名称与您运行的 Python 脚本一致)
# PYTHON_SCRIPT_NAME="szq_1_process_each_line_las_no_split.py" 


# # ==============================================================================
# # 任务 1: SBMU 数据集
# # ==============================================================================
# run_task_SBMU() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始 LiDAR 匹配任务: SBMU ==="
#     echo "================================================="


#         # --- SBMU 任务参数 ---
#     # 第0步脚本的输出目录 (作为输入)
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav"
    
#     # (已修改) 包含 .las/.laz 文件块的目录
#     LAS_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/hav"
    
#     # 最终 .pkl 文件的输出目录 (通常与 DATASET_PATH 相同)
#     LAS_OUTPUT_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/hav"

#     # --- 执行 SBMU 任务 ---
#     # (!!!) (已修正) (!!!)
#     # 将 --las_dir 修改为 --las_path
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH" \
#         --las_path "$LAS_DIR" \
#         --las_output_path "$LAS_OUTPUT_PATH"

#     echo "=== 任务 SBMU 完成 ==="
# }



# # ==============================================================================
# # --- 主执行流程 ---
# # ==============================================================================

# echo "======= 批量 LiDAR 处理任务启动 ======="

# run_task_SBMU
# run_task_SZTTI

# echo ""
# echo "======= 所有批量处理任务已成功完成 ======="


#!/bin/bash
# ==============================================================================
# 批量处理脚本 (用于 1_process_multi_las_gpu.py) - 已修正
#
# 功能:
# 1. 定义多个独立的数据集任务 (SBMU, SZTTI)。
# 2. 逐次执行每个任务。
# 3. 使用 'set -e' 确保任何任务失败时，脚本会立即停止。
# ==============================================================================

set -e

# --- 配置 ---
# (!!!) (核心修正) 
# (请确保这里的名称与我们刚刚修复的、使用“相对时间对齐”的 Python 脚本一致)
PYTHON_SCRIPT_NAME="szq_1_process_each_line_las_no_split.py" 


# ==============================================================================
# 任务 1: SBMU 数据集 (hav)
# ==============================================================================
# run_task_SMBU() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始 LiDAR 匹配任务: SBMU (hav) ==="
#     echo "================================================="

#     # --- SBMU 任务参数 ---
#     # 第0步脚本的输出目录 (包含 image_metadata)
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/SMBU"
    
#     # (!!!) 包含所有 .las/.laz 文件的 *根* 目录 (脚本会递归搜索)
#     LAS_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/SMBU"
    
#     # 最终 .npy 分段文件的输出目录 (通常与 DATASET_PATH 相同)
#     LAS_OUTPUT_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/SMBU"

#     # --- 执行 SBMU 任务 ---
#     # (我们使用 --las_path，这与 Python 脚本的参数一致)
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH" \
#         --las_path "$LAS_DIR" \
#         --las_output_path "$LAS_OUTPUT_PATH"

#     echo "=== 任务 SBMU (hav) 完成 ==="
# }
run_task_lfls() {
    echo ""
    echo "================================================="
    echo "=== 正在开始 LiDAR 匹配任务: SBMU (hav) ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    # 第0步脚本的输出目录 (包含 image_metadata)
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls"
    
    # (!!!) 包含所有 .las/.laz 文件的 *根* 目录 (脚本会递归搜索)
    LAS_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/lfls"
    # 最终 .npy 分段文件的输出目录 (通常与 DATASET_PATH 相同)
    LAS_OUTPUT_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls"

    # --- 执行 SBMU 任务 ---
    # (我们使用 --las_path，这与 Python 脚本的参数一致)
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --las_path "$LAS_DIR" \
        --las_output_path "$LAS_OUTPUT_PATH"

    echo "=== 任务 SBMU (hav) 完成 ==="
}
# run_task_lfls2() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始 LiDAR 匹配任务: SBMU (hav) ==="
#     echo "================================================="

#     # --- SBMU 任务参数 ---
#     # 第0步脚本的输出目录 (包含 image_metadata)
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls2"
    
#     # (!!!) 包含所有 .las/.laz 文件的 *根* 目录 (脚本会递归搜索)
#     LAS_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/lfls2"
    
#     # 最终 .npy 分段文件的输出目录 (通常与 DATASET_PATH 相同)
#     LAS_OUTPUT_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls2"

#     # --- 执行 SBMU 任务 ---
#     # (我们使用 --las_path，这与 Python 脚本的参数一致)
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH" \
#         --las_path "$LAS_DIR" \
#         --las_output_path "$LAS_OUTPUT_PATH"

#     echo "=== 任务 SBMU (hav) 完成 ==="
# }
# # ==============================================================================
# # 任务 2: SZTTI 数据集 (示例)
# # ==============================================================================
# run_task_SZTTI() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始 LiDAR 匹配任务: SZTTI ==="
#     echo "================================================="

#     # (!!!) (请在此处填写 SZTTI 的真实路径) (!!!)

#     # --- SZTTI 任务参数 ---
#     # 第0步脚本的输出目录 (包含 image_metadata)
#     DATASET_PATH="/path/to/your/output_SZTTI"
    
#     # (!!!) 包含所有 .las/.laz 文件的 *根* 目录 (脚本会递归搜索)
#     LAS_DIR="/path/to/your/raw_lidar/SZTTI_root"
    
#     # 最终 .npy 分段文件的输出目录 (通常与 DATASET_PATH 相同)
#     LAS_OUTPUT_PATH="/path/to/your/output_SZTTI"

#     # --- 执行 SZTTI 任务 ---
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH" \
#         --las_path "$LAS_DIR" \
#         --las_output_path "$LAS_OUTPUT_PATH"

#     echo "=== 任务 SZTTI 完成 ==="
# }

run_task_upper() {
    echo ""
    echo "================================================="
    echo "=== 正在开始 LiDAR 匹配任务: upper ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    # 第0步脚本的输出目录 (包含 image_metadata)
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/upper"
    
    # (!!!) 包含所有 .las/.laz 文件的 *根* 目录 (脚本会递归搜索)
    LAS_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/upper"
    
    # 最终 .npy 分段文件的输出目录 (通常与 DATASET_PATH 相同)
    LAS_OUTPUT_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/upper"

    # --- 执行 SBMU 任务 ---
    # (我们使用 --las_path，这与 Python 脚本的参数一致)
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --las_path "$LAS_DIR" \
        --las_output_path "$LAS_OUTPUT_PATH"

    echo "=== 任务 upper 完成 ==="
}
run_task_lower() {
    echo ""
    echo "================================================="
    echo "=== 正在开始 LiDAR 匹配任务: lower ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    # 第0步脚本的输出目录 (包含 image_metadata)
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lower"
    
    # (!!!) 包含所有 .las/.laz 文件的 *根* 目录 (脚本会递归搜索)
    LAS_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/lower"
    
    # 最终 .npy 分段文件的输出目录 (通常与 DATASET_PATH 相同)
    LAS_OUTPUT_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lower"

    # --- 执行 SBMU 任务 ---
    # (我们使用 --las_path，这与 Python 脚本的参数一致)
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --las_path "$LAS_DIR" \
        --las_output_path "$LAS_OUTPUT_PATH"

    echo "=== 任务 lower 完成 ==="
}
run_task_sziit() {
    echo ""
    echo "================================================="
    echo "=== 正在开始 LiDAR 匹配任务:sziit ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    # 第0步脚本的输出目录 (包含 image_metadata)
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sziit"
    
    # (!!!) 包含所有 .las/.laz 文件的 *根* 目录 (脚本会递归搜索)
    LAS_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/sziit"
    
    # 最终 .npy 分段文件的输出目录 (通常与 DATASET_PATH 相同)
    LAS_OUTPUT_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sziit"

    # --- 执行 SBMU 任务 ---
    # (我们使用 --las_path，这与 Python 脚本的参数一致)
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --las_path "$LAS_DIR" \
        --las_output_path "$LAS_OUTPUT_PATH"

    echo "=== 任务sziit 完成 ==="
}
run_task_sztu() {
    echo ""
    echo "================================================="
    echo "=== 正在开始 LiDAR 匹配任务:sztu ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    # 第0步脚本的输出目录 (包含 image_metadata)
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu"
    
    # (!!!) 包含所有 .las/.laz 文件的 *根* 目录 (脚本会递归搜索)
    LAS_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_lidar/sztu"
    
    # 最终 .npy 分段文件的输出目录 (通常与 DATASET_PATH 相同)
    LAS_OUTPUT_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu"

    # --- 执行 SBMU 任务 ---
    # (我们使用 --las_path，这与 Python 脚本的参数一致)
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --las_path "$LAS_DIR" \
        --las_output_path "$LAS_OUTPUT_PATH"

    echo "=== 任务sztu 完成 ==="
}
# ==============================================================================
# --- 主执行流程 ---
# ==============================================================================

echo "======= 批量 LiDAR 处理任务启动 ======="

# 1. 运行 SBMU


# 2. 运行 SZTTI 
# (!!!) (注意: 请确保在上方 run_task_SZTTI 函数中填写了正确的路径) (!!!)
# (!!!) (如果您暂时不想运行此任务，请在行首添加 # 将其注释掉)
run_task_lfls
# run_task_upper
# run_task_lower
# run_task_sziit
# run_task_sztu

echo ""
echo "======= 所有批量处理任务已成功完成 ======="