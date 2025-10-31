#!/bin/bash
# ==============================================================================
# 批量处理脚本 (用于 302-visdepth.py - 扁平化版本)
#
# 功能:
# 1. 定义多个独立的数据集可视化和尺度恢复任务。
# 2. 逐次执行每个任务。
# 3. 'set -e' 确保任何任务失败时，脚本会立即停止。
# ==============================================================================

set -e

# --- 配置 ---
# (请确保这里的名称与您要运行的可视化 Python 脚本一致)
PYTHON_SCRIPT_NAME="szq_302-visdepth.py" 


# ==============================================================================
# 任务 1: hav 数据集 (来自您的示例)
# ==============================================================================
# run_task_test() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始可视化任务: hav ==="
#     echo "================================================="

#     # --- 任务参数 ---
#     # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/SMBU"

#     # --- 执行任务 ---
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH"

#     echo "=== 任务 hav 完成 ==="
# }
# run_task_SMBU() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始可视化任务: hav ==="
#     echo "================================================="

#     # --- 任务参数 ---
#     # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/SMBU"

#     # --- 执行任务 ---
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH"

#     echo "=== 任务 hav 完成 ==="
# }

# run_task_lfls() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始可视化任务: hav ==="
#     echo "================================================="

#     # --- 任务参数 ---
#     # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls"

#     # --- 执行任务 ---
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH"

#     echo "=== 任务 hav 完成 ==="
# }
# run_task_lfls2() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始可视化任务: hav ==="
#     echo "================================================="

#     # --- 任务参数 ---
#     # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls2"

#     # --- 执行任务 ---
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH"

#     echo "=== 任务 hav 完成 ==="
# }
run_task_lower() {
    echo ""
    echo "================================================="
    echo "=== 正在开始可视化任务: lower ==="
    echo "================================================="

    # --- 任务参数 ---
    # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lower"

    # --- 执行任务 ---
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH"

    echo "=== 任务 lower 完成 ==="
}

run_task_sziit() {
    echo ""
    echo "================================================="
    echo "=== 正在开始可视化任务: sziit ==="
    echo "================================================="

    # --- 任务参数 ---
    # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sziit"

    # --- 执行任务 ---
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH"

    echo "=== 任务 sziit 完成 ==="
}

run_task_sztu() {
    echo ""
    echo "================================================="
    echo "=== 正在开始可视化任务: sztu ==="
    echo "================================================="

    # --- 任务参数 ---
    # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu"

    # --- 执行任务 ---
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH"

    echo "=== 任务 sztu 完成 ==="
}

run_task_upper() {
    echo ""
    echo "================================================="
    echo "=== 正在开始可视化任务: upper ==="
    echo "================================================="

    # --- 任务参数 ---
    # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/upper"

    # --- 执行任务 ---
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH"

    echo "=== 任务 upper 完成 ==="
}

# # ==============================================================================
# # 任务 2: 另一个数据集 (模板)
# # ==============================================================================
# run_task_ANOTHER_DATASET() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始可视化任务: ANOTHER_DATASET ==="
#     echo "================================================="
    
#     # (!!!) 在此更改为您的下一个数据集路径 (!!!)
#     DATASET_PATH="/path/to/your/next_dataset/output/folder"

#     # --- 执行任务 ---
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH"

#     echo "=== 任务 ANOTHER_DATASET 完成 ==="
# }


# ==============================================================================
# --- 主执行流程 ---
# ==============================================================================

echo "======= 批量深度图可视化与尺度恢复任务启动 ======="

# 1. 运行 hav (来自您的示例)
# run_task_SMBU  
# run_task_lfls
# run_task_lfls2
# run_task_lower
# run_task_sziit



run_task_sztu
run_task_upper
# 2. 运行下一个
# (!!!) 
# (!!!) 要运行更多任务，请复制 run_task_ANOTHER_DATASET 函数，
# (!!!) 修改它，然后在这里取消注释并调用它。
# (!!!)
# run_task_ANOTHER_DATASET


echo ""
echo "======= 所有批量处理任务已成功完成 ======="