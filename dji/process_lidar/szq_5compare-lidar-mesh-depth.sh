# # # #!/bin/bash
# # # # ==============================================================================
# # # # 批量处理脚本 (用于 302-visdepth.py - 扁平化版本)
# # # #
# # # # 功能:
# # # # 1. 定义多个独立的数据集可视化和尺度恢复任务。
# # # # 2. 逐次执行每个任务。
# # # # 3. 'set -e' 确保任何任务失败时，脚本会立即停止。
# # # # ==============================================================================

# # # set -e

# # # # --- 配置 ---
# # # # (请确保这里的名称与您要运行的可视化 Python 脚本一致)
# # # PYTHON_SCRIPT_NAME="szq_302-visdepth.py" 


# # # # ==============================================================================
# # # # 任务 1: hav 数据集 (来自您的示例)
# # # # ==============================================================================
# # # run_task_SMBU() {
# # #     echo ""
# # #     echo "================================================="
# # #     echo "=== 正在开始可视化任务: hav ==="
# # #     echo "================================================="

# # #     # --- 任务参数 ---
# # #     # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
# # #     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/SMBU"

# # #     # --- 执行任务 ---
# # #     python $PYTHON_SCRIPT_NAME \
# # #         --dataset_path "$DATASET_PATH"

# # #     echo "=== 任务 hav 完成 ==="
# # # }

# # # run_task_lfls() {
# # #     echo ""
# # #     echo "================================================="
# # #     echo "=== 正在开始可视化任务: hav ==="
# # #     echo "================================================="

# # #     # --- 任务参数 ---
# # #     # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
# # #     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls"

# # #     # --- 执行任务 ---
# # #     python $PYTHON_SCRIPT_NAME \
# # #         --dataset_path "$DATASET_PATH"

# # #     echo "=== 任务 hav 完成 ==="
# # # }
# # # run_task_lfls2() {
# # #     echo ""
# # #     echo "================================================="
# # #     echo "=== 正在开始可视化任务: hav ==="
# # #     echo "================================================="

# # #     # --- 任务参数 ---
# # #     # (这是包含 depth_dji/, rgbs/, coordinates.pt 的路径)
# # #     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls2"

# # #     # --- 执行任务 ---
# # #     python $PYTHON_SCRIPT_NAME \
# # #         --dataset_path "$DATASET_PATH"

# # #     echo "=== 任务 hav 完成 ==="
# # # }
# # # # # ==============================================================================
# # # # # 任务 2: 另一个数据集 (模板)
# # # # # ==============================================================================
# # # # run_task_ANOTHER_DATASET() {
# # # #     echo ""
# # # #     echo "================================================="
# # # #     echo "=== 正在开始可视化任务: ANOTHER_DATASET ==="
# # # #     echo "================================================="
    
# # # #     # (!!!) 在此更改为您的下一个数据集路径 (!!!)
# # # #     DATASET_PATH="/path/to/your/next_dataset/output/folder"

# # # #     # --- 执行任务 ---
# # # #     python $PYTHON_SCRIPT_NAME \
# # # #         --dataset_path "$DATASET_PATH"

# # # #     echo "=== 任务 ANOTHER_DATASET 完成 ==="
# # # # }


# # # # ==============================================================================
# # # # --- 主执行流程 ---
# # # # ==============================================================================

# # # echo "======= 批量深度图可视化与尺度恢复任务启动 ======="

# # # # 1. 运行 hav (来自您的示例)
# # # run_task_SMBU  
# # # run_task_lfls
# # # run_task_lfls2
    
# # # # 2. 运行下一个
# # # # (!!!) 
# # # # (!!!) 要运行更多任务，请复制 run_task_ANOTHER_DATASET 函数，
# # # # (!!!) 修改它，然后在这里取消注释并调用它。
# # # # (!!!)
# # # # run_task_ANOTHER_DATASET


# # # echo ""
# # # echo "======= 所有批量处理任务已成功完成 ======="


# # #!/bin/bash

# # # ====================================================================
# # # 深度图清理批处理脚本
# # # 在这里修改你的路径和参数
# # # ====================================================================

# # # --- 1. 定义你的路径 ---

# # # 原始 Lidar 深度图 (包含 1,000,000 无效值)
# # LIDAR_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_dji"

# # # 原始 Mesh 深度图 (包含 1,000,000 无效值)
# # MESH_DIR="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_mesh"

# # # 清理后 .npy (GT) 的输出目录
# # OUTPUT_GT_DIR="./my_cleaned_gt/val_set"


# # # --- 2. 定义算法参数 ---

# # # Lidar点比Mesh点深 "多少" 米时，被认为是透视点
# # THRESHOLD=3.0

# # # 大于等于此值的深度被认为是 "无效" (例如 1,000,000)
# # # 脚本会将 >= MAX_DEPTH 的值设为 0
# # MAX_DEPTH=1000.0


# # # --- 3. 可视化开关 ---

# # # 是否跳过可视化 (False=生成可视化, True=不生成)
# # # 设为 "True" 可以大幅加快处理速度
# # # NO_VIZ_FLAG="--no_viz"
# # NO_VIZ_FLAG=""


# # # ====================================================================
# # # 执行命令 (通常不需要修改)
# # # ====================================================================

# # echo "=== 开始清理任务 ==="
# # echo "Lidar 目录: $LIDAR_DIR"
# # echo "Mesh 目录:  $MESH_DIR"
# # echo "输出目录: $OUTPUT_GT_DIR"
# # echo "阈值: $THRESHOLD"
# # echo "最大深度: $MAX_DEPTH"
# # echo "======================"

# # # 激活你的 python 环境 (如果需要的话)
# # # source /path/to/your/conda/bin/activate your_env_name

# # # 运行 python 脚本
# # python clean_depth.py \
# #     --lidar_dir "$LIDAR_DIR" \
# #     --mesh_dir "$MESH_DIR" \
# #     --output_dir "$OUTPUT_GT_DIR" \
# #     --threshold "$THRESHOLD" \
# #     --max_depth "$MAX_DEPTH" \
# #     $NO_VIZ_FLAG

# # echo "=== 任务执行完毕 ==="



# #!/bin/bash

# # ====================================================================
# # 深度图清理 - 多任务批处理脚本
# # ====================================================================

# # 激活你的 python 环境 (如果需要的话)
# # source /path/to/your/conda/bin/activate your_env_name

# # --- 1. 定义核心执行函数 ---
# # (这个函数接收6个参数并运行 Python 脚本)

# run_task() {
#     # $1: 任务名称 (用于日志)
#     # $2: Lidar 目录
#     # $3: Mesh 目录
#     # $4: 输出 (GT) 目录
#     # $5: 阈值 (Threshold)
#     # $6: 最大深度 (Max Depth)
#     # $7: 可视化标志 (No Viz Flag, e.g., "--no_viz" or "")
    
#     local TASK_NAME="$1"
#     local LIDAR_DIR="$2"
#     local MESH_DIR="$3"
#     local OUTPUT_GT_DIR="$4"
#     local THRESHOLD="$5"
#     local MAX_DEPTH="$6"
#     local NO_VIZ_FLAG="$7"

#     echo ""
#     echo "=========================================================="
#     echo "=== 启动任务: $TASK_NAME"
#     echo "=========================================================="
#     echo "  Lidar 目录: $LIDAR_DIR"
#     echo "  Mesh 目录:  $MESH_DIR"
#     echo "  输出目录: $OUTPUT_GT_DIR"
#     echo "  阈值: $THRESHOLD"
#     echo "  最大深度: $MAX_DEPTH"
#     echo "  可视化标志: ${NO_VIZ_FLAG:-"False"}"
#     echo "----------------------------------------------------------"

#     # 运行 python 脚本
#     # (确保 clean_depth.py 在同一目录或在 $PATH 中)
#     python clean_depth.py \
#         --lidar_dir "$LIDAR_DIR" \
#         --mesh_dir "$MESH_DIR" \
#         --output_dir "$OUTPUT_GT_DIR" \
#         --threshold "$THRESHOLD" \
#         --max_depth "$MAX_DEPTH" \
#         $NO_VIZ_FLAG
        
#     echo "=== 任务: $TASK_NAME 完成 ==="
# }


# # ====================================================================
# # --- 2. 在这里定义和执行您的所有任务 ---
# # ====================================================================

# # 您可以复制/粘贴这些 "run_task" 调用来创建任意多的任务

# # --- 任务 1: 处理验证集 (Val Set) ---
# run_task \
#     "Validation_Set_Cleaning" \
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_dji" \
#     "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7/val/depth_mesh" \
#     "./my_cleaned_gt/val_set" \
#     3.0 \
#     1000.0 \
#     ""  # (可视化标志: 留空 = "False", 填 "--no_viz" = "True")


# # --- 任务 2: 处理训练集 (Train Set) - 示例 ---
# # (注意: 这里我们传入了 "--no_viz" 来跳过可视化)
# # run_task \
# #     "Training_Set_Cleaning" \
# #     "/path/to/your/train/depth_dji" \
# #     "/path/to/your/train/depth_mesh" \
# #     "./my_cleaned_gt/train_set" \
# #     3.0 \
# #     1000.0 \
# #     "--no_viz"


# # --- 任务 3: 使用不同阈值处理另一批数据 - 示例 ---
# # run_task \
# #     "Test_Set_New_Threshold" \
# #     "/path/to/another/dataset/dji" \
# #     "/path/to/another/dataset/mesh" \
# #     "./my_cleaned_gt/another_set_thresh_5" \
# #     5.0 \
# #     1000.0 \
# #     ""


# # ====================================================================
# echo ""
# echo "所有批处理任务已执行完毕。"
# # ====================================================================



#!/bin/bash

# ====================================================================
# 深度图清理 - 多场景（Scene）批处理脚本
#
# 使用方法:
# 1. 在下方 "2. 定义和执行您的所有场景" 区域
# 2. 为您要处理的 *每一个* 场景，复制一个 "run_scene" 任务块
# 3. 填入该场景的 Lidar, Mesh, Output 路径和参数
# ====================================================================

# 激活你的 python 环境 (如果需要的话)
# source /path/to/your/conda/bin/activate your_env_name

# --- 1. 定义核心执行函数 ---
# (这个函数接收6个参数并运行 Python 脚本)

run_scene() {
    # $1: 场景名称 (用于日志)
    # $2: Lidar 目录
    # $3: Mesh 目录
    # $4: 输出 (GT) 目录
    # $5: 阈值 (Threshold)
    # $6: 最大深度 (Max Depth)
    # $7: 可视化标志 (No Viz Flag, e.g., "--no_viz" or "")
    
    local SCENE_NAME="$1"
    local LIDAR_DIR="$2"
    local MESH_DIR="$3"
    local OUTPUT_GT_DIR="$4"
    local THRESHOLD="$5"
    local MAX_DEPTH="$6"
    local NO_VIZ_FLAG="$7"
    local NUM_CORES="$8"
    echo ""
    echo "=========================================================="
    echo "=== 开始处理场景: $SCENE_NAME"
    echo "=== 开始处理场景: $SCENE_NAME (使用 $NUM_CORES 核心)"
    echo "=========================================================="
    echo "  Lidar 目录: $LIDAR_DIR"
    echo "  Mesh 目录:  $MESH_DIR"
    echo "  输出目录: $OUTPUT_GT_DIR"
    echo "  阈值: $THRESHOLD"
    echo "  最大深度: $MAX_DEPTH"
    echo "  可视化标志: ${NO_VIZ_FLAG:-"False"}"
    echo "----------------------------------------------------------"

    # 运行 python 脚本
    # (确保 clean_depth.py 在同一目录或在 $PATH 中)
    python szq_5compare-lidar-mesh-depth.py \
        --lidar_dir "$LIDAR_DIR" \
        --mesh_dir "$MESH_DIR" \
        --output_dir "$OUTPUT_GT_DIR" \
        --threshold "$THRESHOLD" \
        --max_depth "$MAX_DEPTH" \
        -j "$NUM_CORES" \
        $NO_VIZ_FLAG
        
    echo "=== 场景: $SCENE_NAME 处理完成 ==="
}


# ====================================================================
# --- 2. 在这里定义和执行您的所有场景 ---
# ====================================================================

# 您可以复制/粘贴这些 "run_scene" 调用来创建任意多的场景

# --- 场景 1: "DJI 验证集" (这是您的示例) ---
run_scene \
    "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu" \
    "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu/depth_metric" \
    "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu/depth_mesh" \
    "/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu/depth_gt_sztu" \
    1.0 \
    1000.0 \
    ""   \
    -1  
    
#(可视化标志: 留空 = "False", 填 "--no_viz" = "True")
# # --- 场景 2: "另一个城市场景" (示例) ---
# # (注意: 这个场景不生成可视化图，并且使用 5.0 的阈值)
# run_scene \
#     "Urban_Scene_02" \
#     "/path/to/scene2/lidar_data" \
#     "/path/to/scene2/mesh_data" \
#     "./my_cleaned_gt/urban_scene_02" \
#     5.0 \
#     1000.0 \
#     "--no_viz"


# # --- 场景 3: "乡村场景" (示例) ---
# run_scene \
#     "Rural_Scene_01" \
#     "/path/to/scene3_rural/lidar" \
#     "/path/to/scene3_rural/mesh" \
#     "./my_cleaned_gt/rural_scene_01" \
#     3.0 \
#     1000.0 \
#     ""


# ====================================================================
echo ""
echo "所有场景已执行完毕。"
# ====================================================================