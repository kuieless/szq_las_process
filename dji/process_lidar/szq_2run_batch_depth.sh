#!/bin/bash

# ==============================================================================
# 批量处理脚本 (用于 2_project_lidar_to_depth.py)
#
# 功能:
# 1. 定义多个独立的数据集任务 (SBMU, SZTTI)。
# 2. 逐次执行每个任务。
# 3. 使用 'set -e' 确保任何任务失败时，脚本会立即停止。
# ==============================================================================

set -e

# --- 配置 ---
# 请将此名称修改为您保存的 Python 脚本的名称
PYTHON_SCRIPT_NAME="szq_2_convert_lidar_2_depth_color.py"


# ==============================================================================
# 任务 1: SBMU 数据集
# ==============================================================================
# run_task_SMBU() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始点云投影任务: SBMU ==="
#     echo "================================================="

#     # --- SBMU 任务参数 ---
    
#     # (输入/输出) 脚本 0 和 1 的主输出目录
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/SMBU"
    
#     # (输入) 原始图像
#     ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSMBU-all/images/survey"
#     INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSMBU-all/AT/BlocksExchangeUndistortAT.xml"
#     JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pSMBU-all.json"
    
#     # (输入) 'points_lidar_list.pkl' 所在的目录 (与 DATASET_PATH 相同)
#     LAS_OUTPUT_PATH="$DATASET_PATH"
    
#     # (输出) Debug 图像的保存位置
#     DEBUG_OUTPUT_PATH="$DATASET_PATH/debug_projection"

#     # --- SBMU 畸变参数 (与脚本0一致) ---

#     FX="3691.520"
#     FY="3691.520"
#     CX="2755.450"
#     CY="1796.980"
#     K1=" 0.00185019"
#     K2="-0.00826045"
#     P1="-0.00265205"
#     P2="0.00138173"
#     K3="0.00652772"
    
    
#     # --- 脚本控制参数 ---
#     DEBUG_FLAG="False"  # 设置为 "True" 以生成调试图像，而不是 .npy
#     DOWN_SCALE="4"   # 深度图的降采样比例 (1.0 = 不降采样)
#     START_FRAME="0"
#     END_FRAME="99999"  # 处理所有帧

#     # --- 执行 SBMU 任务 ---
#     # 注意: 移除了 --num_val
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH" \
#         --original_images_path "$ORIGINAL_IMAGES_PATH" \
#         --infos_path "$INFOS_PATH" \
#         --original_images_list_json_path "$JSON_PATH" \
#         --las_output_path "$LAS_OUTPUT_PATH" \
#         --output_path "$DEBUG_OUTPUT_PATH" \
#         --debug "$DEBUG_FLAG" \
#         --down_scale "$DOWN_SCALE" \
#         --start "$START_FRAME" \
#         --end "$END_FRAME" \
#         --fx $FX --fy $FY --cx $CX --cy $CY \
#         --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

#     echo "=== 任务 SBMU 完成 ==="
# }
run_task_lfls() {
    echo ""
    echo "================================================="
    echo "=== 正在开始点云投影任务: lfls ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    
    # (输入/输出) 脚本 0 和 1 的主输出目录
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls"
    
    # (输入) 原始图像
    ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plfls45-47,65-67,70-72/images/survey"
    INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plfls45-47,65-67,70-72/AT/BlocksExchangeUndistortAT.xml"
    JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/lfls-new.json"
    
    # (输入) 'points_lidar_list.pkl' 所在的目录 (与 DATASET_PATH 相同)
    LAS_OUTPUT_PATH="$DATASET_PATH"
    
    # (输出) Debug 图像的保存位置
    DEBUG_OUTPUT_PATH="$DATASET_PATH/debug_projection"

    # --- SBMU 畸变参数 (与脚本0一致) ---

    FX="3691.510"
    FY="3691.510"
    CX="2755.710"
    CY="1796.570"
    K1=" 0.00184453"
    K2="-0.00808798"
    P1="-0.00267461"
    P2="0.00137382"
    K3="0.00642050"
    
    
    # --- 脚本控制参数 ---
    DEBUG_FLAG="False"  # 设置为 "True" 以生成调试图像，而不是 .npy
    DOWN_SCALE="4"   # 深度图的降采样比例 (1.0 = 不降采样)
    START_FRAME="0"
    END_FRAME="99999"  # 处理所有帧

    # --- 执行 SBMU 任务 ---
    # 注意: 移除了 --num_val
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --original_images_path "$ORIGINAL_IMAGES_PATH" \
        --infos_path "$INFOS_PATH" \
        --original_images_list_json_path "$JSON_PATH" \
        --las_output_path "$LAS_OUTPUT_PATH" \
        --output_path "$DEBUG_OUTPUT_PATH" \
        --debug "$DEBUG_FLAG" \
        --down_scale "$DOWN_SCALE" \
        --start "$START_FRAME" \
        --end "$END_FRAME" \
        --fx $FX --fy $FY --cx $CX --cy $CY \
        --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

    echo "=== 任务 lfls 完成 ==="
}

run_task_lfls2() {
    echo ""
    echo "================================================="
    echo "=== 正在开始点云投影任务: lfls2 ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    
    # (输入/输出) 脚本 0 和 1 的主输出目录
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls2"
    
    # (输入) 原始图像
    ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plfls68-69,73-75/images/survey"
    INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plfls68-69,73-75/AT/BlocksExchangeUndistortAT.xml"
    JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/plfls68-69,73-75.json"
    
    # (输入) 'points_lidar_list.pkl' 所在的目录 (与 DATASET_PATH 相同)
    LAS_OUTPUT_PATH="$DATASET_PATH"
    
    # (输出) Debug 图像的保存位置
    DEBUG_OUTPUT_PATH="$DATASET_PATH/debug_projection"

    # --- SBMU 畸变参数 (与脚本0一致) ---

    FX="3691.140"
    FY="3691.140"
    CX="2755.610"
    CY="1796.540"
    K1="0.00191560"
    K2="-0.00834335"
    P1="-0.00270065"
    P2="0.00137660"
    K3="0.00673703"
    
    
    # --- 脚本控制参数 ---
    DEBUG_FLAG="False"  # 设置为 "True" 以生成调试图像，而不是 .npy
    DOWN_SCALE="4"   # 深度图的降采样比例 (1.0 = 不降采样)
    START_FRAME="0"
    END_FRAME="99999"  # 处理所有帧

    # --- 执行 SBMU 任务 ---
    # 注意: 移除了 --num_val
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --original_images_path "$ORIGINAL_IMAGES_PATH" \
        --infos_path "$INFOS_PATH" \
        --original_images_list_json_path "$JSON_PATH" \
        --las_output_path "$LAS_OUTPUT_PATH" \
        --output_path "$DEBUG_OUTPUT_PATH" \
        --debug "$DEBUG_FLAG" \
        --down_scale "$DOWN_SCALE" \
        --start "$START_FRAME" \
        --end "$END_FRAME" \
        --fx $FX --fy $FY --cx $CX --cy $CY \
        --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

    echo "=== 任务 lfls2 完成 ==="
}
# # ==============================================================================
# # 任务 2: SZTTI 数据集
# # ==============================================================================
# run_task_SZTTI() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始点云投影任务: SZTTI ==="
#     echo "================================================="

#     # --- SZTTI 任务参数 ---
#     DATASET_PATH="F:/download/SMBU2/Aerial_lifting_early/dji/process_lidar/output_SZTTI"
#     ORIGINAL_IMAGES_PATH="H:/BaiduNetdiskDownload/LiDAR_Raw/SZIIT/PSZIIT018/images/survey"
#     INFOS_PATH="H:/BaiduNetdiskDownload/LiDAR_Raw/SZIIT/PSZIIT018/AT/BlocksExchangeUndistortAT.xml"
#     JSON_PATH="H:/BaiduNetdiskDownload/LiDAR_Raw/SZIIT/PSZIIT018/images/survey/image_list.json"
#     LAS_OUTPUT_PATH="$DATASET_PATH"
#     DEBUG_OUTPUT_PATH="$DATASET_PATH/debug_projection"

#     # --- SZTTI 畸变参数 (与脚本0一致) ---
#     FX="3692.240"
#     FY="3691.094"
#     CX="2755.094"
#     CY="1796.394"
#     K1="0.001630065"
#     K2="-0.007661543"
#     P1="-0.002702158"
#     P2="0.001345463"
#     K3="0.006244946"

#     # --- 脚本控制参数 ---
#     DEBUG_FLAG="False"
#     DOWN_SCALE="1.0"
#     START_FRAME="0"
#     END_FRAME="99999"

#     # --- 执行 SZTTI 任务 ---
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH" \
#         --original_images_path "$ORIGINAL_IMAGES_PATH" \
#         --infos_path "$INFOS_PATH" \
#         --original_images_list_json_path "$JSON_PATH" \
#         --las_output_path "$LAS_OUTPUT_PATH" \
#         --output_path "$DEBUG_OUTPUT_PATH" \
#         --debug "$DEBUG_FLAG" \
#         --down_scale "$DOWN_SCALE" \
#         --start "$START_FRAME" \
#         --end "$END_FRAME" \
#         --fx $FX --fy $FY --cx $CX --cy $CY \
#         --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

#     echo "=== 任务 SZTTI 完成 ==="
# }


# ==============================================================================
# --- 主执行流程 ---
# 脚本将按顺序执行此处列出的任务
# ==============================================================================

echo "======= 批量点云投影任务启动 ======="

# run_task_SMBU
run_task_lfls
# run_task_lfls2

echo "======= 批量点云投影任务完成 =====
echo ""
echo "======= 所有批量处理任务已成功完成 ======="