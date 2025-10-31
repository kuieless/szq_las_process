# #!/bin/bash

# # ==============================================================================
# # 批量处理脚本 (用于 2_project_lidar_to_depth.py)
# #
# # 功能:
# # 1. 定义多个独立的数据集任务 (SBMU, SZTTI)。
# # 2. 逐次执行每个任务。
# # 3. 使用 'set -e' 确保任何任务失败时，脚本会立即停止。
# # ==============================================================================

# set -e

# # --- 配置 ---
# # 请将此名称修改为您保存的 Python 脚本的名称
# PYTHON_SCRIPT_NAME="szq_2_convert_lidar_2_depth_color.py"




# run_task_lower() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始点云投影任务: lower ==="
#     echo "================================================="

#     # --- SBMU 任务参数 ---
    
#     # (输入/输出) 脚本 0 和 1 的主输出目录
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lower"
    
#     # (输入) 原始图像
#     ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plower/images/survey"
#     INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plower/AT/BlocksExchangeUndistortAT.xml"
#     JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/plower.json"
    
#     # (输入) 'points_lidar_list.pkl' 所在的目录 (与 DATASET_PATH 相同)
#     LAS_OUTPUT_PATH="$DATASET_PATH"
    
#     # (输出) Debug 图像的保存位置
#     DEBUG_OUTPUT_PATH="$DATASET_PATH/debug_projection"

#     # --- SBMU 畸变参数 (与脚本0一致) ---

#     FX="3690.36"
#     FY="3690.36"
#     CX="2755.45"
#     CY="1796.28"
#     K1=" 0.00178198"
#     K2="-0.00794891"
#     P1="-0.00276489"
#     P2="0.00136293"
#     K3="0.00618299"
    
    
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

#     echo "=== 任务 lower 完成 ==="
# }


# run_task_sziit() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始点云投影任务: sziit ==="
#     echo "================================================="

#     # --- SBMU 任务参数 ---
    
#     # (输入/输出) 脚本 0 和 1 的主输出目录
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sziit"
    
#     # (输入) 原始图像
#     ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/psziit-all/images/survey"
#     INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/psziit-all/AT/BlocksExchangeUndistortAT.xml"
#     JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/psziit-all.json"
    
#     # (输入) 'points_lidar_list.pkl' 所在的目录 (与 DATASET_PATH 相同)
#     LAS_OUTPUT_PATH="$DATASET_PATH"
    
#     # (输出) Debug 图像的保存位置
#     DEBUG_OUTPUT_PATH="$DATASET_PATH/debug_projection"

#     # --- SBMU 畸变参数 (与脚本0一致) ---

#     FX="3691.83"
#     FY="3691.83"
#     CX="2755.75"
#     CY="1795.83"
#     K1=" 0.00174818"
#     K2="-0.00767000"
#     P1="-0.00271498"
#     P2="0.00138925"
#     K3="0.00599998"
    
    
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

#     echo "=== 任务 sziit 完成 ==="
# }
# run_task_sztu() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始点云投影任务: sztu ==="
#     echo "================================================="

#     # --- SBMU 任务参数 ---
    
#     # (输入/输出) 脚本 0 和 1 的主输出目录
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu"
    
#     # (输入) 原始图像
#     ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSZTU/images/survey"
#     INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSZTU/AT/BlocksExchangeUndistortAT.xml"
#     JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pSZTU-all.json"
    
#     # (输入) 'points_lidar_list.pkl' 所在的目录 (与 DATASET_PATH 相同)
#     LAS_OUTPUT_PATH="$DATASET_PATH"
    
#     # (输出) Debug 图像的保存位置
#     DEBUG_OUTPUT_PATH="$DATASET_PATH/debug_projection"

#     # --- SBMU 畸变参数 (与脚本0一致) ---

#     FX="3692.76"
#     FY="3692.76"
#     CX="2755.71"
#     CY="1796.80"
#     K1=" 0.00195146"
#     K2="-0.00801190"
#     P1="-0.00265830"
#     P2="0.00140022"
#     K3="0.00632481"
    
    
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

#     echo "=== 任务 sztu 完成 ==="
# }
# # ==============================================================================
# # --- 主执行流程 ---
# # 脚本将按顺序执行此处列出的任务
# # ==============================================================================

# echo "======= 批量点云投影任务启动 ======="

# # # run_task_SMBU
# # # run_task_lfls
# # run_task_lfls2

# run_task_lower
# run_task_sziit
# run_task_sztu
# echo "======= 批量点云投影任务完成 =====
# echo ""
# echo "======= 所有批量处理任务已成功完成 ======="



# # ==============================================================================































#!/bin/bash

# ==============================================================================
# 批量处理脚本 (用于 2_project_lidar_to_depth.py) (多核版)
#
# 功能:
# 1. 定义多个独立的数据集任务。
# 2. 逐次执行每个任务。
# 3. 'set -e' 确保任何任务失败时，脚本会立即停止。
# 4. 使用 -j 参数为 Python 脚本指定 CPU 核心数以加速加载。
# ==============================================================================

# 确保脚本因错误而退出
set -e

# --- 配置 ---

# 请将此名称修改为您保存的 Python 脚本的名称
PYTHON_SCRIPT_NAME="szq_2_convert_lidar_2_depth_color.py"

# (!!!) (性能调优) (!!!)
# 定义要用于数据加载的 CPU 核心数。
# 推荐值: 8, 16, 或 32，取决于您的 CPU。
NUM_WORKERS="32"



# # 任务 3: sztu
# # ==============================================================================
run_task_sztu() {
    echo ""
    echo "================================================="
    echo "=== 正在开始点云投影任务: sztu (使用 ${NUM_WORKERS} 核心) ==="
    echo "================================================="

    # --- 任务参数 ---
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu"
    ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSZTU/images/survey"
    INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSZTU/AT/BlocksExchangeUndistortAT.xml"
    JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pSZTU.json"
    LAS_OUTPUT_PATH="$DATASET_PATH"
    DEBUG_OUTPUT_PATH="$DATASET_PATH/debug_projection"

    # --- 畸变参数 ---
    FX="3692.76"
    FY="3692.76"
    CX="2755.71"
    CY="1796.80"
    K1="0.00195146"
    K2="-0.00801190"
    P1="-0.00265830"
    P2="0.00140022"
    K3="0.00632481"
    
    # --- 脚本控制 ---
    DEBUG_FLAG="False"
    DOWN_SCALE="4"
    START_FRAME="0"
    END_FRAME="99999"

    # --- 执行任务 ---
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
        --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3 \
        -j "$NUM_WORKERS"

    echo "=== 任务 sztu 完成 ==="
}
# # 任务 4: upper
# # ==============================================================================
# run_task_upper() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始点云投影任务: upper (使用 ${NUM_WORKERS} 核心) ==="
#     echo "================================================="

#     # --- 任务参数 ---
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/upper"
#     ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pupper/images/survey"
#     INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pupper/AT/BlocksExchangeUndistortAT.xml"
#     JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pupper.json"
#     LAS_OUTPUT_PATH="$DATASET_PATH"
#     DEBUG_OUTPUT_PATH="$DATASET_PATH/debug_projection"

#     # --- 畸变参数 ---
#     FX="3690.39"
#     FY="3690.39"
#     CX="2755.77"
#     CY="1796.08"
#     K1="0.00176217"
#     K2="-0.00844469"
#     P1="-0.00275764"
#     P2=" 0.00137883"
#     K3="0.00676545"
    
#     # --- 脚本控制 ---
#     DEBUG_FLAG="False"
#     DOWN_SCALE="4"
#     START_FRAME="0"
#     END_FRAME="99999"

#     # --- 执行任务 ---
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
#         --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3 \
#         -j "$NUM_WORKERS"

#     echo "=== 任务 sztu 完成 ==="
# }
# ==============================================================================
# --- 主执行流程 ---
# 脚本将按顺序执行此处列出的任务
# ==============================================================================

echo "======= 批量点云投影任务启动 (多核版) ======="


run_task_sztu
# run_task_upper
echo ""
echo "======= 所有批量处理任务已成功完成 ======="