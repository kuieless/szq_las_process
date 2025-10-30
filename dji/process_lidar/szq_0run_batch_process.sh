#!/bin/bash

# ==============================================================================
# 批量处理脚本
#
# 功能:
# 1. 定义多个独立的处理任务。
# 2. 逐次执行每个任务。
# 3. 使用 'set -e' 确保任何任务失败时，脚本会立即停止。
# ==============================================================================

set -e

# --- 配置 ---
# 请将此名称修改为您保存的 Python 脚本的名称
PYTHON_SCRIPT_NAME="szq_0_process_dji_v8_color_no_split.py"


# ==============================================================================
# 任务 1: SBMU 数据集
# ==============================================================================
# run_task_SMBU() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始任务: SBMU ==="
#     echo "================================================="

#     # --- SBMU 任务参数 ---
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/SMBU"
#     ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSMBU-all/images/survey"
#     INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSMBU-all/AT/BlocksExchangeUndistortAT.xml"
#     JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pSMBU-all.json"
    
#     FX="3691.520"
#     FY="3691.520"
#     CX="2755.450"
#     CY="1796.980"
#     K1=" 0.00185019"
#     K2="-0.00826045"
#     P1="-0.00265205"
#     P2="0.00138173"
#     K3="0.00652772"

#     # --- 执行 SBMU 任务 ---
#     # 注意：已移除 --num_val 参数，因为它在修改后的脚本中不再需要
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH" \
#         --original_images_path "$ORIGINAL_IMAGES_PATH" \
#         --infos_path "$INFOS_PATH" \
#         --original_images_list_json_path "$JSON_PATH" \
#         --fx $FX --fy $FY --cx $CX --cy $CY \
#         --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

#     echo "=== 任务 SBMU 完成 ==="
# }

# # ==============================================================================
# run_task_lfls() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始任务: SBMU ==="
#     echo "================================================="

#     # --- SBMU 任务参数 ---
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls"
#     ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plfls45-47,65-67,70-72/images/survey"
#     INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plfls45-47,65-67,70-72/AT/BlocksExchangeUndistortAT.xml"
#     JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/plfls45-47,65-67,70-72.json"
    
#     FX="3691.510"
#     FY="3691.510"
#     CX="2755.710"
#     CY="1796.570"
#     K1=" 0.00184453"
#     K2="-0.00808798"
#     P1="-0.00267461"
#     P2="0.00137382"
#     K3="0.00642050"

#     # --- 执行 SBMU 任务 ---
#     # 注意：已移除 --num_val 参数，因为它在修改后的脚本中不再需要
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH" \
#         --original_images_path "$ORIGINAL_IMAGES_PATH" \
#         --infos_path "$INFOS_PATH" \
#         --original_images_list_json_path "$JSON_PATH" \
#         --fx $FX --fy $FY --cx $CX --cy $CY \
#         --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

#     echo "=== 任务 SBMU 完成 ==="
# }
# run_task_lfls2() {
#     echo ""
#     echo "================================================="
#     echo "=== 正在开始任务: SBMU ==="
#     echo "================================================="

#     # --- SBMU 任务参数 ---
#     DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lfls2"
#     ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plfls68-69,73-75/images/survey"
#     INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plfls68-69,73-75/AT/BlocksExchangeUndistortAT.xml"
#     JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/plfls68-69,73-75.json"
    
#     FX="3691.140"
#     FY="3691.140"
#     CX="2755.610"
#     CY="1796.540"
#     K1="0.00191560"
#     K2="-0.00834335"
#     P1="-0.00270065"
#     P2="0.00137660"
#     K3="0.00673703"

#     # --- 执行 SBMU 任务 ---
#     # 注意：已移除 --num_val 参数，因为它在修改后的脚本中不再需要
#     python $PYTHON_SCRIPT_NAME \
#         --dataset_path "$DATASET_PATH" \
#         --original_images_path "$ORIGINAL_IMAGES_PATH" \
#         --infos_path "$INFOS_PATH" \
#         --original_images_list_json_path "$JSON_PATH" \
#         --fx $FX --fy $FY --cx $CX --cy $CY \
#         --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

#     echo "=== 任务 SBMU 完成 ==="
# }



run_task_upper() {
    echo ""
    echo "================================================="
    echo "=== 正在开始任务: upper ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/upper"
    ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pupper/images/survey"
    INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pupper/AT/BlocksExchangeUndistortAT.xml"
    JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pupper.json"
    
    FX="3690.390"
    FY="3690.390"
    CX="2755.77"
    CY="1796.08"
    K1="0.00176217"
    K2="-0.00844469"
    P1="-0.00275764"
    P2="0.00137883"
    K3="0.00676545"

    # --- 执行 SBMU 任务 ---
    # 注意：已移除 --num_val 参数，因为它在修改后的脚本中不再需要
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --original_images_path "$ORIGINAL_IMAGES_PATH" \
        --infos_path "$INFOS_PATH" \
        --original_images_list_json_path "$JSON_PATH" \
        --fx $FX --fy $FY --cx $CX --cy $CY \
        --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

    echo "=== 任务 upper 完成 ==="
}
run_task_lower() {
    echo ""
    echo "================================================="
    echo "=== 正在开始任务: upper ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/lower"
    ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plower/images/survey"
    INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/plower/AT/BlocksExchangeUndistortAT.xml"
    JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/plower.json"
    
    FX="3690.360"
    FY="3690.360"
    CX="2755.45"
    CY="1796.28"
    K1="0.00178198"
    K2="-0.00794891"
    P1="-0.00276489"
    P2="0.00136293"
    K3="0.00618299"

    # --- 执行 SBMU 任务 ---
    # 注意：已移除 --num_val 参数，因为它在修改后的脚本中不再需要
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --original_images_path "$ORIGINAL_IMAGES_PATH" \
        --infos_path "$INFOS_PATH" \
        --original_images_list_json_path "$JSON_PATH" \
        --fx $FX --fy $FY --cx $CX --cy $CY \
        --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

    echo "=== 任务 lower 完成 ==="
}
# 
run_task_sziit() {
    echo ""
    echo "================================================="
    echo "=== 正在开始任务: sziit ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sziit"
    ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/psziit-all/images/survey"
    INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/psziit-all/AT/BlocksExchangeUndistortAT.xml"
    JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/psziit-all.json"
    
    FX="3691.83"
    FY="3691.83"
    CX="2755.75"
    CY=" 1795.83"
    K1="0.00174818"
    K2="-0.00767000"
    P1="-0.00271498"
    P2="0.00138925"
    K3=" 0.00599998"

    # --- 执行 SBMU 任务 ---
    # 注意：已移除 --num_val 参数，因为它在修改后的脚本中不再需要
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --original_images_path "$ORIGINAL_IMAGES_PATH" \
        --infos_path "$INFOS_PATH" \
        --original_images_list_json_path "$JSON_PATH" \
        --fx $FX --fy $FY --cx $CX --cy $CY \
        --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

    echo "=== 任务 sziit 完成 ==="
}
run_task_SZTU() {
    echo ""
    echo "================================================="
    echo "=== 正在开始任务: sztu ==="
    echo "================================================="

    # --- SBMU 任务参数 ---
    DATASET_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/sztu"
    ORIGINAL_IMAGES_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSZTU/images/survey"
    INFOS_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSZTU/AT/BlocksExchangeUndistortAT.xml"
    JSON_PATH="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/jsons/pSZTU.json"
    
    FX="3692.76"
    FY="3692.76"
    CX="2755.71"
    CY=" 1796.80"
    K1="0.00195146"
    K2="-0.00801190"
    P1="-0.0026583"
    P2="0.00140022"
    K3=" 0.00632481"

    # --- 执行 SBMU 任务 ---
    # 注意：已移除 --num_val 参数，因为它在修改后的脚本中不再需要
    python $PYTHON_SCRIPT_NAME \
        --dataset_path "$DATASET_PATH" \
        --original_images_path "$ORIGINAL_IMAGES_PATH" \
        --infos_path "$INFOS_PATH" \
        --original_images_list_json_path "$JSON_PATH" \
        --fx $FX --fy $FY --cx $CX --cy $CY \
        --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3

    echo "=== 任务 sztu 完成 ==="
}




# ==============================================================================
# --- 主执行流程 ---
# 脚本将按顺序执行此处列出的任务
# ==============================================================================

echo "======= 批量处理任务启动 ======="

# run_task_SMBU
# run_task_lfls
# run_task_lfls2
run_task_upper
run_task_lower
run_task_sziit
run_task_SZTU


echo ""
echo "======= 所有批量处理任务已成功完成 ======="

