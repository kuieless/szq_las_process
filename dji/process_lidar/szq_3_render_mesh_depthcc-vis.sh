#!/bin/bash

# --- 脚本设置 ---
# set -e: 任何命令失败（返回非0退出码），脚本将立即退出。
# 这对于防止在失败的任务上继续执行后续任务至关重要。
set -e

# --- 通用变量 ---
# 将你的 Python 脚本的路径定义在这里
# 假设你的 Python 脚本与这个 bash 脚本在同一个目录
SCRIPT_DIR=$(dirname "$0") # 获取当前脚本所在的目录
PYTHON_SCRIPT_NAME="szq_3_render_mesh_depthcc-vis.py" # 你保存的 Python 脚本名称
PYTHON_SCRIPT_PATH="${SCRIPT_DIR}/${PYTHON_SCRIPT_NAME}"
# ORIGIN="409301.23037930636,3950413.1119947666,151.99999999947258"
# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "错误: 未找到 Python 脚本: $PYTHON_SCRIPT_PATH"
    exit 1
fi

echo "将使用 Python 脚本: $PYTHON_SCRIPT_PATH"

# ==============================================================================
# 任务 1: Xiayuan (示例任务 1)
# ==============================================================================
run_task_lfls2() {
  echo ""
  echo "================================================="
  echo "🚀 开始执行任务: lfls2)"
  echo "================================================="

  # --- 此任务的特定参数 ---
  # 使用 'local' 关键字确保变量只在函数内部有效
  local base_dir="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES"
  local obj_dir="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES"
  local data_dir="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/"
  local xml_path="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/BlocksExchangeUndistortAT.xml"
  local save_dir="/home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/" # 结果保存在数据目录中
  local down_factor=1.0

  # 执行 Python 脚本
  python "$PYTHON_SCRIPT_PATH" \
    --obj_dir "$obj_dir" \
    --nerf_metadata_dir "$data_dir" \
    --metadataXml_path "$xml_path" \
    --save_dir "$save_dir" \
    --down $down_factor \
    --visualize \
    --save_mesh # 比如这个任务我们想保存组合网格

  echo "✅ 任务 'lfls2行完毕。"
}



# ==============================================================================
# --- 任务执行顺序 ---
# 在这里按顺序调用你想要运行的函数
# 脚本将逐个执行它们
# ==============================================================================
main() {
  echo "🔥 开始批量处理任务..."
  
  # 逐次运行
  run_task_lfls2
  
  # run_task_another_area # 如果想暂时跳过这个任务，只需在行首添加'#'注释掉
  
  echo ""
  echo "🎉🎉🎉 所有任务已成功完成! 🎉🎉🎉"
}

# 运行主函数
main