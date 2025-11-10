import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing
import argparse
import cv2

def downsample_image(img_array, factor):
    """
    使用 OpenCV 的 INTER_AREA 算法按因子下采样图片。
    """
    h, w = img_array.shape[:2]
    # 计算新的目标尺寸 (宽度, 高度)
    target_wh = (w // factor, h // factor)
    
    # 确保目标尺寸至少为 1x1 像素
    target_wh = (max(1, target_wh[0]), max(1, target_wh[1]))
    
    return cv2.resize(img_array, target_wh, interpolation=cv2.INTER_AREA)

def process_single_file(task_args):
    """
    工作函数：加载一个图片，执行下采样，并保存结果。
    """
    # *** 修改 ***：任务参数简化
    input_path, output_path, factor = task_args
    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # --- 1. 加载数据 ---
        # 使用Pillow加载，然后转为numpy
        source_array = np.array(Image.open(input_path))
        
        # (可选) 确保是 3 通道 RGB
        if source_array.ndim == 2: # 灰度图
            source_array = cv2.cvtColor(source_array, cv2.COLOR_GRAY2RGB)
        elif source_array.shape[2] == 4: # RGBA
            source_array = cv2.cvtColor(source_array, cv2.COLOR_RGBA2RGB)
            
        # --- 2. 执行下采样 ---
        final_array = downsample_image(source_array, factor)

        # --- 3. 保存结果 ---
        Image.fromarray(final_array).save(output_path)
            
        return None
    except Exception as e:
        return f"处理文件 '{input_path}' 时发生错误: {e}"

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="按比例下采样图像。")
    parser.add_argument('--image_folder', type=str, required=True, 
                        help='包含原始图片的文件夹路径。')
    parser.add_argument('--output_folder', type=str, required=True, 
                        help='存放所有下采样结果的输出根文件夹。')
    
    # *** 保留 ***：下采样因子
    parser.add_argument('--downsample_factor', type=int, default=4, 
                        help='下采样因子（例如：4 表示 4倍下采样，宽高各除以4）。')
                        
    # *** 移除 ***：'--npy_folder' 参数
    # parser.add_argument('--npy_folder', type=str, required=True, ...)
                        
    parser.add_argument('--cores', type=int, help='要使用的CPU核心数。')
    args = parser.parse_args()
    
    # 示例命令:
    # python downsample_images_only.py --image_folder .../hav/rgbs \
    #                                 --output_folder .../GT/images_gt_hav \
    #                                 --downsample_factor 4 \
    #                                 --cores 16
    
    # --- 收集所有待处理的任务 ---
    tasks = []
    
    # *** 修改 ***：输出目录简化
    image_out_dir = os.path.join(args.output_folder, "images_downsampled")
    
    # *** 移除 ***：'npy_out_dir'
    # npy_out_dir = os.path.join(args.output_folder, "npy_downsampled")
    
    image_files = [f for f in os.listdir(args.image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"在 {args.image_folder} 中找到 {len(image_files)} 个图片文件，准备处理...")

    for image_filename in image_files:
        image_path = os.path.join(args.image_folder, image_filename)
        out_image_path = os.path.join(image_out_dir, image_filename)
        
        # *** 修改 ***：只添加图片任务，并使用简化的任务参数
        tasks.append((image_path, out_image_path, args.downsample_factor))
        
        # *** 移除 ***：所有 NPY 相关的路径检查和任务添加
        # npy_path = ...
        # if os.path.exists(npy_path):
        #   ...

    if not tasks:
        print("未找到任何可处理的图像文件。")
    else:
        total_cores = multiprocessing.cpu_count()
        num_processes = min(args.cores, total_cores) if args.cores and args.cores > 0 else total_cores
        
        print(f"找到 {len(tasks)} 个有效文件, 将使用 {num_processes}/{total_cores} 个CPU核心进行 {args.downsample_factor}x 下采样...")
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks), desc="总进度"))
        
        errors = [r for r in results if r is not None]
        if errors:
            print("\n处理过程中出现以下错误:")
            for error in errors:
                print(error)

    print("图像下采样处理完成。")
    '''

python downsample_data_hav.py \
    --image_folder /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/output/finished/lfls2/rgbs \
    --output_folder /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GT/images_gt_lfls2 \
    --downsample_factor 4 \
    --cores 32
    '''

