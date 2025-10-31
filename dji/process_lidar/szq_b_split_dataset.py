import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# 忽略 scikit-learn 关于 n_init 的未来警告
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.cluster._kmeans')
# 忽略 matplotlib 可能的中文显示警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def dms_to_decimal(dms_tuple, ref):
    """
    将 GPS 的 (度, 分, 秒) 元组转换为高精度十进制度数。
    此版本已修复，可以正确处理 IFDRational 对象。
    """
    try:
        # dms_tuple 的每个元素可能是 float，也可能是 IFDRational 对象
        
        def convert_value(val):
            # 检查它是否是 IFDRational 对象 (有 .numerator 属性)
            if hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                if val.denominator == 0:
                    return 0.0
                return float(val.numerator) / float(val.denominator)
            # 检查它是否是普通的 int 或 float
            if isinstance(val, (int, float)):
                return float(val)
            # 检查它是否是 (分子, 分母) 元组
            if isinstance(val, tuple) and len(val) == 2:
                if val[1] == 0:
                    return 0.0
                return float(val[0]) / float(val[1])
            # 如果都不是，尝试最后强制转换
            return float(val)

        degrees = convert_value(dms_tuple[0])
        minutes = convert_value(dms_tuple[1])
        seconds = convert_value(dms_tuple[2])
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        if ref in ['S', 'W']:
            decimal = -decimal
            
        return decimal
        
    except Exception as e:
        # print(f"DMS 转换错误: {e}, 数据: {dms_tuple}, ref: {ref}") # 取消注释以进行调试
        return None

def get_gps_coordinates(image_path):
    """
    从图片文件中提取高精度的 (纬度, 经度)。
    """
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()

        if not exif_data or 34853 not in exif_data:
            return None, None
            
        gps_info = {}
        for tag, value in exif_data[34853].items():
            tag_name = GPSTAGS.get(tag, tag)
            gps_info[tag_name] = value

        lat_dms = gps_info.get('GPSLatitude')
        lat_ref = gps_info.get('GPSLatitudeRef')
        lon_dms = gps_info.get('GPSLongitude')
        lon_ref = gps_info.get('GPSLongitudeRef')

        if lat_dms and lat_ref and lon_dms and lon_ref:
            latitude = dms_to_decimal(lat_dms, lat_ref)
            longitude = dms_to_decimal(lon_dms, lon_ref)
            
            # 只有在两个坐标都成功解析时才返回
            if latitude is not None and longitude is not None:
                return latitude, longitude
            
    except Exception as e:
        # print(f"读取 EXIF 失败 {image_path}: {e}") # 取消注释以进行调试
        pass
        
    return None, None

def plot_coordinates(df, k_clusters, kmeans_model, output_dir):
    """
    生成并保存三张可视化图表。
    """
    print("\n[Visualizing] 正在生成可视化图表...")
    
    # --- 图 1: 原始坐标分布 ---
    plt.figure(figsize=(12, 10))
    plt.scatter(df['longitude'], df['latitude'], s=5, alpha=0.5)
    plt.title('相机坐标地理分布', fontsize=16)
    plt.xlabel('Longitude (经度)', fontsize=12)
    plt.ylabel('Latitude (纬度)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal') # 保持经纬度比例尺一致
    plot_path_1 = os.path.join(output_dir, 'coordinates_distribution.png')
    plt.savefig(plot_path_1)
    print(f"坐标分布图已保存到: {plot_path_1}")
    plt.close()

    # --- 图 2: K-Means 聚类结果 ---
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(df['longitude'], 
                          df['latitude'], 
                          s=5, 
                          c=df['cluster'], 
                          cmap='tab20',
                          alpha=0.8)
    
    centers = kmeans_model.cluster_centers_
    plt.scatter(centers[:, 1], centers[:, 0], c='red', s=150, marker='X', label='聚类中心 (Cluster Centers)')
    
    plt.title(f'K-Means 地理聚类结果 (K={k_clusters})', fontsize=16)
    plt.xlabel('Longitude (经度)', fontsize=12)
    plt.ylabel('Latitude (纬度)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    
    if k_clusters <= 20:
        cbar = plt.colorbar(scatter, ticks=range(k_clusters))
        cbar.set_label('Cluster ID', fontsize=12)
    else:
        plt.legend()
        
    plot_path_2 = os.path.join(output_dir, 'kmeans_clusters.png')
    plt.savefig(plot_path_2)
    print(f"K-Means 聚类图已保存到: {plot_path_2}")
    plt.close()

    # --- 图 3: 训练/测试集划分 ---
    plt.figure(figsize=(12, 10))
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    plt.scatter(train_df['longitude'], train_df['latitude'], 
                s=5, c='blue', alpha=0.6, label=f'Train (训练集 - {len(train_df)} 张)')
    plt.scatter(test_df['longitude'], test_df['latitude'], 
                s=5, c='red', alpha=0.8, label=f'Test (测试集 - {len(test_df)} 张)')
    
    plt.title('训练集 / 测试集 地理划分结果', fontsize=16)
    plt.xlabel('Longitude (经度)', fontsize=12)
    plt.ylabel('Latitude (纬度)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.legend(markerscale=3, fontsize=12)
    
    plot_path_3 = os.path.join(output_dir, 'train_test_split.png')
    plt.savefig(plot_path_3)
    print(f"训练/测试集划分图已保存到: {plot_path_3}")
    plt.close()


def main(args):
    print(f"--- 地理空间数据集划分工具 (V2.1 - 已修复) ---")
    print(f"1. 正在扫描目录: {args.image_dir}")
    
    image_extensions = ('.jpg', '.jpeg', '.tif', '.tiff', '.png')
    metadata_list = []
    
    all_files = []
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(image_extensions):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("错误：在指定目录中未找到任何图片文件。")
        return

    print(f"总共找到 {len(all_files)} 张图片。正在提取 GPS 元数据...")

    skipped_files_count = 0
    skipped_warning_shown = False

    for filepath in tqdm(all_files, desc="提取元数据"):
        lat, lon = get_gps_coordinates(filepath)
        if lat is not None and lon is not None:
            metadata_list.append({
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'latitude': lat,
                'longitude': lon
            })
        else:
            skipped_files_count += 1
            if not skipped_warning_shown:
                # print(f"\n[注意] 至少有一个文件无法读取 GPS: {filepath}") # 取消注释以进行调试
                skipped_warning_shown = True


    if not metadata_list:
        print(f"\n错误：在所有 {len(all_files)} 张图片中均未找到可解析的 GPS 元数据。无法进行划分。")
        print("请使用 'check_exif.py' 脚本检查您的单张图片。")
        return

    n_found = len(metadata_list)
    print(f"成功提取了 {n_found} 张图片的 GPS 坐标。")
    if skipped_files_count > 0:
        print(f"跳过了 {skipped_files_count} 张没有 GPS 数据或解析失败的图片。")

    df = pd.DataFrame(metadata_list)
    coordinates = df[['latitude', 'longitude']].values

    # --- K-Means 聚类 ---
    k_clusters = min(args.k, n_found)
    if k_clusters < 2:
        print(f"错误：有效的 GPS 图片数量 ({n_found}) 太少，无法进行至少 2 个簇的聚类。")
        return
        
    print(f"\n2. 正在使用 K-Means (K={k_clusters}) 对 {n_found} 个坐标点进行地理聚类...")
    
    kmeans = KMeans(n_clusters=k_clusters, random_state=args.random_seed, n_init=10)
    df['cluster'] = kmeans.fit_predict(coordinates)

    # --- 按簇划分数据集 ---
    cluster_ids = np.unique(df['cluster'])
    n_test_clusters = max(1, int(round(k_clusters * args.test_size)))
    
    if k_clusters - n_test_clusters < 1:
        if k_clusters > 1:
            n_test_clusters = k_clusters - 1
        else:
            print("错误：只有一个地理簇，无法划分为训练集和测试集。")
            return

    print(f"3. 正在将 {k_clusters} 个簇划分为训练集/测试集 (测试集簇数: {n_test_clusters})...")
    
    train_cluster_ids, test_cluster_ids = train_test_split(
        cluster_ids, 
        test_size=n_test_clusters, 
        random_state=args.random_seed
    )

    test_cluster_set = set(test_cluster_ids)
    df['split'] = df['cluster'].apply(lambda c: 'test' if c in test_cluster_set else 'train')

    # --- 4. 生成可视化图表 ---
    output_dir = os.path.dirname(args.output_csv)
    if not output_dir:
        output_dir = "." 
    
    plot_coordinates(df, k_clusters, kmeans, output_dir)

    # --- 5. 保存 CSV 结果 ---
    df.to_csv(args.output_csv, index=False, float_format='%.9f')

    print(f"\n--- 划分完成 ---")
    print(f"训练集簇: {list(train_cluster_ids)}")
    print(f"测试集簇: {list(test_cluster_ids)}")
    print("-" * 30)
    print(f"训练集图片数量: {len(df[df['split'] == 'train'])}")
    print(f"测试集图片数量:   {len(df[df['split'] == 'test'])}")
    print("-" * 30)
    print(f"详细划分结果已保存到: {args.output_csv}")
    print(f"可视化图表已保存在: {os.path.abspath(output_dir)} 目录中")
    
    print("\nCSV 结果预览 (前 5 行):")
    print(df.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据 GPS 元数据将航拍图片划分为训练集和测试集，并生成可视化图表。")
    
    parser.add_argument("--image_dir", 
                        type=str, 
                        default=".", 
                        help="包含航拍图片的目录路径 (会递归搜索)。")
                        
    parser.add_argument("--k", 
                        type=int, 
                        default=10, 
                        help="K-Means 聚类的簇数 (K值)。")
                        
    parser.add_argument("--test_size", 
                        type=float, 
                        default=0.2, 
                        help="测试集所占的簇的比例 (例如 0.2 表示 20%%)。")
                        
    parser.add_argument("--output_csv", 
                        type=str, 
                        default="geospatial_split.csv", 
                        help="输出的 CSV 文件名。图表将保存在此文件的同一目录中。")
                        
    parser.add_argument("--random_seed", 
                        type=int, 
                        default=42, 
                        help="随机种子，确保每次划分结果一致。")

    args = parser.parse_args()
    main(args)

    #python szq_b_split_dataset.py --image_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/pSZTU/images/survey --k 10
 #python szq_split_dataset.py --image_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/raw_pics/ed/phav-all/images/survey --k 10
    