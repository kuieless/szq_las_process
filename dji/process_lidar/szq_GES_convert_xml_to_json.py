# import xml.etree.ElementTree as ET
# import numpy as np
# import json
# import os
# import argparse
# from tqdm import tqdm
# from scipy.spatial.transform import Rotation as R

# # =======================================================================================
# # ============= 还原到“原始”的 convert_opk_to_c2w 函数（无修正）================
# # =======================================================================================
# def convert_opk_to_c2w(omega_deg, phi_deg, kappa_deg, C_local):
#     """
#     将 OPK 角度和【局部】相机中心转换为 4x4 C2W 矩阵。
#     这是直接、无修正的转换。
#     """
    
#     # 1. 获取标准的摄影测量 W2C 旋转 (Z-Y-X)
#     r_w2c_photo = R.from_euler('zyx', [kappa_deg, phi_deg, omega_deg], degrees=True)
#     R_w2c_photo_mat = r_w2c_photo.as_matrix()

#     # 2. 通过转置 R_w2c 得到 C2W 旋转
#     R_c2w_mat = R_w2c_photo_mat.T 
    
#     # 3. 构建 4x4 C2W 矩阵
#     c2w = np.eye(4)
#     c2w[:3, :3] = R_c2w_mat
#     c2w[:3, 3] = C_local  # 使用【局部】中心
#     return c2w

# # =======================================================================================
# # ======================== 以下是 main 函数 (无需更改) ==============================
# # =======================================================================================

# def main(args):
#     print(f"Parsing XML file: {args.input_xml}")
#     tree = ET.parse(args.input_xml)
#     root = tree.getroot()
    
#     os.makedirs(args.output_dir, exist_ok=True)
#     print(f"Output directory created/found: {args.output_dir}")
    
#     photogroup_count = 0
#     photo_count = 0
#     all_centers = []

#     # ==================================================================
#     # STEP 1: 第一次遍历，找到全局原点 (所有相机的平均中心)
#     # ==================================================================
#     print("Step 1: Finding global origin (average camera center)...")
#     for photogroup in root.findall('.//Photogroup'):
#         for photo in photogroup.findall('Photo'):
#             try:
#                 C_x = float(photo.find('Pose/Center/x').text)
#                 C_y = float(photo.find('Pose/Center/y').text)
#                 C_z = float(photo.find('Pose/Center/z').text)
#                 all_centers.append([C_x, C_y, C_z])
#             except Exception as e:
#                 print(f"Warning: Could not parse center for a photo: {e}")

#     if not all_centers:
#         print("Error: No camera centers found in XML. Exiting.")
#         return

#     # 计算平均中心
#     global_origin = np.mean(np.array(all_centers), axis=0)
#     print(f"Global Origin (Average Center) calculated: {global_origin}")

#     # 将这个原点保存到 --origin_file 指定的文件中
#     with open(args.origin_file, 'w') as f:
#         json.dump(global_origin.tolist(), f, indent=4)
#     print(f"Global Origin saved to: {args.origin_file}")

#     # ==================================================================
#     # STEP 2: 第二次遍历，创建本地化的 JSON 文件
#     # ==================================================================
#     print("Step 2: Creating localized camera JSONs...")
#     for photogroup in root.findall('.//Photogroup'):
#         photogroup_count += 1
        
#         # --- 1. 获取该组共享的相机内参 ---
#         try:
#             W = int(photogroup.find('ImageDimensions/Width').text)
#             H = int(photogroup.find('ImageDimensions/Height').text)
#             fx = float(photogroup.find('FocalLengthPixels').text)
#             aspect_ratio_elem = photogroup.find('AspectRatio')
#             aspect_ratio = float(aspect_ratio_elem.text) if aspect_ratio_elem is not None else 1.0
#             fy = fx * aspect_ratio 
#             cx = float(photogroup.find('PrincipalPoint/x').text)
#             cy = float(photogroup.find('PrincipalPoint/y').text)
#             intrinsics_dict = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
#         except Exception as e:
#             print(f"Error parsing intrinsics for Photogroup {photogroup_count}: {e}")
#             continue

#         # --- 2. 处理该组中的所有照片 ---
#         for photo in tqdm(photogroup.findall('Photo')):
#             try:
#                 image_path = photo.find('ImagePath').text
#                 image_path_normalized = image_path.replace('\\', '/')
#                 basename = os.path.basename(image_path_normalized)
#                 filename, _ = os.path.splitext(basename)
                
#                 C_x = float(photo.find('Pose/Center/x').text)
#                 C_y = float(photo.find('Pose/Center/y').text)
#                 C_z = float(photo.find('Pose/Center/z').text)
#                 C_global = np.array([C_x, C_y, C_z])
                
#                 # 通过减去全局原点，将全局中心转换为局部中心
#                 C_local = C_global - global_origin
                
#                 omega_deg = float(photo.find('Pose/Rotation/Omega').text)
#                 phi_deg = float(photo.find('Pose/Rotation/Phi').text)
#                 kappa_deg = float(photo.find('Pose/Rotation/Kappa').text)
                
#                 # 使用【原始、无修正】的函数创建 C2W 矩阵
#                 c2w = convert_opk_to_c2w(omega_deg, phi_deg, kappa_deg, C_local)
                
#                 json_data = {
#                     "H": H,
#                     "W": W,
#                     "c2w": c2w.tolist(),
#                     "intrinsics": intrinsics_dict
#                 }
                
#                 output_path = os.path.join(args.output_dir, f"{filename}.json")
#                 with open(output_path, 'w') as f:
#                     json.dump(json_data, f, indent=4)
                
#                 photo_count += 1
#             except Exception as e:
#                 print(f"Error processing photo {photo.find('Id').text}: {e}")
                
#     print(f"\n--- Conversion Complete ---")
#     print(f"Processed {photogroup_count} photogroups.")
#     print(f"Generated {photo_count} LOCALIZED .json files in {args.output_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert BlocksExchange XML to localized JSON camera files.")
#     parser.add_argument('--input_xml', required=True, help='Path to the BlocksExchangeUndistortAT.xml file.')
#     parser.add_argument('--output_dir', required=True, help='Path to the directory to save the output .json files.')
#     parser.add_argument('--origin_file', required=True, help='Path to save the calculated global origin (e.g., origin.json).')
#     args = parser.parse_args()
#     main(args)


# import xml.etree.ElementTree as ET
# import numpy as np
# import json
# import os
# import argparse
# from tqdm import tqdm
# from scipy.spatial.transform import Rotation as R
# import math # 确保导入 math

# # =======================================================================================
# # ==================== convert_opk_to_c2w 函数 (Z-up) ========================
# # =======================================================================================
# def convert_opk_to_c2w(omega_deg, phi_deg, kappa_deg, C_local_zup):
#     """
#     将 OPK 角度和【Z-up 局部】相机中心转换为【Z-up 4x4 C2W】 矩阵。
#     """
#     # 1. 获取标准的摄影测量 W2C 旋转 (Z-Y-X)
#     r_w2c_photo = R.from_euler('zyx', [kappa_deg, phi_deg, omega_deg], degrees=True)
#     R_w2c_photo_mat = r_w2c_photo.as_matrix()

#     # 2. 通过转置 R_w2c 得到 C2W 旋转
#     R_c2w_mat = R_w2c_photo_mat.T 
    
#     # 3. 构建 4x4 C2W 矩阵
#     c2w_zup = np.eye(4)
#     c2w_zup[:3, :3] = R_c2w_mat
#     c2w_zup[:3, 3] = C_local_zup  # 使用【Z-up 局部】中心
#     return c2w_zup

# # =======================================================================================
# # ======================== 以下是 main 函数 (关键修改) ==============================
# # =======================================================================================

# def main(args):
#     print(f"Parsing XML file: {args.input_xml}")
#     tree = ET.parse(args.input_xml)
#     root = tree.getroot()
    
#     os.makedirs(args.output_dir, exist_ok=True)
#     print(f"Output directory created/found: {args.output_dir}")
    
#     photogroup_count = 0
#     photo_count = 0
#     all_centers = []

#     # ==================================================================
#     # STEP 1: 第一次遍历，找到全局 Z-up 原点
#     # ==================================================================
#     print("Step 1: Finding global origin (average camera center)...")
#     for photogroup in root.findall('.//Photogroup'):
#         for photo in photogroup.findall('Photo'):
#             try:
#                 C_x = float(photo.find('Pose/Center/x').text)
#                 C_y = float(photo.find('Pose/Center/y').text)
#                 C_z = float(photo.find('Pose/Center/z').text)
#                 all_centers.append([C_x, C_y, C_z])
#             except Exception as e:
#                 print(f"Warning: Could not parse center for a photo: {e}")

#     if not all_centers:
#         print("Error: No camera centers found in XML. Exiting.")
#         return

#     # 计算平均中心 (Z-up)
#     global_origin_zup = np.mean(np.array(all_centers), axis=0)
#     print(f"Global Z-up Origin calculated: {global_origin_zup}")

#     # 将这个 Z-up 原点保存到 --origin_file
#     with open(args.origin_file, 'w') as f:
#         json.dump(global_origin_zup.tolist(), f, indent=4)
#     print(f"Global Z-up Origin saved to: {args.origin_file}")

#     # ==================================================================
#     # STEP 2: 定义 Z-up -> Y-up 的 4x4 变换矩阵
#     # ==================================================================
#     angle = -90.0
#     R_z_to_y_np = np.array([
#         [1, 0, 0, 0],
#         [0, math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
#         [0, math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
#         [0, 0, 0, 1]
#     ], dtype=np.float64) # 使用 float64 提高精度
#     print("Created Z-up to Y-up conversion matrix.")

#     # ==================================================================
#     # STEP 3: 第二次遍历，创建【Y-up】的 JSON 文件
#     # ==================================================================
#     print("Step 3: Creating LOCALIZED Y-UP camera JSONs...")
#     for photogroup in root.findall('.//Photogroup'):
#         photogroup_count += 1
        
#         # --- 1. 获取该组共享的相机内参 (保持不变) ---
#         try:
#             W = int(photogroup.find('ImageDimensions/Width').text)
#             H = int(photogroup.find('ImageDimensions/Height').text)
#             fx = float(photogroup.find('FocalLengthPixels').text)
#             aspect_ratio_elem = photogroup.find('AspectRatio')
#             aspect_ratio = float(aspect_ratio_elem.text) if aspect_ratio_elem is not None else 1.0
#             fy = fx * aspect_ratio 
#             cx = float(photogroup.find('PrincipalPoint/x').text)
#             cy = float(photogroup.find('PrincipalPoint/y').text)
#             intrinsics_dict = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
#         except Exception as e:
#             print(f"Error parsing intrinsics for Photogroup {photogroup_count}: {e}")
#             continue

#         # --- 2. 处理该组中的所有照片 ---
#         for photo in tqdm(photogroup.findall('Photo')):
#             try:
#                 image_path = photo.find('ImagePath').text
#                 image_path_normalized = image_path.replace('\\', '/')
#                 basename = os.path.basename(image_path_normalized)
#                 filename, _ = os.path.splitext(basename)
                
#                 # A. 获取全局 Z-up 中心
#                 C_x = float(photo.find('Pose/Center/x').text)
#                 C_y = float(photo.find('Pose/Center/y').text)
#                 C_z = float(photo.find('Pose/Center/z').text)
#                 C_global_zup = np.array([C_x, C_y, C_z])
                
#                 # B. 转换为本地 Z-up 中心
#                 C_local_zup = C_global_zup - global_origin_zup
                
#                 # C. 获取 OPK 角度
#                 omega_deg = float(photo.find('Pose/Rotation/Omega').text)
#                 phi_deg = float(photo.find('Pose/Rotation/Phi').text)
#                 kappa_deg = float(photo.find('Pose/Rotation/Kappa').text)
                
#                 # D. 获取【本地 Z-up C2W 矩阵】
#                 c2w_zup = convert_opk_to_c2w(omega_deg, phi_deg, kappa_deg, C_local_zup)
                
#                 # E. 将【本地 Z-up C2W】 转换为 【本地 Y-up C2W】
#                 # c2w_yup = R_z_to_y @ c2w_zup
#                 c2w_yup = R_z_to_y_np @ c2w_zup
                
#                 # F. 将最终的 Y-up 矩阵保存到 JSON
#                 json_data = {
#                     "H": H,
#                     "W": W,
#                     "c2w": c2w_yup.tolist(), # 保存 Y-UP 矩阵
#                     "intrinsics": intrinsics_dict
#                 }
                
#                 output_path = os.path.join(args.output_dir, f"{filename}.json")
#                 with open(output_path, 'w') as f:
#                     json.dump(json_data, f, indent=4)
                
#                 photo_count += 1
#             except Exception as e:
#                 print(f"Error processing photo {photo.find('Id').text}: {e}")
                
#     print(f"\n--- Conversion Complete ---")
#     print(f"Generated {photo_count} LOCALIZED **Y-UP** .json files in {args.output_dir}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert BlocksExchange XML to localized Y-UP JSON camera files.")
#     parser.add_argument('--input_xml', required=True, help='Path to the BlocksExchangeUndistortAT.xml file.')
#     parser.add_argument('--output_dir', required=True, help='Path to the directory to save the output .json files.')
#     parser.add_argument('--origin_file', required=True, help='Path to save the calculated global origin (e.g., origin.json).')
#     args = parser.parse_args()
#     main(args)

import xml.etree.ElementTree as ET
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import math # 确保导入 math

# =======================================================================================
# ==================== convert_opk_to_c2w 函数 (Z-up) ========================
# =======================================================================================
def convert_opk_to_c2w(omega_deg, phi_deg, kappa_deg, C_local_zup):
    """
    将 OPK 角度和【Z-up 局部】相机中心转换为【Z-up 4x4 C2W】 矩阵。
    """
    r_w2c_photo = R.from_euler('zyx', [kappa_deg, phi_deg, omega_deg], degrees=True)
    R_w2c_photo_mat = r_w2c_photo.as_matrix()
    R_c2w_mat = R_w2c_photo_mat.T 
    
    c2w_zup = np.eye(4, dtype=np.float64) # 使用 float64
    c2w_zup[:3, :3] = R_c2w_mat
    c2w_zup[:3, 3] = C_local_zup
    return c2w_zup

# =======================================================================================
# ======================== 以下是 main 函数 (关键修改) ==============================
# =======================================================================================

def main(args):
    print(f"Parsing XML file: {args.input_xml}")
    tree = ET.parse(args.input_xml)
    root = tree.getroot()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created/found: {args.output_dir}")
    
    photogroup_count = 0
    photo_count = 0
    all_centers = []

    # ==================================================================
    # STEP 1: 第一次遍历，找到全局 Z-up 原点
    # ==================================================================
    print("Step 1: Finding global origin (average camera center)...")
    for photogroup in root.findall('.//Photogroup'):
        for photo in photogroup.findall('Photo'):
            try:
                C_x = float(photo.find('Pose/Center/x').text)
                C_y = float(photo.find('Pose/Center/y').text)
                C_z = float(photo.find('Pose/Center/z').text)
                all_centers.append([C_x, C_y, C_z])
            except Exception as e:
                print(f"Warning: Could not parse center for a photo: {e}")

    if not all_centers:
        print("Error: No camera centers found in XML. Exiting.")
        return

    # 计算平均中心 (Z-up)
    global_origin_zup = np.mean(np.array(all_centers), axis=0)
    print(f"Global Z-up Origin calculated: {global_origin_zup}")

    with open(args.origin_file, 'w') as f:
        json.dump(global_origin_zup.tolist(), f, indent=4)
    print(f"Global Z-up Origin saved to: {args.origin_file}")

    # ==================================================================
    # STEP 2: 定义 Z-up -> Y-up 的 4x4 变换矩阵
    # ==================================================================
    angle = -90.0
    R_z_to_y_np = np.array([
        [1, 0, 0, 0],
        [0, math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
        [0, math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    print("Created Z-up to Y-up conversion matrix.")

    # ==================================================================
    # STEP 3: 第二次遍历，创建【包含 R 和 T】的 JSON 文件
    # ==================================================================
    print("Step 3: Creating LOCALIZED Y-UP camera JSONs...")
    for photogroup in root.findall('.//Photogroup'):
        photogroup_count += 1
        
        try:
            W = int(photogroup.find('ImageDimensions/Width').text)
            H = int(photogroup.find('ImageDimensions/Height').text)
            fx = float(photogroup.find('FocalLengthPixels').text)
            aspect_ratio_elem = photogroup.find('AspectRatio')
            aspect_ratio = float(aspect_ratio_elem.text) if aspect_ratio_elem is not None else 1.0
            fy = fx * aspect_ratio 
            cx = float(photogroup.find('PrincipalPoint/x').text)
            cy = float(photogroup.find('PrincipalPoint/y').text)
            intrinsics_dict = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
        except Exception as e:
            print(f"Error parsing intrinsics for Photogroup {photogroup_count}: {e}")
            continue

        for photo in tqdm(photogroup.findall('Photo')):
            try:
                image_path = photo.find('ImagePath').text
                image_path_normalized = image_path.replace('\\', '/')
                basename = os.path.basename(image_path_normalized)
                filename, _ = os.path.splitext(basename)
                
                C_x = float(photo.find('Pose/Center/x').text)
                C_y = float(photo.find('Pose/Center/y').text)
                C_z = float(photo.find('Pose/Center/z').text)
                C_global_zup = np.array([C_x, C_y, C_z])
                C_local_zup = C_global_zup - global_origin_zup
                
                omega_deg = float(photo.find('Pose/Rotation/Omega').text)
                phi_deg = float(photo.find('Pose/Rotation/Phi').text)
                kappa_deg = float(photo.find('Pose/Rotation/Kappa').text)
                
                # A. 获取【本地 Z-up C2W】
                c2w_zup = convert_opk_to_c2w(omega_deg, phi_deg, kappa_deg, C_local_zup)
                
                # B. 转换为 【本地 Y-up C2W】
                c2w_yup = R_z_to_y_np @ c2w_zup
                
                # =================== 关键修改 ===================
                # C. 在 Numpy (float64) 中计算 W2C, R, T
                w2c_yup = np.linalg.inv(c2w_yup)
                R_yup = w2c_yup[:3, :3].T
                T_yup = w2c_yup[:3, 3]
                # ================================================
                
                # D. 将【R 和 T】保存到 JSON
                json_data = {
                    "H": H,
                    "W": W,
                    "R": R_yup.tolist(), # 保存 R
                    "T": T_yup.tolist(), # 保存 T
                    # "c2w": c2w_yup.tolist(), # 不再保存 c2w
                    "intrinsics": intrinsics_dict
                }
                
                output_path = os.path.join(args.output_dir, f"{filename}.json")
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
                
                photo_count += 1
            except Exception as e:
                print(f"Error processing photo {photo.find('Id').text}: {e}")
                
    print(f"\n--- Conversion Complete ---")
    print(f"Generated {photo_count} LOCALIZED **Y-UP (R, T)** .json files in {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BlocksExchange XML to localized Y-UP (R, T) JSON camera files.")
    parser.add_argument('--input_xml', required=True, help='Path to the BlocksExchangeUndistortAT.xml file.')
    parser.add_argument('--output_dir', required=True, help='Path to the directory to save the output .json files.')
    parser.add_argument('--origin_file', required=True, help='Path to save the calculated global origin (e.g., origin.json).')
    args = parser.parse_args()
    main(args)

    '''
mkdir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/camera_jsons

python szq_GES_convert_xml_to_json.py \
    --input_xml /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/BlocksExchangeUndistortAT.xml \
    --output_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/camera_jsons
    
    python szq_GES_convert_xml_to_json.py \
    --input_xml /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/BlocksExchangeUndistortAT.xml \
    --output_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/camera_jsons \
    --origin_file /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar_mesh_szq/GES/origin.json


    '''