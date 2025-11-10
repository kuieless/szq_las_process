import argparse
import os
import sys
import torch
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import exifread
import math
from PIL import Image
from PIL.ExifTags import TAGS
import xml.etree.ElementTree as ET

import pytorch3d

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(call_pdb=True)

from IPython.core.debugger import set_trace # debug

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
# 修复：join_meshes_as_scene 位于 structures 模块
from pytorch3d.structures import Meshes, join_meshes_as_scene 

# Data structures and functions for rendering
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib

from pytorch3d.renderer.mesh.shader import SoftDepthShader, HardDepthShader
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    AmbientLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRendererWithFragments, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    TexturesAtlas,
)
import glob
from tqdm import tqdm


# Setup
if torch.cuda.is_available():
    # 注意：你的 OOM 错误发生在 GPU 4。
    # 如果你有多张卡，请确保 device 索引是正确的。
    device = torch.device("cuda:1") 
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# ==========================================================================================
# ======================== NEW HELPER FUNCTION FOR DEPTH COLOR BAR =========================
# ==========================================================================================
def create_depth_colorbar(height, width, min_val, max_val, colormap=cv2.COLORMAP_JET):
    """Creates a color bar image for depth visualization."""
    # Create a gradient array for the bar itself
    bar = np.linspace(0, 1, height).reshape(height, 1)
    bar_uint8 = (bar * 255).astype(np.uint8)
    
    # Apply colormap and flip it so that the color for min_val is at the bottom
    color_bar_img = cv2.applyColorMap(255 - bar_uint8, colormap)
    
    # Repeat to make it wider
    color_bar_img = cv2.repeat(color_bar_img, 1, width)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)
    
    # Add a title
    cv2.putText(color_bar_img, "Depth (m)", (5, 30), font, 0.8, font_color, 2, cv2.LINE_AA)

    # Add tick marks and labels for several points
    num_labels = 7
    for i in range(num_labels):
        p = i / (num_labels - 1)  # percentage from top (0.0 to 1.0)
        y = int(p * (height - 60)) + 30  # y-coordinate
        val = max_val * (1 - p) + min_val * p  # corresponding depth value
        
        # Draw text label
        cv2.putText(color_bar_img, f"{val:.1f}", (10, y + 8), font, 0.6, font_color, 2, cv2.LINE_AA)
            
    return color_bar_img
# ==========================================================================================
# ================================= END OF NEW FUNCTION ====================================
# ==========================================================================================


def torch_depth_to_colormap_np(depth):
    depth_normalized = torch.zeros(depth.shape, device=device)

    valid_mask = depth > -0.9 # valid
    if valid_mask.sum() > 0:
        d_valid = depth[valid_mask]
        depth_normalized[valid_mask] = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min())

        depth_np = (depth_normalized.cpu().numpy()[0] * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        depth_normalized = depth_normalized.cpu().numpy()[0] * 255
    else:
        print('!!!! No depth projected !!!')
        depth_color = depth_normalized = np.zeros(depth.shape, dtype=np.uint8)
    return depth_color, depth_normalized

def np_depth_to_colormap(depth, normalize=True, thres=-1, min_depth_percentile=5, max_depth_percentile=95):
    """ depth: [H, W] """
    depth_normalized = np.zeros(depth.shape)

    valid_mask = depth > -0.9 # depth for invalid pixel is -1
    if valid_mask.sum() > 0:
        d_valid = depth[valid_mask]

        if normalize:
            depth_normalized[valid_mask] = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min())
        elif thres > 0:
            depth_normalized[valid_mask] = d_valid.clip(0, thres) / thres
        elif min_depth_percentile>0:
            depth_normalized[valid_mask] = d_valid
            min_depth, max_depth = np.percentile(
                d_valid, [min_depth_percentile, max_depth_percentile]) # Corrected to use d_valid
            depth_normalized[valid_mask & (depth < min_depth)] = min_depth
            depth_normalized[valid_mask & (depth > max_depth)] = max_depth
            if max_depth > 0:
                 # Normalize based on the percentile range for better color mapping
                depth_normalized[valid_mask] = (depth_normalized[valid_mask] - min_depth) / (max_depth - min_depth)
            else:
                depth_normalized[valid_mask] = 0
        else:
            depth_normalized[valid_mask] = d_valid

        depth_np = (depth_normalized * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        depth_normalized = depth_normalized
    else:
        print('!!!! No depth projected !!!')
        depth_color = depth_normalized = np.zeros((*depth.shape,3), dtype=np.uint8)
    return depth_color, depth_normalized

def compute_vert_v2(flat_vertices):
    import numpy as np
    from sklearn.decomposition import PCA

    # 假设 vertices 包含了 mesh 的顶点数据
    pca = PCA(n_components=3)
    pca.fit(flat_vertices)
    normal_vector = pca.components_[2]  # 第三个主成分即为法线向量

    return normal_vector

def matrix_rotateB2A(A, B):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    rot_axis = np.cross(A, B)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(A, B))

    # rotate Matrix according to Rodrigues formula
    a = np.cos(rot_angle / 2.0)
    b, c, d = -rot_axis * np.sin(rot_angle / 2.0)  # flip the rot_axis
    rotation_matrix = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]
    ])
    return rotation_matrix

def compute_MRM(flat_vertices):
    X_axis = np.array([1, 0, 0])
    # target_axis = compute_vert(flat_vertices)
    target_axis = compute_vert_v2(flat_vertices)
    MRM = matrix_rotateB2A(X_axis, target_axis)
    return MRM

def load_mesh_debug(obj_filename, nerf_metadata=None, save_mesh=False):
    """
    docstring for load_mesh
    (This debug function remains mostly unchanged, operating on a SINGLE obj_filename)
    """
    verts, faces, aux = load_obj(obj_filename, device=device,
                                 load_textures=True,
                                 create_texture_atlas=True,
                                 texture_atlas_size=8,
                                 texture_wrap=None,
                                 )
    print(f'loaded obj {obj_filename} for debugging')
    atlas = aux.texture_atlas

    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=TexturesAtlas(atlas=[atlas]),
    )        
    IO = pytorch3d.io.IO()
    # 路径是硬编码的，保留原样
    IO.save_mesh(mesh, '/data/jxchen/dji/Yingrenshi/only2mesh.obj')

    if nerf_metadata is not None:
        verts1, verts2, verts3, verts4 = convert_verts_to_nerf_space_debug(verts, nerf_metadata)
    else:
        print("Warning! No nerf metadata!")
        return mesh # Return early if no metadata

    mesh = Meshes(verts=[verts1], faces=[faces.verts_idx], textures=TexturesAtlas(atlas=[atlas]))        
    IO = pytorch3d.io.IO()
    IO.save_mesh(mesh, '/data/jxchen/dji/Yingrenshi/trans_mesh.obj')

    mesh = Meshes(verts=[verts2], faces=[faces.verts_idx], textures=TexturesAtlas(atlas=[atlas]))        
    IO = pytorch3d.io.IO()
    IO.save_mesh(mesh, '/data/jxchen/dji/Yingrenshi/T1_trans_mesh.obj')

    mesh = Meshes(verts=[verts3], faces=[faces.verts_idx], textures=TexturesAtlas(atlas=[atlas]))        
    IO = pytorch3d.io.IO()
    IO.save_mesh(mesh, '/data/jxchen/dji/Yingrenshi/T2_T1_trans_mesh.obj')

    mesh = Meshes(verts=[verts4], faces=[faces.verts_idx], textures=TexturesAtlas(atlas=[atlas]))        
    IO = pytorch3d.io.IO()
    IO.save_mesh(mesh, '/data/jxchen/dji/Yingrenshi/finalmesh.obj')
    return mesh


def load_all_meshes(args, obj_filenames_list, nerf_metadata=None, save_mesh=False):
    """
    Loads all .obj files from the provided list, transforms them to NeRF space,
    and returns a single batched Meshes object.
    """
    all_verts_list = []
    all_faces_list = []
    
    # 仅在 save_mesh=True (即在主循环开始前调用) 或 VRAM 足够大时打印
    # 否则在分批加载时会打印过多信息
    if save_mesh:
        print(f"Loading and processing {len(obj_filenames_list)} mesh files...")

    for obj_filename in obj_filenames_list:
        if save_mesh:
            print(f"  Loading: {os.path.basename(obj_filename)}")
        # Load obj file without textures, as we only need depth
        verts, faces, aux = load_obj(obj_filename, device=device,
                                     load_textures=False,
                                     create_texture_atlas=False
                                     )
        
        # if save_mesh: print(f'    Verts before:', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

        if nerf_metadata is not None:
            verts = convert_verts_to_nerf_space(verts, nerf_metadata, verbose=save_mesh)
            # if save_mesh: print(f'    Verts after NeRF conversion:', verts.max(0)[0], verts.min(0)[0], verts.mean(0))
        else:
            """ normalize """
            if save_mesh: print("    Warning! No nerf metadata for mesh coordinate conversion!")
            verts -= verts.mean(0, keepdim=True)
            verts /= verts.max()
            # if save_mesh: print(f'    Verts after normalization:', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

        all_verts_list.append(verts)
        all_faces_list.append(faces.verts_idx)

    # Create a batched Meshes object. The renderer will handle
    # rendering all meshes in this batch.
    if save_mesh: print("Combining all meshes into a batched Meshes object.")
    mesh = Meshes(
        verts=all_verts_list,
        faces=all_faces_list,
    )        
    
    if save_mesh:
        print("Saving combined (joined) mesh...")
        # To save a batched mesh, we first join it into a single mesh
        mesh_joined = join_meshes_as_scene(mesh)
        IO = pytorch3d.io.IO()
        save_path = f'{args.save_dir}/norm_combined.obj'
        IO.save_mesh(mesh_joined, save_path)
        print(f"Saved joined mesh to {save_path}")

    return mesh

def load_nerf_metadata(args, metadata_dir, filename, verbose=True):
    """
    docstring for load_nerf_metadata
    (REMOVED args.split from paths)
    """
    coordinates = torch.load(os.path.join(metadata_dir, 'coordinates.pt'))
    
    # Load RGB
    rgb_path = os.path.join(metadata_dir, 'rgbs/%s.jpg' % filename)
    if not os.path.exists(rgb_path):
        if verbose: print(f"Error: Cannot find RGB image at {rgb_path}")
        return None
    rgb = cv2.imread(rgb_path)
    
    depth = np.array([]) # Not used in this script
    
    # Load metadata
    metadata_path = os.path.join(metadata_dir, 'metadata/%s.pt' % filename)
    if not os.path.exists(metadata_path):
        if verbose: print(f"Error: Cannot find metadata at {metadata_path}")
        return None
    metadata = torch.load(metadata_path)

    root = ET.parse(args.metadataXml_path).getroot()
    # Robustly find SRSOrigin, default to 0 if not found.
    srs_origin_element = root.find('SRSOrigin')
    if srs_origin_element is not None and srs_origin_element.text:
        translation = np.array(srs_origin_element.text.split(','), dtype=float)
        if verbose: print("Found <SRSOrigin> in XML, using it as translation.")
    else:
        if verbose: print("WARNING: <SRSOrigin> tag not found in XML. Assuming mesh vertices are in absolute coordinates (translation set to zero).")
        translation = np.array([0.0, 0.0, 0.0], dtype=float)

    if verbose:
        print('rgb', rgb.shape)
        print('depth', depth.shape)
        print('meta data', metadata)
        print(coordinates)

    nerf_metadata = {'coordinates': coordinates, 'rgb': rgb, 'depth': depth,
                     'metadata': metadata, 'translation': translation}
    return nerf_metadata

def convert_verts_to_nerf_space(verts, nerf_metadata, verbose=True):
    """docstring for convert_mesh_to_nerf_space"""
    origin_drb = nerf_metadata['coordinates']['origin_drb'].to(device).to(torch.float64)
    # This is a more robust way to handle the scale factor
    pose_scale_factor = torch.tensor(nerf_metadata['coordinates']['pose_scale_factor']).to(device).to(torch.float64)
    translation = torch.from_numpy(nerf_metadata['translation']).float().to(device)

    def rad(x):
        return math.radians(x)

    T1 = torch.FloatTensor([[0, 0, -1],
                            [0, 1, 0],
                            [1, 0, 0]]).to(device).to(torch.float64)
    T2 = torch.FloatTensor([[1, 0, 0],
                            [0, math.cos(rad(135)), math.sin(rad(135))],
                            [0, -math.sin(rad(135)), math.cos(rad(135))]]).to(device).to(torch.float64)

    verts += translation
    verts = verts.to(torch.float64)

    verts_nerf = T1 @ verts.T
    verts_nerf = (T2 @ verts_nerf).T

    verts_nerf = (verts_nerf - origin_drb) / pose_scale_factor
    if verbose:
        print("origin:{}  scale:{}".format(origin_drb, pose_scale_factor))
    return verts_nerf.to(torch.float32)

def convert_verts_to_nerf_space_debug(verts, nerf_metadata):
    """docstring for convert_mesh_to_nerf_space"""
    origin_drb = nerf_metadata['coordinates']['origin_drb'].to(device).to(torch.float64)
    # This is a more robust way to handle the scale factor
    pose_scale_factor = torch.tensor(nerf_metadata['coordinates']['pose_scale_factor']).to(device).to(torch.float64)
    translation = torch.from_numpy(nerf_metadata['translation']).float().to(device)

    def rad(x):
        return math.radians(x)

    T1 = torch.FloatTensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).to(device).to(torch.float64)
    T2 = torch.FloatTensor([[1, 0, 0],
                            [0, math.cos(rad(135)), math.sin(rad(135))],
                            [0, -math.sin(rad(135)), math.cos(rad(135))]]).to(device).to(torch.float64)

    verts1 = verts + translation
    verts1 = verts1.to(torch.float64)

    verts2 = (T1 @ verts1.T).T
    verts3 = (T2 @ verts2.T).T

    verts4 = (verts3 - origin_drb) / pose_scale_factor
    print("origin:{}  scale:{}".format(origin_drb, pose_scale_factor))
    return verts1, verts2, verts3, verts4

def setup_renderer(args, nerf_metadata, verbose=True):
    if args.camera_type == 'perspective' and nerf_metadata is not None:
        c2w = nerf_metadata['metadata']['c2w']
        R, T = c2w[:3, :3], c2w[:3, 3:]

        if verbose: print('origin c2w\n', torch.cat([R, T], 1))
        R_ = torch.stack([-R[:, 0], R[:, 1], -R[:, 2]], 1) # Renamed to R_
        new_c2w = torch.cat([R_, T], 1)
        if verbose: print('c2w\n', new_c2w)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
        
        # =========================================================
        # ==================== 警告修复 ====================
        R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3] # 旧代码
        # R, T = w2c[:3, :3], w2c[:3, 3] # 新代码
        # R, T = w2c[:3, :3], w2c[:3, 3]
        # 解释：PerspectiveCameras 需要 world-to-camera 旋转矩阵 (R_w2c)。
        # w2c[:3, :3] *已经* 是 R_w2c。
        # 对其转置 (.permute(1,0)) 会得到 R_c2w，这是错误的，
        # 导致了 "R is not a valid rotation matrix" 警告。
        # =========================================================
        
        if verbose: print('w2c\n', w2c)

        R = R[None]
        T = T[None]

        H, W = nerf_metadata['metadata']['H'], nerf_metadata['metadata']['W']
        H, W = int(H / args.down / args.simplify), int(W / args.down / args.simplify)

        intrinsics = nerf_metadata['metadata']['intrinsics'] / args.down / args.simplify

        image_size_tuple = ((H, W),)
        fcl_screen = ((intrinsics[0], intrinsics[1]),)
        prp_screen = ((intrinsics[2], intrinsics[3]), )
        cameras = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size_tuple, R=R, T=T, device=device)
        
        if verbose: print(cameras)
        image_size = (H, W)

    elif args.camera_type == 'fovperspective':
        R, T = look_at_view_transform(2.7, 0, 180) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        image_size = args.image_size # 假设 args.image_size 已定义

    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        perspective_correct=True,
    )

    lights = AmbientLights(device=device)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # Use HardDepthShader for rendering depth
    renderer = MeshRendererWithFragments(
        rasterizer=rasterizer,
        shader=HardDepthShader(device=device, cameras=cameras, lights=lights)
    )
    render_setup = {'cameras': cameras, 'raster_settings': raster_settings, 'lights': lights,
                    'rasterizer': rasterizer, 'renderer': renderer}
    return render_setup


def _get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nerf_metadata_dir', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan2', help='Directory containing rgbs/, metadata/, coordinates.pt')
    parser.add_argument('--obj_dir', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/original_image/terra_obj/Block', help='Directory containing all .obj mesh files (e.g., Block_1.obj, Block_2.obj)')
    parser.add_argument('--save_dir', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan2', help='Directory to save outputs')
    parser.add_argument('--metadataXml_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/original_image/terra_obj/metadata.xml', help='Path to metadata.xml for SRSOrigin')
    parser.add_argument('--down', default=1, type=float, help='Downsampling factor for images')
    parser.add_argument('--camera_type', default='perspective', help='[perspective, fovperspective]')
    parser.add_argument('--save_mesh', default=False, action='store_true', help='Save the transformed, combined mesh')
    parser.add_argument('--visualize', default=True, action='store_true', help='Generate and save visualization images')
    parser.add_argument('--debug', default=False, action='store_true', help='Run in debug mode (loads only first mesh with debug steps)')
    parser.add_argument('--simplify', default=1, type=int)
    # 添加一个新参数来控制分批渲染的大小
    parser.add_argument('--mesh_batch_size', default=10, type=int, help='Number of .obj files to load and render in one batch to control VRAM usage.')
    return parser.parse_known_args()[0]


# ==============================================================
# =================== 新的辅助函数 ===================
# ==============================================================
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
# ==============================================================
# =================== 完整的 MAIN 函数 ===================
# ==============================================================
# def main(args):
#     save_dir = args.save_dir
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Create output directories without 'split'
#     if args.visualize:
#         os.makedirs(os.path.join(save_dir, 'cat_results'), exist_ok=True)
#         os.makedirs(os.path.join(save_dir, 'altitude_depth_vis'), exist_ok=True)
        
#     os.makedirs(os.path.join(save_dir, 'depth_mesh'), exist_ok=True)

#     nerf_metadata_dir = args.nerf_metadata_dir    

#     # --- Load Image Names ---
#     names = sorted(glob.glob(os.path.join(nerf_metadata_dir, 'rgbs/*.jpg')))
#     if not names:
#         print(f"Error: No images found in {os.path.join(nerf_metadata_dir, 'rgbs/')}")
#         print("Please check --nerf_metadata_dir. It should contain an 'rgbs' subfolder.")
#         return
        
#     print(f"Found {len(names)} images to process.")

#     # --- Load NeRF Metadata for first image (needed for mesh transform) ---
#     print("Loading metadata from first image to get coordinate transform...")
#     filename_first = os.path.basename(names[0])[:-4]
#     nerf_metadata_for_coords = load_nerf_metadata(args, nerf_metadata_dir, filename_first) 
#     if nerf_metadata_for_coords is None:
#         print(f"Error loading metadata for {filename_first}. Exiting.")
#         return

#     # --- Get Mesh Filenames (但不加载) ---
#     obj_filenames = sorted(glob.glob(os.path.join(args.obj_dir, '*.obj')))
#     if not obj_filenames:
#         print(f"Error: No .obj files found in {args.obj_dir}")
#         print("Please check --obj_dir.")
#         return
        
#     print(f"Found {len(obj_filenames)} .obj files in {args.obj_dir}")

#     # --- [!!] OOM 修复：修改网格加载逻辑 [!!] ---
    
#     debug_mesh = None
#     if args.debug:
#         print(f"--- DEBUG MODE ---")
#         print(f"Loading and debugging ONLY the first mesh: {obj_filenames[0]}")
#         # 在 debug 模式下，我们加载并保留第一个网格
#         debug_mesh = load_mesh_debug(obj_filenames[0], nerf_metadata_for_coords, save_mesh=args.save_mesh)      
    
#     elif args.save_mesh:
#         # 在非 debug 模式下，如果 save_mesh=True, 
#         # 我们只在开始前加载一次所有网格，以保存组合文件
#         print("--- Pre-processing: Loading all meshes ONCE to save combined .obj ---")
#         mesh_to_save = load_all_meshes(args, obj_filenames, nerf_metadata_for_coords, save_mesh=True)
#         # 保存后立即删除，释放 VRAM
#         del mesh_to_save
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         print("--- Combined mesh saved. Proceeding to batch rendering loop. ---")
    
#     # 在正常（非 debug, 非 save_mesh）模式下，我们什么都不做，
#     # 因为网格将在下面的循环中分批加载。

#     print("**********************************")
#     print(f"Now rendering {len(names)} images...")
#     print(f"Using mesh batch size: {args.mesh_batch_size}")
#     print("**********************************")
    
#     # 定义网格批次大小
#     MESH_BATCH_SIZE = args.mesh_batch_size
    
#     for name in tqdm(names[:]):
#         filename = os.path.basename(name)[:-4]
#         # 详细模式 (verbose=False) 关闭，避免在循环中打印过多信息
#         nerf_metadata = load_nerf_metadata(args, nerf_metadata_dir, filename, verbose=False)
#         if nerf_metadata is None:
#             print(f"Skipping {filename} due to metadata loading error.")
#             continue
            
#         render_setup = setup_renderer(args, nerf_metadata, verbose=False)
#         renderer = render_setup['renderer']
        
#         # 获取图像尺寸用于初始化深度图
#         H, W = render_setup['raster_settings'].image_size
        
#         # 1. 初始化一个空的 (或全为无穷大) 的深度缓冲
#         # 我们用一个很大的值 (1e10) 来表示"未渲染"
#         final_depth_buffer = torch.full((H, W), 1e10, device=device, dtype=torch.float32)
        
#         if args.debug:
#             # --- DEBUG 渲染路径 ---
#             # 只渲染已加载的 'debug_mesh'
#             if debug_mesh is not None:
#                 images, fragments = renderer(debug_mesh)
#                 current_depth = fragments.zbuf[0, ..., 0]
#                 valid_mask = current_depth > -1
#                 current_depth[~valid_mask] = 1e10 # 将无效 (-1) 设置为无穷大
#                 final_depth_buffer = current_depth # 覆盖
            
#         else:
#             # --- 分批渲染路径 (正常操作) ---
#             # 2. 将所有 obj 文件名分成小批次
#             for mesh_file_batch in chunks(obj_filenames, MESH_BATCH_SIZE):
                
#                 # 3. 加载这一小批 mesh
#                 # 使用 nerf_metadata_for_coords 进行坐标变换
#                 batched_mesh = load_all_meshes(args, mesh_file_batch, nerf_metadata_for_coords, save_mesh=False)
                
#                 # 4. 渲染这一小批 mesh
#                 images, fragments = renderer(batched_mesh)
                
#                 # 5. 获取当前批次的深度图
#                 current_depth = fragments.zbuf[0, ..., 0] 
                
#                 # 6. 将当前深度图合并到最终的深度缓冲中
#                 valid_mask = current_depth > -1
#                 current_depth[~valid_mask] = 1e10 # 将无效 (-1) 设置为无穷大
                
#                 # 关键一步：取两个深度图的逐像素最小值 (Z-buffering)
#                 final_depth_buffer = torch.minimum(final_depth_buffer, current_depth)
                
#                 # 7. (关键) 清理 VRAM，为下一批做准备
#                 del batched_mesh, images, fragments, current_depth
#                 if torch.cuda.is_available():
#                     torch.cuda.empty_cache()

#         # --- 循环结束 ---
#         # 此时, final_depth_buffer 包含了所有 mesh 的组合深度
        
#         # 将最终的 torch 深度张量转为 numpy
#         depth = final_depth_buffer.cpu().numpy()

#         # 从元数据中获取尺度恢复因子
#         pose_scale_factor = nerf_metadata['coordinates']['pose_scale_factor']

#         # 将有效的深度值乘以尺度因子
#         valid_mask = depth < 1e9 # 我们的"无效"值是 1e10
#         depth[valid_mask] *= pose_scale_factor
        
#         # 将无效值设置为 1e6 以便保存
#         depth[~valid_mask] = 1e6

#         if args.simplify == 1:
#             # Save to the non-split directory
#             np.save(os.path.join(save_dir, 'depth_mesh', f'{filename}.npy'), depth[:,:,None].astype(np.float32))
        
#         # --- Visualization Logic ---
#         if args.visualize:
#             # Set invalid depth back to -1 for visualization processing
#             depth[depth == 1e6] = -1

#             # --- 修复后的原始可视化 ---
#             # 我们现在使用最终的 'depth' 来生成两个 colormap
#             depth_color, _ = np_depth_to_colormap(depth, normalize=False)
#             lidar_depth_color, _ = np_depth_to_colormap(depth, normalize=False)
            
#             ori_image = nerf_metadata['rgb']
#             ori_image = cv2.resize(ori_image, (depth_color.shape[1], depth_color.shape[0]))
            
#             # `rgb_np_color` (来自'images') 在分批逻辑下是无效的，
#             # 所以我们只连接原始图像和最终深度图。
#             cat_results = np.concatenate([ori_image, depth_color], 1)
#             cv2.imwrite(os.path.join(save_dir, 'cat_results', f'{filename}_cat_results.jpg'), cat_results.astype(np.uint8))
            
#             # ==========================================================================================
#             # ======================== NEW VISUALIZATION LOGIC (不变) ===============================
#             # ==========================================================================================
            
#             valid_depth = depth[depth > -0.9]
#             if valid_depth.size > 0:
#                 # 1. Get altitude from metadata and annotate the original image
#                 c2w = nerf_metadata['metadata']['c2w']
#                 altitude = c2w[2, 3].item()  # Camera's Z position in world coordinates
#                 image_with_text = ori_image.copy()
#                 altitude_text = f"Altitude: {altitude:.2f}m"
                
#                 # Add a black background for better text readability
#                 (text_width, text_height), baseline = cv2.getTextSize(altitude_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
#                 cv2.rectangle(image_with_text, (10, 10), (20 + text_width, 20 + text_height + baseline), (0,0,0), -1)
#                 cv2.putText(image_with_text, altitude_text, (15, 20 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

#                 # 2. Prepare the depth map and its color bar
#                 min_d, max_d = np.percentile(valid_depth, [5, 95])
                
#                 # Handle case where depth is constant
#                 if max_d <= min_d: max_d = min_d + 1.0

#                 colorbar = create_depth_colorbar(height=depth_color.shape[0], width=120, min_val=min_d, max_val=max_d)
                
#                 # 3. Combine the images into the new visualization format
#                 new_visualization = np.concatenate([image_with_text, depth_color, colorbar], axis=1)

#                 # 4. Save the new visualization
#                 new_vis_dir = os.path.join(save_dir, 'altitude_depth_vis')
#                 save_path = os.path.join(new_vis_dir, f'{filename}_comparison.jpg')
#                 cv2.imwrite(save_path, new_visualization)
#             # ==========================================================================================
#             # ========================== NEW VISUALIZATION LOGIC ENDS HERE =============================
#             # ==========================================================================================

#     # 循环后清理 debug_mesh (如果存在)
#     if debug_mesh is not None:
#         del debug_mesh
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

# if __name__ == '__main__':
#     main(_get_opts())

# ==============================================================
# =================== 完整的 MAIN 函数 (已修改) ===================
# ==============================================================
def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # Create output directories
    if args.visualize:
        os.makedirs(os.path.join(save_dir, 'cat_results'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'altitude_depth_vis'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'depth_mesh'), exist_ok=True)

    nerf_metadata_dir = args.nerf_metadata_dir 

    # --- Load Image Names ---
    names = sorted(glob.glob(os.path.join(nerf_metadata_dir, 'rgbs/*.jpg')))
    if not names:
        print(f"Error: No images found in {os.path.join(nerf_metadata_dir, 'rgbs/')}")
        return
    print(f"Found {len(names)} images to process.")

    # --- Load NeRF Metadata for first image (needed for mesh transform) ---
    print("Loading metadata from first image to get coordinate transform...")
    filename_first = os.path.basename(names[0])[:-4]
    nerf_metadata_for_coords = load_nerf_metadata(args, nerf_metadata_dir, filename_first) 
    if nerf_metadata_for_coords is None:
        print(f"Error loading metadata for {filename_first}. Exiting.")
        return

    # --- Get Mesh Filenames ---
    obj_filenames = sorted(glob.glob(os.path.join(args.obj_dir, '*.obj')))
    if not obj_filenames:
        print(f"Error: No .obj files found in {args.obj_dir}")
        return
    
    # =====================================================================
    # ======================== 核心修改：加载一次 MESH ========================
    # =====================================================================
    
    # 假设你已经合并了，所以 obj_filenames 列表应该只有一个（或几个）文件
    # 我们使用 load_all_meshes 函数一次性加载所有找到的 .obj 文件
    # 并且我们传入 save_mesh=False 来避免它再次保存
    print(f"--- Loading and transforming {len(obj_filenames)} mesh file(s) ONCE... ---")
    print("This may take a while if the mesh is large...")
    
    # load_all_meshes 会加载列表中的所有 obj，将它们全部转换到 NeRF 空间，
    # 并返回一个单一的、在 GPU 上的 batched Meshes 对象。
    # 这正是我们想要的。
    persistent_mesh = load_all_meshes(
        args, 
        obj_filenames, 
        nerf_metadata_for_coords, 
        save_mesh=False # 确保不要再次保存
    )
    
    print("--- Mesh is now loaded and persistent on GPU. Starting render loop. ---")
    # =====================================================================
    # ========================   修改结束   ========================
    # =====================================================================

    # [!!] 我们不再需要这个逻辑，因为 mesh 已经在 persistent_mesh 中了
    # if args.debug: ...
    # elif args.save_mesh: ...
    
    print("**********************************")
    print(f"Now rendering {len(names)} images...")
    # MESH_BATCH_SIZE 也不再需要了
    # print(f"Using mesh batch size: {args.mesh_batch_size}") 
    print("**********************************")
    
    
    for name in tqdm(names[:]):
        filename = os.path.basename(name)[:-4]
        nerf_metadata = load_nerf_metadata(args, nerf_metadata_dir, filename, verbose=False)
        if nerf_metadata is None:
            print(f"Skipping {filename} due to metadata loading error.")
            continue
            
        # 这一步必须在循环内，因为每张图的相机都不同
        render_setup = setup_renderer(args, nerf_metadata, verbose=False)
        renderer = render_setup['renderer']
        
        # 获取图像尺寸
        H, W = render_setup['raster_settings'].image_size
        
        # =====================================================================
        # ======================== 核心修改：渲染 =========================
        # =====================================================================
        
        # [!!] 删除了内部的 "chunks" 循环
        # 我们不再需要分批加载 mesh，我们直接渲染那个持久化的 mesh
        
        images, fragments = renderer(persistent_mesh)
        
        # 获取深度图 (zbuf)
        # (B, H, W, K) -> (H, W)
        current_depth = fragments.zbuf[0, ..., 0] 
        
        # [!!] 删除了 Z-buffering 逻辑 (torch.minimum)
        # 因为我们一次性渲染了整个合并的 mesh，
        # 渲染器已经帮我们处理好了 Z-buffering。
        
        # [!!] 删除了 VRAM 清理逻辑 (del batched_mesh)
        # 我们希望 persistent_mesh 一直存在。

        # =====================================================================
        # ========================   修改结束   ========================
        # =====================================================================

        # --- 循环结束 ---
        # 此时, current_depth 包含了组合深度
        
        # 将最终的 torch 深度张量转为 numpy
        depth = current_depth.cpu().numpy()

        # 从元数据中获取尺度恢复因子
        pose_scale_factor = nerf_metadata['coordinates']['pose_scale_factor']

        # 将有效的深度值乘以尺度因子
        # HardDepthShader 对于无效像素返回 -1
        valid_mask = depth > -1 
        depth[valid_mask] *= pose_scale_factor
        
        # 将无效值设置为 1e6 以便保存
        depth[~valid_mask] = 1e6

        if args.simplify == 1:
            np.save(os.path.join(save_dir, 'depth_mesh', f'{filename}.npy'), depth[:,:,None].astype(np.float32))
        
        # --- Visualization Logic (保持不变) ---
        if args.visualize:
            depth[depth == 1e6] = -1 # 设置回 -1 以便可视化
            
            depth_color, _ = np_depth_to_colormap(depth, normalize=False)
            
            ori_image = nerf_metadata['rgb']
            ori_image = cv2.resize(ori_image, (depth_color.shape[1], depth_color.shape[0]))
            
            cat_results = np.concatenate([ori_image, depth_color], 1)
            cv2.imwrite(os.path.join(save_dir, 'cat_results', f'{filename}_cat_results.jpg'), cat_results.astype(np.uint8))
            
            # --- 新的可视化 (保持不变) ---
            valid_depth = depth[depth > -0.9]
            if valid_depth.size > 0:
                c2w = nerf_metadata['metadata']['c2w']
                altitude = c2w[2, 3].item()
                image_with_text = ori_image.copy()
                altitude_text = f"Altitude: {altitude:.2f}m"
                
                (text_width, text_height), baseline = cv2.getTextSize(altitude_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                cv2.rectangle(image_with_text, (10, 10), (20 + text_width, 20 + text_height + baseline), (0,0,0), -1)
                cv2.putText(image_with_text, altitude_text, (15, 20 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

                min_d, max_d = np.percentile(valid_depth, [5, 95])
                if max_d <= min_d: max_d = min_d + 1.0
                colorbar = create_depth_colorbar(height=depth_color.shape[0], width=120, min_val=min_d, max_val=max_d)
                
                new_visualization = np.concatenate([image_with_text, depth_color, colorbar], axis=1)

                new_vis_dir = os.path.join(save_dir, 'altitude_depth_vis')
                save_path = os.path.join(new_vis_dir, f'{filename}_comparison.jpg')
                cv2.imwrite(save_path, new_visualization)
        
        # 清理循环内的变量，但保留 persistent_mesh
        del images, fragments, current_depth, depth, nerf_metadata, render_setup, renderer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 循环全部结束后，再清理 persistent_mesh
    del persistent_mesh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main(_get_opts())