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

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
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


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:4")
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
    """docstring for load_mesh"""
    verts, faces, aux = load_obj(obj_filename, device=device,
                                 load_textures=True,
                                 create_texture_atlas=True,
                                 texture_atlas_size=8,
                                 texture_wrap=None,
                                 )
    print('loaded obj')
    atlas = aux.texture_atlas

    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
        textures=TexturesAtlas(atlas=[atlas]),
    )       
    IO = pytorch3d.io.IO()
    IO.save_mesh(mesh, '/data/jxchen/dji/Yingrenshi/only2mesh.obj')

    if nerf_metadata is not None:
        verts1, verts2, verts3, verts4 = convert_verts_to_nerf_space_debug(verts, nerf_metadata)
    else:
        print("Warning! No nerf metadata!")

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


def load_mesh(args, obj_filename, nerf_metadata=None, save_mesh=False):
    # Load obj file without textures, as we only need depth
    verts, faces, aux = load_obj(obj_filename, device=device,
                                   load_textures=False,
                                   create_texture_atlas=False
                                   )

    print('loaded obj')
    
    print('verts before', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

    if nerf_metadata is not None:
        verts = convert_verts_to_nerf_space(verts, nerf_metadata)
        print('Convert tot NeRF', verts.max(0)[0], verts.min(0)[0], verts.mean(0))
    else:
        """ normalize """
        print("Warning! No nerf metadata for mesh coordinate conversion!")
        verts -= verts.mean(0, keepdim=True)
        verts /= verts.max()
        print('normalized', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

    # Create a Meshes object
    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
    )       
    print('created mesh without texture')
    
    if save_mesh:
        IO = pytorch3d.io.IO()
        IO.save_mesh(mesh, f'{args.save_dir}/norm.obj')
    return mesh

def load_nerf_metadata(args, metadata_dir, filename, verbose=True):
    """docstring for load_nerf_metadata"""
    coordinates = torch.load(os.path.join(metadata_dir, 'coordinates.pt'))
    rgb = cv2.imread(os.path.join(metadata_dir, args.split, 'rgbs/%s.jpg' % filename))
    depth = np.array([])
    metadata = torch.load(os.path.join(metadata_dir, args.split, 'metadata/%s.pt' % filename))

    root = ET.parse(args.metadataXml_path).getroot()
    # Robustly find SRSOrigin, default to 0 if not found.
    srs_origin_element = root.find('SRSOrigin')
    if srs_origin_element is not None and srs_origin_element.text:
        translation = np.array(srs_origin_element.text.split(','), dtype=float)
        if verbose: print("Found <SRSOrigin> in XML, using it as translation.")
    else:
        print("WARNING: <SRSOrigin> tag not found in XML. Assuming mesh vertices are in absolute coordinates (translation set to zero).")
        translation = np.array([0.0, 0.0, 0.0], dtype=float)

    if verbose:
        print('rgb', rgb.shape)
        print('depth', depth.shape)
        print('meta data', metadata)
        print(coordinates)

    nerf_metadata = {'coordinates': coordinates, 'rgb': rgb, 'depth': depth,
                     'metadata': metadata, 'translation': translation}
    return nerf_metadata

def convert_verts_to_nerf_space(verts, nerf_metadata):
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
        R = torch.stack([-R[:, 0], R[:, 1], -R[:, 2]], 1)
        new_c2w = torch.cat([R, T], 1)
        if verbose: print('c2w\n', new_c2w)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
        R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3]
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
        image_size = args.image_size

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


import glob
from tqdm import tqdm


def _get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nerf_metadata_dir', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan2', help='')
    parser.add_argument('--obj_filename', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/original_image/terra_obj/Block/Block.obj', help='')
    parser.add_argument('--save_dir', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan2', help='')
    parser.add_argument('--metadataXml_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/original_image/terra_obj/metadata.xml', help='')
    parser.add_argument('--down', default=1, type=float, help='')
    parser.add_argument('--camera_type', default='perspective', help='')
    parser.add_argument('--save_mesh', default=False, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--split', default='val', help='')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--simplify', default=1, type=int)
    return parser.parse_known_args()[0]

def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(save_dir, 'cat_results_' + args.split), exist_ok=True)
        # Create a new directory for the requested visualization
        os.makedirs(os.path.join(save_dir, 'altitude_depth_vis_' + args.split), exist_ok=True)
        
    os.makedirs(os.path.join(save_dir, args.split, 'depth_mesh'), exist_ok=True)

    nerf_metadata_dir = args.nerf_metadata_dir    

    # Load mesh once
    name = sorted(glob.glob(os.path.join(nerf_metadata_dir, args.split, 'rgbs/*.jpg')))[0]
    filename = os.path.basename(name)[:-4]
    nerf_metadata = load_nerf_metadata(args, nerf_metadata_dir, filename) 

    if args.debug:
        mesh = load_mesh_debug(args.obj_filename, nerf_metadata, save_mesh=args.save_mesh)      
    else:
        mesh = load_mesh(args, args.obj_filename, nerf_metadata, save_mesh=args.save_mesh)

    print("**********************************")
    print(f"Now rendering {args.split} set")
    print("**********************************")
    names = sorted(glob.glob(os.path.join(nerf_metadata_dir, args.split, 'rgbs/*.jpg')))
    for name in tqdm(names[:]):
        filename = os.path.basename(name)[:-4]
        nerf_metadata = load_nerf_metadata(args, nerf_metadata_dir, filename, verbose=False)
        
        render_setup = setup_renderer(args, nerf_metadata, verbose=False)

        ## Render the mesh
        renderer = render_setup['renderer']
        images, fragments = renderer(mesh)

        depth = fragments.zbuf[0, :, :, 0].cpu().numpy()

        # 从元数据中获取尺度恢复因子
        pose_scale_factor = nerf_metadata['coordinates']['pose_scale_factor']

        # 将有效的深度值乘以尺度因子，恢复到真实物理尺度（米）
        valid_mask = depth != -1
        depth[valid_mask] *= pose_scale_factor
        # Use a large number for invalid depth before saving
        depth[depth == -1] = 1e6

        if args.simplify == 1:
            np.save(os.path.join(save_dir, args.split, 'depth_mesh', f'{filename}.npy'), depth[:,:,None].astype(np.float32))
        
        # --- Visualization Logic ---
        if args.visualize:
            # Set invalid depth back to -1 for visualization processing
            depth[depth == 1e6] = -1

            # --- Your Original Visualization ---
            rgb_np = images[0, ..., :3].cpu().numpy() # This is a grayscale depth image
            rgb_np_uint8 = (rgb_np.clip(0, 1) * 255).astype(np.uint8)
            rgb_np_color = cv2.cvtColor(rgb_np_uint8, cv2.COLOR_GRAY2BGR)

            depth_color, _ = np_depth_to_colormap(depth, normalize=False)
            
            # For comparison, let's just use the rendered depth as "lidar depth"
            lidar_depth_color, _ = np_depth_to_colormap(depth, normalize=False)
            
            ori_image = nerf_metadata['rgb']
            ori_image = cv2.resize(ori_image, (rgb_np.shape[1], rgb_np.shape[0]))

            results_1 = np.concatenate([rgb_np_color, depth_color], 1)
            results_2 = np.concatenate([ori_image, lidar_depth_color], 1)
            cat_results = np.concatenate([results_1, results_2], 0)
            cv2.imwrite(os.path.join(save_dir, 'cat_results_' + args.split, f'{filename}_cat_results.jpg'), cat_results.astype(np.uint8))
            
            # ==========================================================================================
            # ======================== NEW VISUALIZATION LOGIC STARTS HERE ===============================
            # ==========================================================================================
            
            valid_depth = depth[depth > -0.9]
            if valid_depth.size > 0:
                # 1. Get altitude from metadata and annotate the original image
                c2w = nerf_metadata['metadata']['c2w']
                altitude = c2w[2, 3].item()  # Camera's Z position in world coordinates
                image_with_text = ori_image.copy()
                altitude_text = f"Altitude: {altitude:.2f}m"
                
                # Add a black background for better text readability
                (text_width, text_height), baseline = cv2.getTextSize(altitude_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                cv2.rectangle(image_with_text, (10, 10), (20 + text_width, 20 + text_height + baseline), (0,0,0), -1)
                cv2.putText(image_with_text, altitude_text, (15, 20 + text_height), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

                # 2. Prepare the depth map and its color bar
                # The colormap in np_depth_to_colormap uses percentiles, so the color bar should reflect that range.
                min_d, max_d = np.percentile(valid_depth, [5, 95])
                
                # Handle case where depth is constant
                if max_d <= min_d: max_d = min_d + 1.0

                colorbar = create_depth_colorbar(height=depth_color.shape[0], width=120, min_val=min_d, max_val=max_d)
                
                # 3. Combine the images into the new visualization format
                # Left: original image with altitude. Right: depth map + color bar
                new_visualization = np.concatenate([image_with_text, depth_color, colorbar], axis=1)

                # 4. Save the new visualization
                new_vis_dir = os.path.join(save_dir, 'altitude_depth_vis_' + args.split)
                save_path = os.path.join(new_vis_dir, f'{filename}_comparison.jpg')
                cv2.imwrite(save_path, new_visualization)
            # ==========================================================================================
            # ========================== NEW VISUALIZATION LOGIC ENDS HERE =============================
            # ==========================================================================================

if __name__ == '__main__':
    main(_get_opts())

'''
python 3_render_mesh_depthcc-vis.py --obj_filename /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/Mesh.obj --nerf_metadata_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7 --metadataXml_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/BlocksExchangeUndistortAT.xml --save_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7 --split "val" --down 4.0



'''
