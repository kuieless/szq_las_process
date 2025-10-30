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
import glob  # <--- ADD THIS LINE
from tqdm import tqdm # <--- ADD THIS LINE
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
    bar = np.linspace(0, 1, height).reshape(height, 1)
    bar_uint8 = (bar * 255).astype(np.uint8)
    color_bar_img = cv2.applyColorMap(255 - bar_uint8, colormap)
    color_bar_img = cv2.repeat(color_bar_img, 1, width)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_color = (255, 255, 255)
    cv2.putText(color_bar_img, "Depth (m)", (5, 30), font, 0.8, font_color, 2, cv2.LINE_AA)
    num_labels = 7
    for i in range(num_labels):
        p = i / (num_labels - 1)
        y = int(p * (height - 60)) + 30
        val = max_val * (1 - p) + min_val * p
        cv2.putText(color_bar_img, f"{val:.1f}", (10, y + 8), font, 0.6, font_color, 2, cv2.LINE_AA)
    return color_bar_img
# ==========================================================================================
# ================================= END OF NEW FUNCTION ====================================
# ==========================================================================================

def np_depth_to_colormap(depth, normalize=True, thres=-1, min_depth_percentile=5, max_depth_percentile=95):
    """ depth: [H, W] """
    depth_normalized = np.zeros(depth.shape)
    valid_mask = depth > -0.9
    if valid_mask.sum() > 0:
        d_valid = depth[valid_mask]
        if normalize:
            depth_normalized[valid_mask] = (d_valid - d_valid.min()) / (d_valid.max() - d_valid.min())
        elif thres > 0:
            depth_normalized[valid_mask] = d_valid.clip(0, thres) / thres
        elif min_depth_percentile > 0:
            depth_normalized[valid_mask] = d_valid
            min_depth, max_depth = np.percentile(d_valid, [min_depth_percentile, max_depth_percentile])
            depth_normalized[valid_mask & (depth < min_depth)] = min_depth
            depth_normalized[valid_mask & (depth > max_depth)] = max_depth
            if max_depth > min_depth:
                depth_normalized[valid_mask] = (depth_normalized[valid_mask] - min_depth) / (max_depth - min_depth)
            else:
                depth_normalized[valid_mask] = 0
        else:
            depth_normalized[valid_mask] = d_valid
        depth_np = (depth_normalized * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
    else:
        print('!!!! No depth projected !!!')
        depth_color = np.zeros((*depth.shape, 3), dtype=np.uint8)
    return depth_color, depth_normalized

# ==========================================================================================
# ======================== MODIFIED load_mesh FUNCTION =====================================
# ==========================================================================================
def load_mesh(args, obj_filename, nerf_metadata=None, save_mesh=False):
    verts, faces, aux = load_obj(obj_filename, device=device,
                                   load_textures=False,
                                   create_texture_atlas=False)
    print(f'Loaded obj with {verts.shape[0]} vertices.')
    print(f'Initial mesh center: {verts.mean(0).cpu().numpy()}')

    # --- Automatic Alignment using SRSOrigin from metadata.xml ---
    if args.srs_metadata_path:
        if os.path.exists(args.srs_metadata_path):
            print(f"Reading SRSOrigin from: {args.srs_metadata_path}")
            tree = ET.parse(args.srs_metadata_path)
            root = tree.getroot()
            srs_origin_element = root.find('SRSOrigin')
            if srs_origin_element is not None:
                srs_origin = np.fromstring(srs_origin_element.text, dtype=float, sep=',')
                srs_origin_tensor = torch.tensor(srs_origin, device=device, dtype=torch.float32)
                
                # Apply the translation by subtracting the origin
                verts -= srs_origin_tensor
                
                print(f"Applied SRSOrigin translation: {-srs_origin}")
                print(f"New mesh center after alignment: {verts.mean(0).cpu().numpy()}")
            else:
                print("Warning: SRSOrigin tag not found in the provided XML file. Skipping alignment.")
        else:
            print(f"Warning: srs_metadata_path provided but file not found at '{args.srs_metadata_path}'. Skipping alignment.")
    else:
        print("Info: No srs_metadata_path provided. Assuming mesh is already in local coordinates.")
    
    print('Vertices before NeRF conversion', verts.max(0)[0].cpu().numpy(), verts.min(0)[0].cpu().numpy(), verts.mean(0).cpu().numpy())

    if nerf_metadata is not None:
        verts = convert_verts_to_nerf_space(verts, nerf_metadata)
        print('Vertices converted to NeRF space', verts.max(0)[0].cpu().numpy(), verts.min(0)[0].cpu().numpy(), verts.mean(0).cpu().numpy())
    else:
        print("Warning! No nerf metadata for mesh coordinate conversion!")
        verts -= verts.mean(0, keepdim=True)
        verts /= verts.max()

    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
    )
    print('Created mesh.')
    
    if save_mesh:
        IO = pytorch3d.io.IO()
        IO.save_mesh(mesh, f'{args.save_dir}/norm.obj')
    return mesh
# ==========================================================================================
# ================================= END OF MODIFICATION ====================================
# ==========================================================================================

def load_nerf_metadata(args, metadata_dir, filename, verbose=True):
    coordinates = torch.load(os.path.join(metadata_dir, 'coordinates.pt'))
    rgb = cv2.imread(os.path.join(metadata_dir, args.split, 'rgbs/%s.jpg' % filename))
    metadata = torch.load(os.path.join(metadata_dir, args.split, 'metadata/%s.pt' % filename))

    # This translation is for the original NeRF pipeline, we leave it as is.
    # The new mesh alignment logic is now handled separately in load_mesh.
    translation = np.array([0.0, 0.0, 0.0], dtype=float)
    if os.path.exists(args.metadataXml_path):
        root = ET.parse(args.metadataXml_path).getroot()
        srs_origin_element = root.find('SRSOrigin')
        if srs_origin_element is not None and srs_origin_element.text:
            translation = np.array(srs_origin_element.text.split(','), dtype=float)

    nerf_metadata = {'coordinates': coordinates, 'rgb': rgb, 'depth': np.array([]),
                     'metadata': metadata, 'translation': translation}
    return nerf_metadata

def convert_verts_to_nerf_space(verts, nerf_metadata):
    origin_drb = nerf_metadata['coordinates']['origin_drb'].to(device).to(torch.float64)
    pose_scale_factor = torch.tensor(nerf_metadata['coordinates']['pose_scale_factor']).to(device).to(torch.float64)
    
    # The 'translation' from load_nerf_metadata is part of the original pipeline's logic
    # for transforming camera poses, not the mesh. The mesh has already been aligned.
    # So we should NOT apply it again to the vertices here.
    # However, looking at the original code, it seems the NeRF poses might already be local,
    # and this 'translation' might be for the mesh. Let's keep the original logic for now,
    # as our new alignment step should place the verts correctly relative to the origin.
    # translation = torch.from_numpy(nerf_metadata['translation']).float().to(device)

    def rad(x): return math.radians(x)

    T1 = torch.FloatTensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).to(device).to(torch.float64)
    T2 = torch.FloatTensor([[1, 0, 0],
                            [0, math.cos(rad(135)), math.sin(rad(135))],
                            [0, -math.sin(rad(135)), math.cos(rad(135))]]).to(device).to(torch.float64)

    # Note: The original script added 'translation' here. Let's analyze if this is still needed.
    # If the NeRF cameras are centered around (0,0,0) and we've moved the mesh to be centered
    # around them, this `translation` (if it's the SRSOrigin again) should not be added.
    # The `nerf_metadata['translation']` comes from `args.metadataXml_path`, which might be different
    # from our new `srs_metadata_path`. Let's assume the original logic is for a different purpose
    # and keep it, but be aware it might need to be removed if double-translation occurs.
    # verts += translation
    verts = verts.to(torch.float64)
    verts_nerf = (T2 @ (T1 @ verts.T)).T
    verts_nerf = (verts_nerf - origin_drb) / pose_scale_factor
    return verts_nerf.to(torch.float32)

def setup_renderer(args, nerf_metadata, verbose=True):
    c2w = nerf_metadata['metadata']['c2w']
    R, T = c2w[:3, :3], c2w[:3, 3:]
    R = torch.stack([-R[:, 0], R[:, 1], -R[:, 2]], 1)
    new_c2w = torch.cat([R, T], 1)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
    R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3]
    R = R[None]
    T = T[None]
    H, W = int(nerf_metadata['metadata']['H'] / args.down), int(nerf_metadata['metadata']['W'] / args.down)
    intrinsics = nerf_metadata['metadata']['intrinsics'] / args.down
    cameras = PerspectiveCameras(focal_length=((intrinsics[0], intrinsics[1]),),
                                 principal_point=((intrinsics[2], intrinsics[3]),),
                                 in_ndc=False, image_size=((H, W),), R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1)
    lights = AmbientLights(device=device)
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardDepthShader(device=device, cameras=cameras, lights=lights)
    )
    return {'renderer': renderer}

def _get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nerf_metadata_dir', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan2', help='')
    parser.add_argument('--obj_filename', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/original_image/terra_obj/Block/Block.obj', help='')
    parser.add_argument('--save_dir', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan2', help='')
    # This is for the original pipeline's camera pose transformation
    parser.add_argument('--metadataXml_path', default='/data/yuqi/Datasets/DJI/DJI_20230726_xiayuan_lidar2/original_image/terra_obj/BlocksExchangeUndistortAT.xml', help='')
    # --- NEW ARGUMENT FOR MESH ALIGNMENT ---
    parser.add_argument('--srs_metadata_path', type=str, default=None, help='Path to metadata.xml containing the SRSOrigin for mesh alignment.')
    parser.add_argument('--down', default=4, type=float, help='')
    parser.add_argument('--split', default='val', help='')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--save_mesh', default=False, action='store_true')
    return parser.parse_known_args()[0]

def main(args):
    save_dir = args.save_dir
    os.makedirs(os.path.join(save_dir, args.split, 'depth_mesh'), exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(save_dir, 'altitude_depth_vis_' + args.split), exist_ok=True)

    nerf_metadata_dir = args.nerf_metadata_dir
    
    # Load metadata once to pass to load_mesh
    first_name = sorted(glob.glob(os.path.join(nerf_metadata_dir, args.split, 'rgbs/*.jpg')))[0]
    first_filename = os.path.basename(first_name)[:-4]
    nerf_metadata = load_nerf_metadata(args, nerf_metadata_dir, first_filename)
    
    mesh = load_mesh(args, args.obj_filename, nerf_metadata, save_mesh=args.save_mesh)

    print("**********************************")
    print(f"Now rendering {args.split} set")
    print("**********************************")
    names = sorted(glob.glob(os.path.join(nerf_metadata_dir, args.split, 'rgbs/*.jpg')))
    for name in tqdm(names[:]):
        filename = os.path.basename(name)[:-4]
        nerf_metadata = load_nerf_metadata(args, nerf_metadata_dir, filename, verbose=False)
        render_setup = setup_renderer(args, nerf_metadata, verbose=False)
        renderer = render_setup['renderer']
        images, fragments = renderer(mesh)

        depth = fragments.zbuf[0, :, :, 0].cpu().numpy()
        pose_scale_factor = nerf_metadata['coordinates']['pose_scale_factor']
        valid_mask = depth != -1
        depth[valid_mask] *= pose_scale_factor
        depth[~valid_mask] = 1e6

        np.save(os.path.join(save_dir, args.split, 'depth_mesh', f'{filename}.npy'), depth[:,:,None].astype(np.float32))
        
        if args.visualize:
            depth[depth == 1e6] = -1
            valid_depth = depth[depth > -0.9]
            if valid_depth.size > 0:
                ori_image = cv2.resize(nerf_metadata['rgb'], (depth.shape[1], depth.shape[0]))
                depth_color, _ = np_depth_to_colormap(depth, normalize=False, min_depth_percentile=5, max_depth_percentile=95)
                
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

                new_vis_dir = os.path.join(save_dir, 'altitude_depth_vis_' + args.split)
                save_path = os.path.join(new_vis_dir, f'{filename}_comparison.jpg')
                cv2.imwrite(save_path, new_visualization)

if __name__ == '__main__':
    # Cleanup unnecessary arguments and simplify
    parser = argparse.ArgumentParser()
    parser.add_argument('--nerf_metadata_dir', required=True, help='Path to the NeRF dataset directory (e.g., .../output7)')
    parser.add_argument('--obj_filename', required=True, help='Path to the .obj mesh file.')
    parser.add_argument('--save_dir', required=True, help='Directory to save the output depth maps and visualizations.')
    parser.add_argument('--srs_metadata_path', required=True, help='Path to metadata.xml containing the SRSOrigin for mesh alignment.')
    parser.add_argument('--metadataXml_path', default="/dev/null", help='Path to the original XML for NeRF camera poses (can often be ignored if not used by pose processing).')
    parser.add_argument('--down', default=1.0, type=float, help='Downsampling factor for images.')
    parser.add_argument('--split', default='val', help='Dataset split to process (e.g., train, val).')
    parser.add_argument('--visualize', action='store_true', help='Generate and save visualization images.')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false')
    parser.set_defaults(visualize=True)
    parser.add_argument('--save_mesh', action='store_true', help='Save the transformed mesh for debugging.')
    
    # Fake known_args for compatibility if other scripts import _get_opts
    class ArgsWrapper:
        def __init__(self, parsed_args):
            self.__dict__.update(vars(parsed_args))
        def parse_known_args(self):
            return self, []
    
    # Simplified main call
    main(parser.parse_args())

'''
python 3_render_mesh_depth-terr-vis.py --obj_filename /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/merged_model.obj --nerf_metadata_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7 --metadataXml_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/BlocksExchangeUndistortAT.xml --save_dir /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/output7 --split "val" --down 1.0 --visualize --srs_metadata_path /home/data1/szq/Megadepth/Aerial_lifting_early/dji/process_lidar/metadata.xml 



'''
