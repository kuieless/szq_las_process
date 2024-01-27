import argparse
import os, glob
import sys
import torch
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import exifread
import math
from tqdm import tqdm
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
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    TexturesAtlas,
)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def convert_verts_to_nerf_space(verts, nerf_metadata):
    """docstring for convert_mesh_to_nerf_space"""
    origin_drb = nerf_metadata['coordinates']['origin_drb'].to(device).to(torch.float64)
    pose_scale_factor = torch.tensor(nerf_metadata['coordinates']['pose_scale_factor'].item()).to(torch.float64)
    # translation = torch.from_numpy(nerf_metadata['translation']).float().to(device)

    def rad(x):
        return math.radians(x)

    ZYQ = torch.DoubleTensor([[0, 0, -1],
                             [0, 1, 0],
                             [1, 0, 0]]).to(device)
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                              [0, math.cos(rad(135)), math.sin(rad(135))],
                              [0, -math.sin(rad(135)), math.cos(rad(135))]]).to(device)

    # verts += translation
    verts = verts.to(torch.float64)

    # MRM = compute_MRM(verts.cpu().numpy())
    # MRM = torch.FloatTensor(MRM).to(device)
    # verts_nerf = (MRM @ verts.T).T

    verts_nerf = ZYQ @ verts.T
    verts_nerf = (ZYQ_1 @ verts_nerf).T

    verts_nerf = (verts_nerf - origin_drb) / pose_scale_factor #     c2w[:,:, 3] = (c2w[:,:, 3] - origin) / scale
    print("origin:{}  scale:{}".format(origin_drb, pose_scale_factor))
    return verts_nerf.to(torch.float32)


def load_mesh(args, obj_filename, nerf_metadata=None, save_mesh=False):
    """docstring for load_mesh"""

    # Load obj file
    #mesh = load_objs_as_meshes([obj_filename], device=device)
    verts, faces, aux = load_obj(obj_filename, device=device
                                )

    print('loaded obj')
    atlas = aux.texture_atlas
    
    print('verts before', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

    if nerf_metadata is not None:
        verts = convert_verts_to_nerf_space(verts, nerf_metadata)
        print('Convert tot NeRF', verts.max(0)[0], verts.min(0)[0], verts.mean(0))
    else:
        """ normalize """
        verts -= verts.mean(0, keepdim=True)
        verts /= verts.max()
        print('normalized', verts.max(0)[0], verts.min(0)[0], verts.mean(0))

    mesh = Meshes(
        verts=[verts],
        faces=[faces.verts_idx],
    )        
    print('create mesh', mesh)
    print(mesh)
    
    if save_mesh:
        IO = pytorch3d.io.IO()

        IO.save_mesh(mesh, f'{args.save_dir}/norm.obj')
    return mesh


def load_nerf_metadata(args, metadata_dir, filename, verbose=True):
    """docstring for load_nerf_metadata"""

    coordinates = torch.load(os.path.join(metadata_dir, 'coordinates.pt'))
    rgb = cv2.imread(os.path.join(metadata_dir, args.split, 'rgbs/%s.jpg' % filename))
    # depth = np.load(os.path.join(metadata_dir, args.split, 'depth_dji/%s.npy' % filename))
    depth = np.array([])
    metadata = torch.load(os.path.join(metadata_dir, args.split, 'metadata/%s.pt' % filename))

    # root = ET.parse(args.metadataXml_path).getroot()
    # translation = np.array(root.find('SRSOrigin').text.split(',')).astype(np.float)

    if verbose:
        print('rgb', rgb.shape)
        print('depth', depth.shape)
        print('meta data', metadata)
        print(coordinates)

    nerf_metadata = {'coordinates': coordinates, 'rgb': rgb, 'depth': depth,
            'metadata': metadata}
    return nerf_metadata



def setup_renderer(args, nerf_metadata, verbose=True):
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
    #R, T = look_at_view_transform(0.8, 0, 0) 

    if args.camera_type == 'perspective' and nerf_metadata is not None:
        c2w = nerf_metadata['metadata']['c2w']
        R, T = c2w[:3, :3], c2w[:3, 3:]

        if verbose:
            print('origin c2w\n', torch.cat([R, T], 1))
        R = torch.stack([-R[:, 0], R[:, 1], -R[:, 2]], 1)
        new_c2w = torch.cat([R, T], 1)
        if verbose:
            print('c2w\n', new_c2w)
        w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
        R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3]
        if verbose:
            print('w2c\n', w2c)

        R = R[None]
        T = T[None]

        H, W = nerf_metadata['metadata']['H'], nerf_metadata['metadata']['W']
        H, W = int(H / args.down / args.simplify), int(W / args.down / args.simplify)

        intrinsics = nerf_metadata['metadata']['intrinsics'] / args.down / args.simplify

        image_size = ((H, W),)    # (h, w)
        fcl_screen = ((intrinsics[0], intrinsics[1]),)  # fcl_ndc * min(image_size) / 2
        prp_screen = ((intrinsics[2], intrinsics[3]), )  # w / 2 - px_ndc * min(image_size) / 2, h / 2 - py_ndc * min(image_size) / 2
        cameras = PerspectiveCameras(focal_length=fcl_screen, principal_point=prp_screen, in_ndc=False, image_size=image_size, R=R, T=T, device=device)
        #cameras = FoVPerspectiveCameras(fov=32, device=device, R=R, T=T)
        
        if verbose:
            print(cameras)
        image_size = (H, W)

    elif args.camera_type == 'fovperspective':
        R, T = look_at_view_transform(2.7, 0, 180) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        image_size = args.image_size
    # print('camera-center\n', cameras.get_camera_center())
    # print('world-to-view\n', cameras.get_world_to_view_transform().get_matrix())
    # print('full-projection\n', cameras.get_full_projection_transform().get_matrix())
    # print('ndc-transform\n', cameras.get_ndc_camera_transform().get_matrix())
    #print('Camera R T'); print(R); print(T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 

    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        perspective_correct=True,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = AmbientLights(device=device)
    #lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRendererWithFragments(
        rasterizer = rasterizer,
        # shader=SoftPhongShader(
        #     device=device, 
        #     cameras=cameras,
        #     lights=lights
        # )
        shader = SoftSilhouetteShader()
    )
    render_setup =  {'cameras': cameras, 'raster_settings': raster_settings, 'lights': lights,
            'rasterizer': rasterizer, 'renderer': renderer}

    return render_setup


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
                depth_normalized, [min_depth_percentile, max_depth_percentile])
            depth_normalized[depth_normalized < min_depth] = min_depth
            depth_normalized[depth_normalized > max_depth] = max_depth
            depth_normalized = depth_normalized / max_depth
        else:
            depth_normalized[valid_mask] = d_valid

        depth_np = (depth_normalized * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
        depth_normalized = depth_normalized
    else:
        print('!!!! No depth projected !!!')
        depth_color = depth_normalized = np.zeros((*depth.shape,3), dtype=np.uint8)
    return depth_color, depth_normalized


def main(args):
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir+'/cat_results_'+ args.split) and args.visualize:
        os.makedirs(save_dir+'/cat_results_'+ args.split)
    if not os.path.exists(save_dir+ '/' + args.split +'/depth_mesh'):
        os.makedirs(save_dir+ '/' + args.split +'/depth_mesh')

    nerf_metadata_dir = args.nerf_metadata_dir    

    name = sorted(glob.glob(os.path.join(nerf_metadata_dir, args.split, 'rgbs/*.jpg')))[0]
    filename = os.path.basename(name)[:-4]

    # 加载meta
    nerf_metadata = load_nerf_metadata(args, nerf_metadata_dir, filename) 
    # 加载obj，并根据meta归一化obj
    mesh = load_mesh(args, args.obj_filename, nerf_metadata, save_mesh=args.save_mesh)

    print("**********************************")
    print("now rendering {} set".format(args.split))
    print("Attention: please confirm whether you need to rendering the other dataset!!")
    print("**********************************")
    names = sorted(glob.glob(os.path.join(nerf_metadata_dir, args.split, 'rgbs/*.jpg')))
    for name in tqdm(names[:]):
        filename = os.path.basename(name)[:-4]
        # print('Processing %s' % filename)
        nerf_metadata = load_nerf_metadata(args, nerf_metadata_dir, filename, verbose=False)
        
        render_setup = setup_renderer(args, nerf_metadata, verbose=False)

        ## 3. Render the mesh
        renderer = render_setup['renderer']
        images, fragments = renderer(mesh)
        rgb_np = images[0, ..., :3].cpu().numpy()[:, :, ::-1]

        # save_name = os.path.basename(args.obj_filename)[:-4] + '_%s' % filename
        #cv2.imwrite(os.path.join(save_dir, '%s_mesh.jpg' % filename), (rgb_np * 255).astype(np.uint8))

        #depth = renderer_depth(mesh)
        depth = fragments.zbuf[0, :, :, 0].cpu().numpy()
        depth[(depth==-1)] = 1e6

        if args.simplify == 1:
            np.save(os.path.join(save_dir+ '/' + args.split + '/depth_mesh', '%s.npy' % filename), depth[:,:,None].astype(np.float16))
        


        ############# 以下代码是可视化cat_results ######################
        #           第一行：mesh渲染的图像,depth
        #           第二行拍摄的图像, lidar depth
        #           第三行：color errormap, depth errormap
    
        if args.visualize:
            depth[(depth==1e6)] = -1
            depth_color, depth_normalized = np_depth_to_colormap(depth, normalize=False)

            # get lidar depth
            # lidar_depth = nerf_metadata['depth'].copy()[:, :, 0]
            lidar_depth = depth
            #print(lidar_depth.shape)
            lidar_depth[lidar_depth == 1e6] = -1
            lidar_depth_color, lidar_depth_normalized = np_depth_to_colormap(lidar_depth, normalize=False)
            # depth_v2 = render_setup['renderer_depth'](mesh) # not working

            # compute difference between mesh depth and lidar depth
            #print(depth.shape, depth.max(), depth.min())
            #print(lidar_depth.shape, lidar_depth.max(), lidar_depth.min())
            depth_diff = np.zeros(depth.shape)
            gt_valid = lidar_depth > -0.9
            valid_diff = np.abs(depth[gt_valid] - lidar_depth[gt_valid])
            #valid_diff = (valid_diff - valid_diff.min()) / (valid_diff.max() - valid_diff.min())
            valid_diff = valid_diff.clip(0, 0.01) / 0.01
            depth_diff[gt_valid] = valid_diff
            depth_diff_color = cv2.applyColorMap((depth_diff*255).astype(np.uint8), cv2.COLORMAP_JET)

            # Capture image
            ori_image = nerf_metadata['rgb']
            ori_image = cv2.resize(ori_image, (rgb_np.shape[1], rgb_np.shape[0]))

            image_diff = np.abs(ori_image / 255.0 - rgb_np).mean(2) # .clip(0.2) / 0.2
            image_diff_color = cv2.applyColorMap((image_diff*255).astype(np.uint8), cv2.COLORMAP_JET)

            results_1 = np.concatenate([rgb_np * 255, depth_color], 1)
            results_2 = np.concatenate([ori_image, lidar_depth_color], 1)
            results_3 = np.concatenate([image_diff_color, depth_diff_color], 1)

            #results = [rgb_np, depth[:, :, None].repeat(3, 2), ori_image, lidar_depth[:, :, None].repeat(3, 2)]
            cat_results = np.concatenate([results_1,results_2, results_3], 0)
            #print(cat_results.shape)
            #cv2.imwrite(os.path.join(save_dir, '%s_depth_color.jpg' % (filename)), (depth_color).astype(np.uint8))
            #cv2.imwrite(os.path.join(save_dir, '%s_depth.jpg' % (filename)), (depth_normalized).astype(np.uint8))
            cv2.imwrite(os.path.join(save_dir, 'cat_results_%s' % args.split, '%s_cat_results.jpg' % filename), cat_results.astype(np.uint8))
            # print('Saved %s' % os.path.join(save_dir, '%s.jpg' % filename))

def _get_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--nerf_metadata_dir', default='/data/yuqi/Datasets/InstanceBuilding/3D/scene1/output', help='')
    parser.add_argument('--obj_filename', default='/data/yuqi/Datasets/InstanceBuilding/3D/scene1/scene1.obj', help='')
    parser.add_argument('--save_dir', default='/data/yuqi/jx_rebuttal/IB_test', help='')
    # parser.add_argument('--metadataXml_path', default='/data/jxchen/dataset/dji/campus/metadata.xml', help='')
    parser.add_argument('--down', default=4, type=float, help='')
    parser.add_argument('--camera_type', default='perspective', help='')
    parser.add_argument('--save_mesh', default=False, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--split', default='train', help='')
    # parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--simplify', default=1, type=int)
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    main(_get_opts())    