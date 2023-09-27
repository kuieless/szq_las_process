
import numpy as np
from plyfile import PlyData, PlyElement
import trimesh
import xml.etree.ElementTree as ET
from tqdm import tqdm
from scipy.spatial import cKDTree


def segment_mesh(mesh, pointcloud, labels, output_path):
    
    

    
    # Build a KDTree for fast nearest neighbor search
    kdtree = cKDTree(pointcloud[:, :3])
    

    # Initialize the mesh segmentation
    mesh_segmentation = np.zeros((len(mesh.faces),3), dtype=int)
    

    # Segment the mesh based on per-point class labels
    for i, face in enumerate(mesh.faces):
        # Find the nearest point in the point cloud
        vertex_coords = mesh.vertices[face]
        _, idx = kdtree.query(vertex_coords.mean(axis=0))
        
        # Assign the class label to the mesh face
        mesh_segmentation[i] = labels[idx]
    
    # Create a new mesh with per-face color information
    segmented_mesh = trimesh.Trimesh(vertices=mesh.vertices,
                                     faces=mesh.faces,
                                     vertex_colors=mesh_segmentation)
    
    # Export the segmented mesh as an OBJ file
    segmented_mesh.export(output_path)




if __name__ == '__main__':
    


    # 1. 读取点云和OBJ文件，提取颜色和顶点数据
    point_cloud_file = 'data/data8_new_visualize_label.ply'  # 点云文件路径
    mesh_path = 'data/mesh_trans.obj'  # OBJ文件路径

    # 读取点云数据
    ply_data = PlyData.read(point_cloud_file)
    point_cloud_xyz = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T
    point_cloud_rgb = np.vstack([ply_data['vertex']['red'], ply_data['vertex']['green'], ply_data['vertex']['blue']]).T

    # 读取OBJ文件
    mesh = trimesh.load_mesh(mesh_path)


    # root = ET.parse('data/metadata.xml').getroot()
    # translation = np.array(root.find('SRSOrigin').text.split(',')).astype(np.float)

    mesh_vertices = np.array(mesh.vertices)
    point_cloud_kd_tree = cKDTree(point_cloud_xyz)
    mesh_colors = []
    for vertex in tqdm(mesh_vertices):
        _, index = point_cloud_kd_tree.query(vertex)
        mesh_colors.append(point_cloud_rgb[index])

    # 将Mesh的颜色数据应用到Mesh上
    mesh.vertex_colors = mesh_colors
    mesh.visual = trimesh.visual.color.ColorVisuals(vertex_colors=mesh_colors)

        # 将具有颜色的Mesh保存为新的OBJ文件
    output_obj_file_path = 'segmented_mesh.obj'

    mesh.export(output_obj_file_path)


    