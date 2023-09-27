
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

    # Specify the paths to the mesh and point cloud data
    mesh_path = "data/mesh_trans.obj"
    pointcloud_path = 'data/data8_new_visualize_label.ply'



    # Load the per-point class labels (assuming it's a 1D array/list)
    # labels = [0, 1, 1, 2, 0, ...]

    # Specify the output path for the segmented mesh
    output_path = 'segmented_mesh.obj'

    # Load
    mesh = trimesh.load(mesh_path)
    root = ET.parse('data/metadata.xml').getroot()
    translation = np.array(root.find('SRSOrigin').text.split(',')).astype(np.float)
    # mesh.vertices = mesh.vertices + translation

    # pointcloud = np.loadtxt(pointcloud_path)
    ply_data = PlyData.read(pointcloud_path)
    vertex_data = ply_data['vertex'][::10]
    xyz = []
    rgb = []

    for vertex in tqdm(vertex_data):
        xyz.append([vertex['x'], vertex['y'], vertex['z']])
        rgb.append([vertex['red'], vertex['green'], vertex['blue']])
    
    pointcloud = np.array(xyz)
    labels = np.array(rgb)



    # Segment the mesh and export it
    segment_mesh(mesh, pointcloud, labels, output_path)
    