import struct
import os

import collections
import numpy as np
import torch
import trimesh


Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def main(seq_path, mrm=False):
    points3D_path = os.path.join(seq_path, "sparse/0/points3D.bin")
    
    points = read_points3D_binary(points3D_path)
    points_xyz = np.array([point.xyz for point in points.values()])
    points_rgb = np.array([point.rgb for point in points.values()])
    points_rgb = np.hstack((points_rgb, np.ones([points_rgb.shape[0], 1])))
    if mrm:
        mrm_path = os.path.join(seq_path, "postprocess/MRM.pt")
        MRM = torch.load(mrm_path).numpy()
        points_xyz = (MRM @ points_xyz.T).T
        pointCloud = trimesh.points.PointCloud(points_xyz, colors=points_rgb)
        print("with MRM")
        save_path = os.path.join(seq_path, "sparsePointCloud_mrm.ply")
        pointCloud.export(save_path) 
    else:
        pointCloud = trimesh.points.PointCloud(points_xyz, colors=points_rgb)
        print("without MRM")
        save_path = os.path.join(seq_path, "sparsePointCloud.ply")
        pointCloud.export(save_path)    


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

def compute_MRM(flat_vertices, normal=None):
    X_axis = np.array([1, 0, 0])
    # target_axis = compute_vert(flat_vertices)
    if normal:
        target_axis = normal
    else:
        target_axis = compute_vert_v2(flat_vertices)
    MRM = matrix_rotateB2A(X_axis, target_axis)
    return MRM


if __name__ == "__main__":
    # seq_path = "/data/yuqi/jx_rebuttal/IB_scene_1"
    seq_path = "/data/yuqi/Datasets/colmap/yingrenshi"
    # points_path = "/data/yuqi/jx_rebuttal/seq14_ds_10_val/sparse/points3D.bin"
    # mrm_path = "/data/yuqi/jx_rebuttal/seq14_ds_10_val/postprocess/MRM.pt"
    main(seq_path, mrm=False)