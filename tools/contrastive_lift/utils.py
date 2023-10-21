from mega_nerf.misc_utils import main_print, main_tqdm
import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path
from torchvision.utils import make_grid

# from contrastive-lift
from sklearn.cluster import MeanShift
from scipy.stats import gaussian_kde
# from hdbscan import HDBSCAN
import time
from tools.contrastive_lift.util.distinct_colors import DistinctColors
from tools.contrastive_lift.util.misc import visualize_depth, probability_to_normalized_entropy, get_boundary_mask


def create_instances_from_semantics(instances, semantics, thing_classes, device):
    stuff_mask = ~torch.isin(semantics, torch.tensor(thing_classes).to(semantics.device))
    padded_instances = torch.ones((instances.shape[0], instances.shape[1] + 1), device=instances.device) * -float('inf')    
    padded_instances[:, 1:] = instances
    padded_instances[stuff_mask, 0] = float('inf')
    return padded_instances


# copy from contrastive-lift

def cluster(all_thing_features, bandwidth, device, num_images=None, use_dbscan=False,
            use_silverman=False, cluster_size=500, num_points=50000):
    thing_mask = all_thing_features[...,0] == -float('inf')
    features = all_thing_features[thing_mask]
    features = features[:,1:]
    all_thing_features = all_thing_features[:,1:]
    
    # remove outliers assuming Gaussian distribution
    centmean, centstd = features.mean(axis=0), features.std(axis=0)
    outlier_mask = np.all(np.abs(features - centmean) < 3 * centstd, axis=1)
    centers_filtered = features[outlier_mask]
    print("Num centers pre-filtering: ", features.shape[0], features.min(axis=0), features.max(axis=0))
    print("Num centers post-filtering: ", centers_filtered.shape[0], centers_filtered.min(axis=0), centers_filtered.max(axis=0))
    rescaling_bias = centers_filtered.min(axis=0)
    rescaling_factor = 1/(centers_filtered.max(axis=0) - centers_filtered.min(axis=0))
    centers_rescaled = (centers_filtered - rescaling_bias) * rescaling_factor
    # perform clustering
    fps_points_indices = np.random.choice(centers_rescaled.shape[0], num_points, replace=False)

    fps_points_rescaled = centers_rescaled[fps_points_indices]
    
    if not use_dbscan:
        t1_ms = time.time()
        if use_silverman:
            kde = gaussian_kde(fps_points_rescaled.T, bw_method='silverman')
            bandwidth_ = kde.covariance_factor()
            print("Using Silverman bandwidth: ", bandwidth_)
        else:
            bandwidth_ = bandwidth
        clustering = MeanShift(bandwidth=bandwidth_, cluster_all=False, bin_seeding=True,
                               min_bin_freq=10).fit(fps_points_rescaled)
        t2_ms = time.time()
        print(f"MeanShift took {t2_ms-t1_ms} seconds")
        labels = clustering.labels_
        centroids = clustering.cluster_centers_
        all_labels = clustering.predict(
            (all_thing_features.reshape(-1, all_thing_features.shape[-1]) - rescaling_bias) * rescaling_factor
        )
    # else: # Use HDBSCAN
    #     t1_dbscan = time.time()
    #     clusterer = HDBSCAN(min_cluster_size=cluster_size, min_samples=1, prediction_data=True,
    #                                 allow_single_cluster=True).fit(fps_points_rescaled)
    #     t2_dbscan = time.time()
    #     print(f"HDBSCAN took {t2_dbscan-t1_dbscan} seconds")
    #     labels = clusterer.labels_
    #     centroids = np.stack([clusterer.weighted_cluster_centroid(cluster_id=cluster_id) \
    #                           for cluster_id in np.unique(labels) if cluster_id != -1])
    #     distances = torch.zeros((all_thing_features.shape[0], centroids.shape[0]), device=device)
    #     chunksize = 10**7
    #     all_thing_features_rescaled = (all_thing_features.reshape(-1, all_thing_features.shape[-1]) - rescaling_bias) * rescaling_factor
    #     for i in range(0, all_thing_features.shape[0], chunksize):
    #         distances[i:i+chunksize] = torch.cdist(
    #             torch.FloatTensor(all_thing_features_rescaled[i:i+chunksize]).to(device),
    #             torch.FloatTensor(centroids).to(device)
    #         )
    #     all_labels = torch.argmin(distances, dim=-1).cpu().numpy()

    all_labels[~thing_mask] = -1
    # to one hot
    all_labels = all_labels + 1 # -1,0,...,K-1 -> 0,1,...,K
    all_labels_onehot = np.zeros((all_labels.shape[0], centroids.shape[0]+1))
    all_labels_onehot[np.arange(all_labels.shape[0]), all_labels] = 1
    all_points_instances = torch.from_numpy(all_labels_onehot).view(num_images, -1, centroids.shape[0]+1).to(device)
    return all_points_instances


def visualize_panoptic_outputs(p_rgb, p_semantics, p_instances, p_depth, rgb, semantics, instances, H, W, thing_classes, visualize_entropy=True,
                               m2f_semantics=None, m2f_instances=None):
    alpha = 0.65
    distinct_colors = DistinctColors()
    img = p_rgb.view(H, W, 3).cpu()
    img = torch.clamp(img, 0, 1).permute(2, 0, 1)
    if visualize_entropy:
        img_sem_entropy = visualize_depth(probability_to_normalized_entropy(torch.nn.functional.softmax(p_semantics, dim=-1)).reshape(H, W), minval=0.0, maxval=1.00, use_global_norm=True)
    else:
        img_sem_entropy = torch.zeros_like(img)
    if p_depth is not None:
        depth = visualize_depth(p_depth.view(H, W))
    else:
        depth = torch.zeros_like(img)
    if len(p_instances.shape) > 1:
        p_instances = p_instances.argmax(dim=1)
    if len(p_semantics.shape) > 1:
        p_semantics = p_semantics.argmax(dim=1)
    img_semantics = distinct_colors.apply_colors_fast_torch(p_semantics.cpu()).view(H, W, 3).permute(2, 0, 1) * alpha + img * (1 - alpha)
    boundaries_img_semantics = get_boundary_mask(p_semantics.cpu().view(H, W))
    img_semantics[:, boundaries_img_semantics > 0] = 0
    colored_img_instance = distinct_colors.apply_colors_fast_torch(p_instances.cpu()).float()
    boundaries_img_instances = get_boundary_mask(p_instances.cpu().view(H, W))
    colored_img_instance[boundaries_img_instances.reshape(-1) > 0, :] = 0
    thing_mask = torch.logical_not(sum(p_semantics == s for s in thing_classes).bool()).cpu()
    colored_img_instance[thing_mask, :] = p_rgb.cpu()[thing_mask, :]
    img_instances = colored_img_instance.view(H, W, 3).permute(2, 0, 1) * alpha + img * (1 - alpha)
    if rgb is not None and semantics is not None and instances is not None:
        semantics = semantics.to(torch.int64)
        instances = instances.to(torch.int64)

        img_gt = rgb.view(H, W, 3).permute(2, 0, 1).cpu()
        img_semantics_gt = distinct_colors.apply_colors_fast_torch(semantics.cpu()).view(H, W, 3).permute(2, 0, 1) * alpha + img_gt * (1 - alpha)
        boundaries_img_semantics_gt = get_boundary_mask(semantics.cpu().view(H, W))
        img_semantics_gt[:, boundaries_img_semantics_gt > 0] = 0
        colored_img_instance_gt = distinct_colors.apply_colors_fast_torch(instances.cpu()).float()
        boundaries_img_instances_gt = get_boundary_mask(instances.cpu().view(H, W))
        colored_img_instance_gt[instances.cpu() == 0, :] = rgb.cpu()[instances.cpu() == 0, :]
        img_instances_gt = colored_img_instance_gt.view(H, W, 3).permute(2, 0, 1) * alpha + img_gt * (1 - alpha)
        img_instances_gt[:, boundaries_img_instances_gt > 0] = 0
        # stack = torch.cat([torch.stack([img_gt, img_semantics_gt, img_instances_gt, torch.zeros_like(img_gt), torch.zeros_like(img_gt)]), torch.stack([img, img_semantics, img_instances, depth, img_sem_entropy])], dim=0)
        stack = torch.cat([torch.stack([img_gt, img_semantics_gt, img_instances_gt]), torch.stack([img, img_semantics, img_instances])], dim=0)
        if m2f_semantics is not None and m2f_instances is not None:
            img_semantics_m2f = distinct_colors.apply_colors_fast_torch(m2f_semantics.cpu()).view(H, W, 3).permute(2, 0, 1) * alpha + img_gt * (1 - alpha)
            boundaries_img_semantics_m2f = get_boundary_mask(m2f_semantics.cpu().view(H, W))
            img_semantics_m2f[:, boundaries_img_semantics_m2f > 0] = 0
            colored_img_instance_m2f = distinct_colors.apply_colors_fast_torch(m2f_instances.cpu()).float()
            boundaries_img_instances_m2f = get_boundary_mask(m2f_instances.cpu().view(H, W))
            colored_img_instance_m2f[m2f_instances.cpu() == 0, :] = rgb.cpu()[m2f_instances.cpu() == 0, :]
            img_instances_m2f = colored_img_instance_m2f.view(H, W, 3).permute(2, 0, 1) * alpha + img_gt * (1 - alpha)
            img_instances_m2f[:, boundaries_img_instances_m2f > 0] = 0
            stack = torch.cat([stack[0:5], torch.stack([torch.zeros_like(img_gt), img_semantics_m2f, img_instances_m2f, torch.zeros_like(img_gt), torch.zeros_like(img_gt)]), stack[5:]], dim=0)
    else:
        # stack = torch.stack([img, img_semantics, img_instances, depth, img_sem_entropy])
        stack = torch.stack([img, img_semantics, img_instances])

    return stack
