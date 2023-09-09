
from mega_nerf.metrics import psnr, ssim, lpips
from mega_nerf.misc_utils import main_print, main_tqdm
import torch
from tools.unetformer.uavid2rgb import custom2rgb
import numpy as np
from PIL import Image

### https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/evaluate_depth.py#L214
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3




def calculate_metric_rendering(viz_rgbs, viz_result_rgbs, train_index, wandb, writer, val_metrics, i, f, hparams, metadata_item, typ, results, device, pose_scale_factor):                            
    eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
    eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
    
    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))
    metric_key = 'val/psnr/{}'.format(train_index)
    
    if wandb is not None:
        wandb.log({'val/psnr/{}'.format(train_index): val_psnr, 'epoch': i})
    if writer is not None:
        writer.add_scalar('3_val_each_image/psnr/{}'.format(train_index), val_psnr, i)
    val_metrics['val/psnr'] += val_psnr
    main_print('The psnr of the {} image is: {}'.format(i, val_psnr))
    f.write('The psnr of the {} image is: {}\n'.format(i, val_psnr))

    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

    metric_key = 'val/ssim/{}'.format(train_index)
    # TODO: 暂时不放ssim
    if wandb is not None:
        wandb.log({'val/ssim/{}'.format(train_index): val_ssim, 'epoch':i})
    if writer is not None:
        writer.add_scalar('3_val_each_image/ssim/{}'.format(train_index), val_ssim, i)
    val_metrics['val/ssim'] += val_ssim

    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)
    for network in val_lpips_metrics:
       agg_key = 'val/lpips/{}'.format(network)
       metric_key = '{}/{}'.format(agg_key, train_index)
       # TODO: 暂时不放lpips
       # if self.wandb is not None:
       #     self.wandb.log({'val/lpips/{}/{}'.format(network, train_index): val_lpips_metrics[network], 'epoch':i})
       # if self.writer is not None:
       #     self.writer.add_scalar('3_val_each_image/lpips/{}'.format(network), val_lpips_metrics[network], i)
       val_metrics[agg_key] += val_lpips_metrics[network]


    # Depth metric
    if hparams.depth_dji_loss:
        gt_depths = metadata_item.load_depth_dji()
        valid_depth_mask = ~torch.isinf(gt_depths)
        # if hparams.depth_dji_type == 'mesh':
        #     valid_depth_mask[:,:]=False
        #     valid_depth_mask[::3]=True
        #     valid_depth_mask[gt_depths==-1] = False 


        gt_depths_valid = gt_depths[valid_depth_mask]
        
        from mega_nerf.ray_utils import get_ray_directions
        directions = get_ray_directions(metadata_item.W,
                                        metadata_item.H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        hparams.center_pixels,
                                        torch.device('cpu'))
        depth_scale = torch.abs(directions[:, :, 2])
        pred_depths = (results[f'depth_{typ}'].view(gt_depths.shape[0],gt_depths.shape[1],1)) * (depth_scale.unsqueeze(-1))
        pred_depths_valid = pred_depths[valid_depth_mask]
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt_depths_valid.view(-1).numpy(), pred_depths_valid.view(-1).numpy())
        rmse_actual = rmse * pose_scale_factor
        
        if wandb is not None:
            wandb.log({'val/depth_abs_rel/{}'.format(train_index): abs_rel, 'epoch':i})
        if writer is not None:
            writer.add_scalar('3_val_each_image/abs_rel/{}'.format(train_index), abs_rel, i)
        val_metrics['val/abs_rel'] += abs_rel

        if wandb is not None:
            wandb.log({'val/depth_rmse_actual/{}'.format(train_index): rmse_actual, 'epoch':i})
        if writer is not None:
            writer.add_scalar('3_val_each_image/rmse_actual/{}'.format(train_index), rmse_actual, i)
        val_metrics['val/rmse_actual'] += rmse_actual
        


    return val_metrics




def get_semantic_gt_pred(results, val_type, metadata_item, viz_rgbs, logits_2_label, typ, remapping, 
                         metrics_val, metrics_val_each, img_list, experiment_path_current, i, writer, hparams):
    if f'sem_map_{typ}' in results:
        sem_logits = results[f'sem_map_{typ}']
        if val_type == 'val':
                gt_label = metadata_item.load_gt()
        elif val_type == 'train':
            gt_label = metadata_item.load_label()
        if hparams.dataset_type == 'sam_project':
            pass
        else:
            gt_label = remapping(gt_label)
            sem_label = remapping(sem_label)

        gt_label_rgb = custom2rgb(gt_label.view(*viz_rgbs.shape[:-1]).cpu().numpy())
        sem_label = logits_2_label(sem_logits)
        visualize_sem = custom2rgb(sem_label.view(*viz_rgbs.shape[:-1]).cpu().numpy())
        if hparams.remove_cluster:
            ignore_cluster_index = gt_label.view(-1) * sem_label
            gt_label_ig = gt_label.view(-1)[ignore_cluster_index.nonzero()].view(-1)
            sem_label_ig = sem_label[ignore_cluster_index.nonzero()].view(-1)
            metrics_val.add_batch(gt_label_ig.cpu().numpy(), sem_label_ig.cpu().numpy())
            metrics_val_each.add_batch(gt_label.view(-1).cpu().numpy(), sem_label.cpu().numpy())
        else:
            metrics_val.add_batch(gt_label.view(-1).cpu().numpy(), sem_label.cpu().numpy())
            metrics_val_each.add_batch(gt_label.view(-1).cpu().numpy(), sem_label.cpu().numpy())
        
        if val_type == 'val':
            gt_label_rgb = torch.from_numpy(gt_label_rgb)
            pseudo_gt_label_rgb = metadata_item.load_label()
            pseudo_gt_label_rgb = custom2rgb(pseudo_gt_label_rgb.view(*viz_rgbs.shape[:-1]).cpu().numpy())
            pseudo_gt_label_rgb = torch.from_numpy(pseudo_gt_label_rgb)
        elif val_type == 'train':
            pseudo_gt_label_rgb = torch.from_numpy(gt_label_rgb)
            gt_label_rgb = None

        img_list.append(pseudo_gt_label_rgb)
        img_list.append(gt_label_rgb)
        img_list.append(torch.from_numpy(visualize_sem))
        Image.fromarray((visualize_sem).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / ("%06d_pred_label.jpg" % i)))
        if writer is not None:
            writer.add_image('5_val_images_semantic/{}'.format(i), torch.from_numpy(visualize_sem).permute(2, 0, 1), i)

    return


def get_sdf_normal_map(metadata_item, results, typ, viz_rgbs):
    #  NSR  SDF ------------------------------------  save the normal_map
    # world -> camera 
    w2c = torch.linalg.inv(torch.cat((metadata_item.c2w,torch.tensor([[0,0,0,1]])),0))
    viz_result_normal_map = results[f'normal_map_{typ}']
    viz_result_normal_map = torch.mm(w2c[:3,:3],viz_result_normal_map.T).T
    # normalize 
    viz_result_normal_map = viz_result_normal_map / (1e-5 + torch.linalg.norm(viz_result_normal_map, ord = 2, dim=-1, keepdim=True))
    viz_result_normal_map = viz_result_normal_map.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()

    normal_viz = (viz_result_normal_map+1)*0.5*255

    return normal_viz

def save_semantic_metric(metrics_val_each, CLASSES, samantic_each_value, wandb, writer, train_index, i):
    mIoU = np.nanmean(metrics_val_each.Intersection_over_Union())
    F1 = np.nanmean(metrics_val_each.F1())
    # OA = np.nanmean(metrics_val_each.OA())
    FW_IoU = metrics_val_each.Frequency_Weighted_Intersection_over_Union()
    iou_per_class = metrics_val_each.Intersection_over_Union()

    samantic_each_value['mIoU'].append(mIoU)
    samantic_each_value['FW_IoU'].append(FW_IoU)
    samantic_each_value['F1'].append(F1)
    # samantic_each_value['OA'].append(OA)

    for class_name, iou in zip(CLASSES, iou_per_class):
        samantic_each_value[f'{class_name}_iou'].append(iou)
    

    for class_name, iou in zip(CLASSES, iou_per_class):
        if np.isnan(iou):
            continue
        if wandb is not None:
            wandb.log({f'val/mIoU_each_class/{train_index}_{class_name}': iou, 'epoch':i})
            wandb.log({'val/FW_IoU_each_images/{}'.format(train_index): FW_IoU, 'epoch':i})
        if writer is not None:
            writer.add_scalar(f'4_{class_name}/{i}', iou, train_index)
            writer.add_scalar('3_val_each_image_FW_IoU/{}'.format(train_index), FW_IoU, i)

    return samantic_each_value

def write_metric_to_folder_logger(metrics_val, CLASSES, experiment_path_current, samantic_each_value, wandb, writer, train_index):
    mIoU = np.nanmean(metrics_val.Intersection_over_Union())
    FW_IoU = metrics_val.Frequency_Weighted_Intersection_over_Union()
    F1 = np.nanmean(metrics_val.F1())
    # OA = np.nanmean(metrics_val.OA())
    iou_per_class = metrics_val.Intersection_over_Union()

    eval_value = {'mIoU': mIoU,
                    'FW_IoU': FW_IoU,
                    'F1': F1,
                #   'OA': OA,
                    }
    print("eval_value")
    print('val:', eval_value)

    iou_value = {}
    for class_name, iou in zip(CLASSES, iou_per_class):
        iou_value[class_name] = iou
    print(iou_value)

    with(experiment_path_current / 'semantic_each.txt').open('w') as f2:
        for key in samantic_each_value:
            f2.write(f'{key}:\n')
            for k in range(len(samantic_each_value[key])):
                f2.write(f'\t\t{k:<3}: {samantic_each_value[key][k]}\n')

    with (experiment_path_current /'metrics.txt').open('a') as f:
        for key in eval_value:
            f.write(f'{eval_value[key]}\t')
        for key in iou_value:
            f.write(f'{iou_value[key]}\t')
        f.write(f'\n\n')
        f.write('eval_value:\n')
        for key in eval_value:
            f.write(f'\t\t{key:<12}: {eval_value[key]}\n')
        f.write('iou_value:\n')
        for key in iou_value:
            f.write(f'\t\t{key:<12}: {iou_value[key]}\n' )
    

    if wandb is not None:
        wandb.log({'val/mIoU': mIoU, 'epoch':train_index})
        wandb.log({'val/FW_IoU': FW_IoU, 'epoch':train_index})
        wandb.log({'val/F1': F1, 'epoch':train_index})
        # self.wandb.log({'val/OA': OA, 'epoch':train_index})
    if writer is not None:
        writer.add_scalar('2_val_metric_average/mIoU', mIoU, train_index)
        writer.add_scalar('2_val_metric_average/FW_IoU', FW_IoU, train_index)
        writer.add_scalar('2_val_metric_average/F1', F1, train_index)
        # self.writer.add_scalar('val/OA', OA, train_index)



def prepare_depth_normal_visual(img_list, hparams, metadata_item, typ, results, visualize_scalars):
    depth_map = None
    H, W = metadata_item.H, metadata_item.W
    if f'depth_{typ}' in results:
        depth_map = results[f'depth_{typ}']
        if f'fg_depth_{typ}' in results:
            to_use = results[f'fg_depth_{typ}'].view(-1)
            while to_use.shape[0] > 2 ** 24:
                to_use = to_use[::2]
            ma = torch.quantile(to_use, 0.95)
            depth_clamp = depth_map.clamp_max(ma)
        else:
            depth_clamp = depth_map

        depth_vis = torch.from_numpy(visualize_scalars(
                torch.log(depth_clamp + 1e-8).view(H, W).cpu()))
        img_list.append(depth_vis)

    if hparams.depth_loss:  # GT depth
        depth_cue = metadata_item.load_depth().float()
        depth_cue = torch.from_numpy(visualize_scalars(depth_cue))
        img_list.append(depth_cue)
    
    if (hparams.depth_dji_loss or (hparams.dataset_type=='memory_depth_dji')) and not hparams.render_zyq:  # DJI Gt depth
        depth_dji = metadata_item.load_depth_dji().float()
        invalid_mask = torch.isinf(depth_dji)
        depth_dji = torch.from_numpy(visualize_scalars(depth_dji,invalid_mask))
        img_list.append(depth_dji)


    if f'normal_map_{typ}' in results:
        # world -> camera 
        w2c = torch.linalg.inv(torch.cat((metadata_item.c2w,torch.tensor([[0,0,0,1]])), 0))
        normal_map = results[f'normal_map_{typ}']
        # normal_map = torch.mm(normal_map, w2c[:3,:3])# + w2c[:3,3]
        normal_map = torch.mm(w2c[:3,:3], normal_map.T).T
        # normalize 
        normal_map = normal_map / (1e-5 + torch.linalg.norm(normal_map, ord = 2, dim=-1, keepdim=True))

        if 'sdf' not in hparams.network_type:
            normal_map = normal_map * -1
        normal_map = normal_map.view(H, W, 3).cpu()
        
        normal_viz = (normal_map + 1)*0.5
        img_list.append(normal_viz*255)

        

        # camera_world = origin_to_world(metadata_item)
        # light_source = camera_world[0,0] 

        light_source = torch.Tensor([0.05, -0.05, 0.05]).float()
        
        # camera_world = torch.concat((metadata_item.c2w, torch.tensor([[0,0,0,1]])), 0)
        # light_source = torch.inverse(camera_world)[:3,3].float()

        # light_source = metadata_item.c2w[:3,3]
        # light_source = torch.Tensor([0.05, light_source[1], light_source[2]]).float()

        light = (light_source / light_source.norm(2)).unsqueeze(1)

        diffuse_per = torch.Tensor([0.7,0.7,0.7]).float()
        ambiant = torch.Tensor([0.3,0.3,0.3]).float()
        
        diffuse = torch.mm(normal_viz.view(-1,3), light).clamp_min(0).repeat(1, 3) * diffuse_per.unsqueeze(0)

        geo_viz = (ambiant.unsqueeze(0) + diffuse).clamp_max(1.0)
        geo_viz = geo_viz.view(H, W, 3).cpu()
        img_list.append(geo_viz*255)



    if hparams.normal_loss:
        normal_cue = metadata_item.load_normal()
        normal_cue = (normal_cue + 1) * 0.5 * 255
        img_list.append(normal_cue)


    if 'bg_lambda_fine' in results:
        fg_mask = (results['bg_lambda_fine'] < 0.01).reshape(H, W, 1)
        fg_mask = fg_mask.repeat(1, 1, 3) * 255
        img_list.append(fg_mask)
    return


def origin_to_world(metadata_item, pose_scale_factor, invert=True):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''

    # Create origin in homogen coordinates
    p = torch.zeros(1, 4, 1)
    p[:, -1] = 1.

    K1 = metadata_item.intrinsics
    K1 = np.array([[K1[0], 0, K1[2]],[0, K1[1], K1[3]],[0,0,1]])

    E1 = np.array(metadata_item.c2w)
    E1 = np.stack([E1[:, 0], E1[:, 1]*-1, E1[:, 2]*-1, E1[:, 3]], 1)

    camera_mat = np.concatenate((E1, [[0,0,0,1]]), 0)

    world_mat = K1

    # # Invert matrices
    # if invert:
    #     camera_mat = torch.inverse(torch.from_numpy(camera_mat)).float()
    #     world_mat = torch.inverse(torch.from_numpy(world_mat)).float()

    # Apply transformation
    p_world = world_mat @ camera_mat @ p[:,:3,:]

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world

def get_depth_vis(results, typ):
    if f'depth_{typ}' in results:
        viz_depth = results[f'depth_{typ}']
        if f'fg_depth_{typ}' in results:
            to_use = results[f'fg_depth_{typ}'].view(-1)
            while to_use.shape[0] > 2 ** 24:
                to_use = to_use[::2]
            ma = torch.quantile(to_use, 0.95)

            viz_depth = viz_depth.clamp_max(ma)
    else: 
        viz_depth = None
