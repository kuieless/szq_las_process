import copy
import random

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from tools.sa3d.self_prompting import mask_to_prompt
from typing import Optional


def to_tensor(array, device=torch.device('cuda')):
    '''cvt numpy array to cuda tensor, if already tensor, do nothing
    '''
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor) and not array.is_cuda:
        array = array.to(device)
    else:
        pass
    return array.float()

def seg_loss(mask: Tensor, selected_mask: Optional[Tensor], seg_m: Tensor, lamda: float = 5.0) -> Tensor:
    """
    Compute segmentation loss using binary mask and predicted mask.

    Args:
        mask: Binary ground truth segmentation mask tensor.
        seg_m: Predicted segmentation mask tensor.
        lamda: Weighting factor for outside mask loss. Default is 5.0.

    Returns:
        Computed segmentation loss.

    Raises:
        AssertionError: If `seg_m` is `None`.
    """
    assert seg_m is not None, "Segmentation mask is None."

    # mask_loss = -(to_tensor(mask, device) * seg_m.squeeze(-1)).sum()
    # out_mask_loss = lamda * ((1 - to_tensor(mask, device)) * seg_m.squeeze(-1)).sum()
    
    if selected_mask is not None:     #selected_mask   选择sam的mask中得分最高的一个
        device = seg_m.device
        mask_loss = -(to_tensor(mask[selected_mask], device) * seg_m.squeeze(-1)).sum()
        out_mask_loss = lamda * (to_tensor(1 - mask[selected_mask], device) * seg_m.squeeze(-1)).sum()
    else:
        mask_loss = -(mask * seg_m.squeeze(-1)).sum()
        out_mask_loss = lamda * ((1 - mask) * seg_m.squeeze(-1)).sum()


    return mask_loss + out_mask_loss


def _generate_index_matrix(H, W, depth_map):
    '''generate the index matrix, which contains the coordinate of each pixel and cooresponding depth'''
    xs = torch.arange(1, H+1) / H # NOTE, range (1, H) = arange(1, H+1)
    ys = torch.arange(1, W+1) / W
    grid_x, grid_y = torch.meshgrid(xs, ys)
    index_matrix = torch.stack([grid_x, grid_y], dim = -1) # [H, W, 2]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) # [H, W, 1]
    index_matrix = torch.cat([index_matrix, depth_map], dim = -1)
    return index_matrix

@torch.jit.script
def cal_IoU(a: Tensor, b: Tensor) -> Tensor:
    """Calculates the Intersection over Union (IoU) between two tensors.

    Args:
        a: A tensor of shape (N, H, W).
        b: A tensor of shape (N, H, W).

    Returns:
        A tensor of shape (N,) containing the IoU score between each pair of
        elements in a and b.
    """
    intersection = torch.count_nonzero(torch.logical_and(a == b, a != 0))
    union = torch.count_nonzero(a + b)
    return intersection / union


def prompting_coarse(self, H, W, seg_m, index_matrix, num_obj):
        '''TODO, for coarse stage, we use the self-prompting method to generate the prompt and mask'''
        seg_m_clone = seg_m.detach().clone()
        seg_m_for_prompt = seg_m_clone
        # kernel_size = 3
        # padding = kernel_size // 2
        # seg_m_for_prompt = torch.nn.functional.avg_pool2d(seg_m_clone.permute([2,0,1]).unsqueeze(0), kernel_size, stride = 1, padding = padding)
        # seg_m_for_prompt = seg_m_for_prompt.squeeze(0).permute([1,2,0])

        loss = 0

        for num in range(num_obj):
            with torch.no_grad():
                # self-prompting
                prompt_points, input_label = mask_to_prompt(predictor = self.predictor, rendered_mask_score = seg_m_for_prompt[:,:,num][:,:,None], 
                                                            index_matrix = index_matrix, num_prompts = 20)

                masks, selected = None, -1
                if len(prompt_points) != 0:
                    masks, scores, logits = self.predictor.predict(
                        point_coords=prompt_points,
                        point_labels=input_label,
                        multimask_output=False,
                    )
                    selected = np.argmax(scores)

            if num == 0:
                # used for single object only
                sam_seg_show = masks[selected].astype(np.float32) if masks is not None else np.zeros((H,W))
                sam_seg_show = np.stack([sam_seg_show,sam_seg_show,sam_seg_show], axis = -1)
                r = 8
                for ip, point in enumerate(prompt_points):
                    sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, :] = 0
                    if ip < 3:
                        sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, ip] = 1
                    else:
                        sam_seg_show[point[1]-r : point[1]+r, point[0] - r : point[0]+r, -1] = 1
                    

            if masks is not None:
                tmp_seg_m = seg_m[:,:,num]
                tmp_rendered_mask = tmp_seg_m.detach().clone()
                tmp_rendered_mask[torch.logical_or(tmp_rendered_mask <= tmp_rendered_mask.mean(), tmp_rendered_mask <= 0)] = 0
                tmp_rendered_mask[tmp_rendered_mask != 0] = 1
                tmp_IoU = cal_IoU(torch.as_tensor(masks[selected]).float(), tmp_rendered_mask)
                print(f"current IoU is: {tmp_IoU}")
                if tmp_IoU < 0.5:
                    print("SKIP, unacceptable sam prediction, IoU is", tmp_IoU)
                    continue

                loss += seg_loss(masks, selected, tmp_seg_m)
                for neg_i in range(seg_m.shape[-1]):
                    if neg_i == num:
                        continue
                    loss += (torch.tensor(masks[selected]).to(seg_m.device) * seg_m[:,:,neg_i]).sum()
        return loss, sam_seg_show