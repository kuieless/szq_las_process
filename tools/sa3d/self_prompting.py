'''
    Self prompting strategy
    
    INPUT: 
        predictor: the initialized sam predictor :
        rendered_mask_score: - : H*W*1
        num_prompt: -  
        index_matrix: the matrix contains the 3D index of the rendered view : H*W*3
    OUTPUT: a list of prompts
'''
import os
import torch
import math
import numpy as np

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

@torch.no_grad()
def mask_to_prompt_N(predictor, rendered_mask_score, index_matrix, num_prompts = 3, selected_points=None):
    '''main function for self prompting'''
    h, w, _ = index_matrix.shape

    tmp = rendered_mask_score.view(-1)
    print("tmp min:", tmp.min(), "tmp max:", tmp.max())
    rand = torch.ones_like(tmp)
    topk_v, topk_p = torch.topk(tmp*rand, k = 1)[0].cpu(), torch.topk(tmp*rand, k = 1)[1].cpu()

    if topk_v <= 0:
        print("No prompt is available")
        return np.zeros((0,2)), np.ones((0))

    # 最开始， 从render_mask中选择score最高的一点（这个score是3d mask volume rendering的结果）， value  location
    prompt_points = []
    # prompt_points.append([topk_p[0] % w, topk_p[0] // w])
    # print((topk_p[0] % w).item(), (topk_p[0] // w).item(), h, w)

    prompt_points.append([selected_points[topk_p, 0].item(), selected_points[topk_p, 1].item()])
    


    # zyq: 这里初始化的score改成 -1e5,然后采样点的位置附上score
    tmp_mask = -1e5 * torch.ones((h,w,1))
    # tmp_mask[selected_points[:, 0], selected_points[:, 1] = rendered_mask_score
    for index_c in range(len(selected_points)):
        x, y = selected_points[index_c, 0], selected_points[index_c, 1]
        tmp_mask[x, y] = rendered_mask_score[index_c]
    # tmp_mask = rendered_mask_score.clone().detach()
    # NOTE 下面这个部分具体看的不是很明白，大概作用是mask out the last prompt point
    area = to8b(tmp_mask.cpu().numpy()).sum() / 255
    r = np.sqrt(area / math.pi)
    masked_r = max(int(r) // 2, 2)
    
    pre_tmp_mask_score = None
    for _ in range(num_prompts - 1):
        # mask out a region around the last prompt point
        input_label = np.ones(len(prompt_points))
        previous_masks, previous_scores, previous_logits = predictor.predict(
            point_coords=np.array(prompt_points),
            point_labels=input_label,
            multimask_output=False,
        )

        l = 0 if prompt_points[-1][0]-masked_r <= 0 else prompt_points[-1][0]-masked_r
        r = w-1 if prompt_points[-1][0]+masked_r >= w-1 else prompt_points[-1][0]+masked_r

        t = 0 if prompt_points[-1][1]-masked_r <= 0 else prompt_points[-1][1]-masked_r
        b = h-1 if prompt_points[-1][1]+masked_r >= h-1 else prompt_points[-1][1]+masked_r
        tmp_mask[t:b+1, l:r+1, :] = -1e5   # 每一次加一个点，这里是把 the last prompt point置为负值，不参与后面的排名

        # bool: H W
        previous_mask_tensor = torch.tensor(previous_masks[0]) #[1, H, W] -> [H, W]
        previous_mask_tensor = previous_mask_tensor.unsqueeze(0).unsqueeze(0).float()  #->[1, 1, H, W]
        previous_mask_tensor = torch.nn.functional.max_pool2d(previous_mask_tensor, 25, stride = 1, padding = 12)  #->[1, 1, H, W]
        previous_mask_tensor = previous_mask_tensor.squeeze(0).permute([1,2,0])   # -> [H, W, 1]
#         tmp_mask[previous_mask_tensor > 0] = -1e5
        #zyq: 取出对应的位置
        # previous_max_score = torch.max(rendered_mask_score[previous_mask_tensor > 0])
        previous_max_score = torch.max(rendered_mask_score[(previous_mask_tensor > 0)[selected_points[:, 0], selected_points[:, 1]]]).cpu()

        # 坐标 + depth
        previous_point_index = torch.zeros_like(index_matrix)
        previous_point_index[:,:,0] = prompt_points[-1][1] / h
        previous_point_index[:,:,1] = prompt_points[-1][0] / w
        previous_point_index[:,:,2] = index_matrix[int(prompt_points[-1][1]), int(prompt_points[-1][0]), 2]
        distance_matrix = torch.sqrt(((index_matrix - previous_point_index)**2).sum(-1))  #所有的深度和选取的一点算距离
        distance_matrix = (distance_matrix.unsqueeze(-1) - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min()) # 归一化
        # NOTE 这个 cur_tmp_mask为什么这么算   score - distance_matrix * max(score)
        cur_tmp_mask = tmp_mask - distance_matrix * max(previous_max_score, 0)

        if pre_tmp_mask_score is None:
            pre_tmp_mask_score = cur_tmp_mask
        else:
            pre_tmp_mask_score[pre_tmp_mask_score < cur_tmp_mask] = cur_tmp_mask[pre_tmp_mask_score < cur_tmp_mask]  # 如果当前的分数>上一步的，则替代
            pre_tmp_mask_score[tmp_mask == -1e5] = -1e5

        tmp_val_point = pre_tmp_mask_score.view(-1).max(dim = 0)  # 取出最大的位置（除去已被提取的point）

        if tmp_val_point[0] <= 0:
            print("There are", len(prompt_points), "prompts")
            break
        prompt_points.append([int(tmp_val_point[1].cpu() % w), int(tmp_val_point[1].cpu() // w)])

    prompt_points = np.array(prompt_points)
    input_label = np.ones(len(prompt_points))

    return prompt_points, input_label


@torch.no_grad()
def mask_to_prompt(predictor, rendered_mask_score, index_matrix, num_prompts = 3):
    '''main function for self prompting'''
    h, w, _ = rendered_mask_score.shape
    tmp = rendered_mask_score.view(-1)
    print("tmp min:", tmp.min(), "tmp max:", tmp.max())
    rand = torch.ones_like(tmp)
    topk_v, topk_p = torch.topk(tmp*rand, k = 1)[0].cpu(), torch.topk(tmp*rand, k = 1)[1].cpu()

    if topk_v <= 0:
        print("No prompt is available")
        return np.zeros((0,2)), np.ones((0))

    # 最开始， 从render_mask中选择score最高的一点（这个score是3d mask volume rendering的结果）， value  location
    prompt_points = []
    prompt_points.append([topk_p[0] % w, topk_p[0] // w])

    print((topk_p[0] % w).item(), (topk_p[0] // w).item(), h, w)

    tmp_mask = rendered_mask_score.clone().detach()
    # NOTE 下面这个部分具体看的不是很明白，大概作用是mask out the last prompt point
    area = to8b(tmp_mask.cpu().numpy()).sum() / 255
    r = np.sqrt(area / math.pi)
    masked_r = max(int(r) // 2, 2)
    # masked_r = max(int(r) // 3, 2)

    pre_tmp_mask_score = None
    for _ in range(num_prompts - 1):
        # mask out a region around the last prompt point
        input_label = np.ones(len(prompt_points))
        previous_masks, previous_scores, previous_logits = predictor.predict(
            point_coords=np.array(prompt_points),
            point_labels=input_label,
            multimask_output=False,
        )

        l = 0 if prompt_points[-1][0]-masked_r <= 0 else prompt_points[-1][0]-masked_r
        r = w-1 if prompt_points[-1][0]+masked_r >= w-1 else prompt_points[-1][0]+masked_r

        t = 0 if prompt_points[-1][1]-masked_r <= 0 else prompt_points[-1][1]-masked_r
        b = h-1 if prompt_points[-1][1]+masked_r >= h-1 else prompt_points[-1][1]+masked_r
        tmp_mask[t:b+1, l:r+1, :] = -1e5   # 每一次加一个点，这里是把 the last prompt point置为负值，不参与后面的排名

        # bool: H W
        previous_mask_tensor = torch.tensor(previous_masks[0]) #[1, H, W] -> [H, W]
        previous_mask_tensor = previous_mask_tensor.unsqueeze(0).unsqueeze(0).float()  #->[1, 1, H, W]
        previous_mask_tensor = torch.nn.functional.max_pool2d(previous_mask_tensor, 25, stride = 1, padding = 12)  #->[1, 1, H, W]
        previous_mask_tensor = previous_mask_tensor.squeeze(0).permute([1,2,0])   # -> [H, W, 1]
#         tmp_mask[previous_mask_tensor > 0] = -1e5
        previous_max_score = torch.max(rendered_mask_score[previous_mask_tensor > 0])

        # 坐标 + depth
        previous_point_index = torch.zeros_like(index_matrix)
        previous_point_index[:,:,0] = prompt_points[-1][1] / h
        previous_point_index[:,:,1] = prompt_points[-1][0] / w
        previous_point_index[:,:,2] = index_matrix[int(prompt_points[-1][1]), int(prompt_points[-1][0]), 2]
        distance_matrix = torch.sqrt(((index_matrix - previous_point_index)**2).sum(-1))  #所有的深度和选取的一点算距离
        distance_matrix = (distance_matrix.unsqueeze(-1) - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min()) # 归一化
        # NOTE 这个 cur_tmp_mask为什么这么算   score - distance_matrix * max(score)
        cur_tmp_mask = tmp_mask - distance_matrix * max(previous_max_score, 0)

        if pre_tmp_mask_score is None:
            pre_tmp_mask_score = cur_tmp_mask
        else:
            pre_tmp_mask_score[pre_tmp_mask_score < cur_tmp_mask] = cur_tmp_mask[pre_tmp_mask_score < cur_tmp_mask]  # 如果当前的分数>上一步的，则替代
            pre_tmp_mask_score[tmp_mask == -1e5] = -1e5

        tmp_val_point = pre_tmp_mask_score.view(-1).max(dim = 0)  # 取出最大的位置（除去已被提取的point）

        if tmp_val_point[0] <= 0:
            print("There are", len(prompt_points), "prompts")
            break
        prompt_points.append([int(tmp_val_point[1].cpu() % w), int(tmp_val_point[1].cpu() // w)])

    prompt_points = np.array(prompt_points)
    input_label = np.ones(len(prompt_points))

    return prompt_points, input_label
