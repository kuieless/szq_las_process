from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ImageMetadata:
    #zyq : add label_path for semantic 
    def __init__(self, image_path: Path, c2w: torch.Tensor, W: int, H: int, intrinsics: torch.Tensor, image_index: int,
                 mask_path: Optional[Path], is_val: bool, label_path: Optional[Path], metadata_label=None, ):
        self.image_path = image_path
        self.c2w = c2w
        self.W = W
        self.H = H
        self.intrinsics = intrinsics
        self.image_index = image_index
        self._mask_path = mask_path
        self.is_val = is_val
        self.label = metadata_label
        self.label_path = label_path

    def load_image(self) -> torch.Tensor:
        rgbs = Image.open(self.image_path).convert('RGB')
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        return torch.ByteTensor(np.asarray(rgbs))

    def load_mask(self) -> Optional[torch.Tensor]:
        if self._mask_path is None:
            return None

        with ZipFile(self._mask_path) as zf:
            with zf.open(self._mask_path.name) as f:
                keep_mask = torch.load(f, map_location='cpu')

        if keep_mask.shape[0] != self.H or keep_mask.shape[1] != self.W:
            keep_mask = F.interpolate(keep_mask.unsqueeze(0).unsqueeze(0).float(),
                                      size=(self.H, self.W)).bool().squeeze()

        return keep_mask
    
    def load_label(self) -> torch.Tensor:
        labels = Image.open(self.label_path).convert('RGB')
        size = labels.size

        if size[0] != self.W or size[1] != self.H:
            labels = labels.resize((self.W, self.H), Image.LANCZOS)

        return torch.ByteTensor(np.asarray(labels))
    
    def load_label_class(self):
        labels = Image.open(self.label_path).convert('RGB')
        size = labels.size

        if size[0] != self.W or size[1] != self.H:
            labels = labels.resize((self.W, self.H), Image.LANCZOS)
        label = torch.ByteTensor(np.asarray(labels)).view(-1, 3)
        label_class = torch.zeros((label.shape[0]), dtype=torch.int)
        label_class[(label[:,0]==128+0) * (label[:,1]==0 + 0) * (label[:,2]==0 + 0)] = 0
        label_class[(label[:,0]==128+0) * (label[:,1]==64 + 0) * (label[:,2]==128 + 0)] = 1
        label_class[(label[:,0]==0  +0) * (label[:,1]==128 + 0) * (label[:,2]==0 + 0)] = 2
        label_class[(label[:,0]==128+0) * (label[:,1]==128 + 0) * (label[:,2]==0 + 0)] = 3
        label_class[(label[:,0]==64+0) * (label[:,1]==0 + 0) * (label[:,2]==128 + 0)] = 4
        label_class[(label[:,0]==192+0) * (label[:,1]==0 + 0) * (label[:,2]==192 + 0)] = 5
        label_class[(label[:,0]==64+0) * (label[:,1]==0 + 0) * (label[:,2]==64 + 0)] = 6
        label_class[(label[:,0]==0+0) * (label[:,1]==0 + 0) * (label[:,2]==0 + 0)] = 7

        return label_class
    

