# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def agaussian(kernel_size, sigma=None):
    if type(kernel_size) == int:
        kernel_size = (kernel_size, kernel_size)
    muh, muw = kernel_size[0] // 2, kernel_size[1] // 2
    if sigma is None:
        sigmah, sigmaw = math.ceil(kernel_size[0] / 4), math.ceil(kernel_size[1] / 4)
    else:
        sigmah, sigmaw = sigma, sigma
    gaussh = lambda x: math.exp(-(x - muh) ** 2 / float(2 * sigmah ** 2))
    gaussw = lambda x: math.exp(-(x - muw) ** 2 / float(2 * sigmaw ** 2))
    hseq = torch.tensor([gaussh(x) for x in range(kernel_size[0])]).unsqueeze(1)
    wseq = torch.tensor([gaussw(x) for x in range(kernel_size[1])]).unsqueeze(0)

    kernels = hseq @ wseq
    weight = kernels.reshape(1, 1, kernel_size[0], kernel_size[1])
    weight = weight / weight.sum()
    return weight

class MSELoss(nn.modules.loss._Loss):
    def __init__(self, factor, reduction='sum') -> None:
        super().__init__()
        
        self.gsize = 17
        self.register_buffer('gaussian', agaussian(self.gsize, sigma=4))
        self.factor = factor

        self.lossfunc = nn.MSELoss(reduction=reduction)
    
    def forward(self, denmap, tarden=None, tardot=None):
        if tardot is not None:
            with torch.no_grad():
                tarnum = tardot.sum(dim=(1, 2, 3), keepdim=True) * self.factor
                tar = F.conv2d(tardot, self.gaussian, stride=1, padding=self.gsize // 2)
                if tar.size(-1) != denmap.size(-1) or tar.size(-2) != denmap.size(-2):
                    tar = F.interpolate(tar, size=denmap.shape[-2:], mode='bilinear', align_corners=False)
                tar = (tar / (tar.sum(dim=(1, 2, 3), keepdim=True) + 1e-6)) * tarnum
        else:
            tar = tarden
        loss = self.lossfunc(denmap, tar)
        return loss