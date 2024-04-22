# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNetFPN import ResNetFPN
from .COMPSER import COMPSER
from einops import rearrange

EPS = 1e-4

class PSCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetFPN() # SwinFPN() # 
        self.predictor = COMPSER(512)

    def forward(self, image, dotmap=None, boxmaps=None):
        b, _, imh, imw = image.shape
        fea_8, fea_16 = self.encoder(image)
        rgb_fea_8 = fea_8
        rgb_fea_16 = F.avg_pool2d(rgb_fea_8, 2, stride=2)

        if self.training:
            mask = F.adaptive_max_pool2d(dotmap, rgb_fea_16.shape[-2:])
            mask = mask * torch.rand_like(mask)
            gls_token = (mask * rgb_fea_16).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + EPS)
            box_token = (boxmaps * rgb_fea_16).sum(dim=(-1, -2)) / (boxmaps.sum(dim=(-1, -2)) + EPS)

            fea_group = [[fea_16], [rgb_fea_8], [rgb_fea_16]]
            nt = 1
            for i, fea in enumerate(fea_group):
                for r in range(min(nt, b)):
                    fea.append(torch.roll(fea[0], r + 1, 0))
                fea_group[i] = torch.cat(fea, dim=0)
            fea_16, rgb_fea_8, rgb_fea_16 = fea_group
            
            seq_fea_16 = rearrange(rgb_fea_16, "(t b) c h w -> b c (h w t)", t = nt + 1)
            
            
            token = box_token.reshape(b, -1)
            token = token.repeat(nt + 1, 1)
            
            denmap = self.predictor(rgb_fea_8, token)
            for _ in range(1):
                dwmask = F.adaptive_max_pool2d(denmap, (imh // 16, imw // 16)).detach()
                dwmask = rearrange(dwmask, "(t b) c h w -> b c (h w t)", t = nt + 1)
                token = ((dwmask / (dwmask.sum(dim=-1, keepdim=True) + EPS)) * seq_fea_16).sum(dim=-1)
                token = token.repeat(nt + 1, 1)
                denmap = self.predictor(rgb_fea_8, token)

            posden, negdens = denmap[:b], denmap[b:]

            gls_token = gls_token.repeat(nt + 1, 1)
            gdemap = self.predictor(rgb_fea_8, gls_token)
            pdemap, ndemaps = gdemap[:b], gdemap[b:]

            return posden, negdens, pdemap, ndemaps
        else:
            box_token = (boxmaps * rgb_fea_16).sum(dim=(-1, -2)) / (boxmaps.sum(dim=(-1, -2)) + EPS)
            token = box_token.reshape(b, -1)
            
            posden = self.predictor(rgb_fea_8, token)
            for _ in range(1):
                weights = F.adaptive_max_pool2d(posden, (imh // 16, imw // 16))
                token = (weights * rgb_fea_16).sum(dim=(-1, -2)) / (weights.sum(dim=(-1, -2)) + EPS)
                
                posden = self.predictor(rgb_fea_8, token)

            return posden
