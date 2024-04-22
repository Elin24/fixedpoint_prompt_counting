# -*- coding: utf-8 -*-

from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils_transformer import token_inference

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, same_padding=False):
        super(Conv2d, self).__init__()
        padding = ((kernel_size - 1) // 2) if same_padding else 0
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class COMPSER(nn.Module):
    def __init__(self, indim):
        super(COMPSER, self).__init__()
        self.denfea_pre = token_inference(dim=indim, num_heads=16)
        self.decoder = nn.Sequential(
                                     Conv2d(indim, 256, 3, same_padding=True), # (256, h, w)
                                     Conv2d(256,  256, 3, same_padding=True),
                                     nn.PixelShuffle(8),
                                     nn.Conv2d(4, 1, 3, padding=1),
                                     nn.ReLU()
                                    )
        
        self.weights_normal_init(self.decoder, dev=0.005)

    def forward(self, x, token):
        b, c, h, w = x.shape
        x_ = rearrange(x, "b c h w -> b (h w) c")
        token = token.reshape(b, 1, c)
        x = torch.cat((token, x_), dim=1)
        x = self.denfea_pre(x, h, w)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.decoder(x)
        return x
    
    def weights_normal_init(self, model, dev=0.01):
        if isinstance(model, list):
            for m in model:
                self.weights_normal_init(m, dev)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):                
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)