import torch
from torch import nn
import torch.nn.functional as F

class token_inference(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sigmoid = nn.Sigmoid()

        scale = dim ** -0.5
        self.H, self.W = 512 // 8, 768 // 8
        self.positional_embedding = nn.Parameter(scale * torch.randn(1, dim, self.H, self.W))
        self.key_embedding = nn.Parameter(scale * torch.randn(1, 1, dim))

    def forward(self, fea, H=48, W=72):
        B, N, C = fea.shape

        pos_embedding = F.interpolate(self.positional_embedding, (H, W), mode='bilinear', align_corners=False)
        pos_embedding = pos_embedding.flatten(-2).permute(0, 2, 1)
        pos_embedding = torch.cat((self.key_embedding, pos_embedding), dim=1)
        
        fea = fea + pos_embedding

        x = self.norm(fea)
        T_s, F_s = x[:, :1, :], x[:, 1:, :]

        q = self.q(F_s).reshape(B, N-1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(T_s).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.sigmoid(attn)
        attn = self.attn_drop(attn)

        infer_fea = (attn @ v).transpose(1, 2).reshape(B, N-1, C)
        infer_fea = self.proj(infer_fea)
        infer_fea = self.proj_drop(infer_fea)

        infer_fea = infer_fea
        return infer_fea