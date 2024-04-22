# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from yacs.config import CfgNode
from models import build_model
from datasets import build_normal_loader
import numpy as np
import cv2

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        mean = self.mean.to(tensor.device).view(1, 3, 1, 1)
        std = self.std.to(tensor.device).view(1, 3, 1, 1)
        return tensor * std + mean

denormal = DeNormalize(mean=[0.56347245, 0.50660025, 0.45908741], std=[0.28393339, 0.2804536 , 0.30424776])

def CN(**kwgs):
    return CfgNode(kwgs)

epo = 'last'

config = CfgNode(dict(
    ENCODER = 'VGG',
    DECODER = 'PSCC',
    FACTOR = 100,
    RESUME = f'exp/0615162748/output/ckpt_epoch_{epo}.pth',
    DATA_PATH = '/mnt/lustre/linwei/resense/fsc147/',
    BATCH_SIZE = 1,
    PIN_MEMORY = True
))

# dataset
data_loader = build_normal_loader(config, 'val')

# model
model = build_model(config)
model.cuda()
model.eval()
# checkpoint = torch.load(config.RESUME, map_location='cpu')
# msg = model.load_state_dict(checkpoint['model'], strict=False)
# print(f'load parameters : {msg}')

for idx, (images, templates, dotmap) in enumerate(data_loader):
    with torch.no_grad():
        images = images.cuda(non_blocking=True)
        dotmap = dotmap.cuda(non_blocking=True)
        templates = templates.cuda(non_blocking=True)

        features, feaboxes = model(images, templates, dotmap)

        h, w = features.size(-2) - 7, features.size(-1) - 7

        features = F.unfold(features, 8, dilation=1, padding=0, stride=1)
        feaboxes = feaboxes.reshape(3, -1).unsqueeze(-1)
        print(features.shape, feaboxes.shape)
        sim = torch.sum((feaboxes - features) ** 2, dim=1)
        sim = torch.sqrt(sim).reshape(1, 3, h, w)
        sim = F.interpolate(sim, size=((h + 7) * 8, (w + 7) * 8))[0]
        sim = sim.cpu().numpy()
        for i, s in enumerate(sim):
            plt.imsave(f'sim{i}.png', s)
        
        images = denormal(images) * 255
        images = torch.clamp(images, 0, 255)
        images = images.permute(0, 2, 3, 1) @ torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).to(images.device)
        images = images[0].cpu().numpy().astype('uint8')
        
        for i, temp in enumerate(templates[0]):
            temp = temp.int().cpu().numpy()
            print(temp)
            img = cv2.rectangle(images, (temp[1], temp[0]), (temp[3], temp[2]), (0, 0, 255), 2)
            denmap = cv2.imread(f'sim{i}.png')
            print(img.shape, denmap.shape)
            x = img * 0.2 + denmap * 0.8
            cv2.imwrite(f'image{i}.jpg', x)
    break


# srun --mpi=pmi2 --partition=vi_irdc -n1 --gres=gpu:1 python visual.py