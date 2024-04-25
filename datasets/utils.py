# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import random
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class NormalSample(object):
    def __init__(self, imh, imw, train=False):
        self.imh, self.imw = imh, imw
        self.flip = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
        ]) if train else None

        self.aug = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        ]) if train else None

        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
               mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )

    def __call__(self, image, dots, boxes=None, clip_mask=None, pimg=None):
        if self.aug is not None:
            image = self.aug(image)
        image = self.normalize(self.totensor(image))
        _, h, w = image.shape
        
        pptmap = torch.zeros((3, h, w)).float()
        dots[:, 0] = torch.clip(dots[:, 0], min=0, max=w-1)
        dots[:, 1] = torch.clip(dots[:, 1], min=0, max=h-1)
        for i, p in enumerate(dots):
            pptmap[0, p[1], p[0]] += 1
        
        if boxes is not None:
            for i, box in enumerate(boxes):
                # print(box)
                t, b = int(max(0, box[1])), int(min(h, box[3]))
                l, r = int(max(0, box[0])), int(min(w, box[2]))
                pptmap[1, max(0, t):min(b, h - 1), max(0, l):min(r, w - 1)] = 1
            
        if clip_mask is not None:
            pptmap[2] = self.totensor(clip_mask)

        if pimg is not None:
            pimg = self.normalize(self.totensor(pimg))
        else:
            pimg = torch.zeros_like(image)
        
        sample = torch.cat((image, pptmap, pimg), dim=0)
        if self.flip is not None:
            sample = self.flip(sample)
        image, dotmap, boxmap, clip_mask, pimg = sample[:3], sample[3:4], sample[4:5], sample[5:6], sample[6:]
        
        return image, dotmap, boxmap, clip_mask, pimg

def jpg2id(jpg):
    return jpg.replace('.jpg', '')