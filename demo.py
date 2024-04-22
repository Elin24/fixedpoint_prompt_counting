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
from torchvision import transforms
from PIL import Image

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        mean = self.mean.to(tensor.device).view(1, 3, 1, 1)
        std = self.std.to(tensor.device).view(1, 3, 1, 1)
        return tensor * std + mean

normal = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
            mean=[0.56347245, 0.50660025, 0.45908741],
            std=[0.28393339, 0.2804536 , 0.30424776]
        )
])

denormal = DeNormalize(mean=[0.56347245, 0.50660025, 0.45908741], std=[0.28393339, 0.2804536 , 0.30424776])

def CN(**kwgs):
    return CfgNode(kwgs)

epo = 'best'

config = CfgNode(dict(
    ENCODER = 'VGG',
    DECODER = 'PSCC',
    FACTOR = 16,
    RESUME =  f'../output/ckpt_epoch_{epo}.pth',
    DATA_PATH = '/qnap/home_archive/wlin38/coey/gsc147',
    BATCH_SIZE = 1,
    PIN_MEMORY = True
))


# model
model, _ = build_model(config)#, single=True)
model.cuda()
model.eval()
checkpoint = torch.load(config.RESUME, map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=False)
print(f'load parameters : {msg}')

# test
denfile = f'den{epo}.png'
segfile = f'seg{epo}.png'
tarfile = f'tar{epo}.png'
profile = f'pro{epo}.png'
os.makedirs(f'xdemo{epo}', exist_ok=True)


root_path = "/home/grads/wlin38/qnap/coey/coc100_512x768/"
import json
with open(os.path.join(root_path, "label.json")) as f:
    dataset = json.load(f)
    dataset = dataset["berry"]# [dataset["berry"][7]]

for data in dataset:
    impath = os.path.join(root_path, data['imagepath'])
    imid = os.path.basename(data['imagepath'])[:-4]
    image = Image.open(impath).convert('RGB')
    image = normal(image)

    with torch.inference_mode():
        images = image[None, ...].cuda(non_blocking=True)
        B, _, H, W = images.shape
        label = torch.zeros(1, 2, H, W).to(images)

        for box in data["boxes"][:3]:
            box = (torch.Tensor(box) + 0.5).long().flatten()
            w1, h1, w2, h2 = box
            label[0, 0, h1:h2, w1:w2] = 1
        for pot in data["points"]:
            pot = (torch.Tensor(pot) + 0.5).long().flatten()
            w, h = pot
            label[0, 1, h, w] = 1
        boxmap, target = label[:1, :1], label[:1, 1:]

        
        imgs = denormal(images) * 255
        imgs = torch.clamp(imgs, 0, 255)
        imgs = imgs.permute(0, 2, 3, 1) @ torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).to(imgs.device)

        target_cpu = target.squeeze().cpu().numpy()
        print("[tar cpu]:", target_cpu.shape)
        plt.imsave(os.path.join(tarfile), target_cpu)
        
        tags = ["box", "point", "text"]

        for i, masks in enumerate([boxmap]):
            nh, nw = images.shape[-2:]
            boxmaps = F.adaptive_max_pool2d(masks, (nh // 16, nw // 16))
            # boxmaps = boxmaps * (boxmaps > torch.amax(boxmaps, dim=(-1, -2), keepdim=True) / 2)
            # boxmaps = boxmaps / boxmaps.amax(dim=(-1, -2), keepdim=True)
            # boxmaps = boxmaps 
            # boxmaps = torch.ones_like(boxmaps)
            denmap = model(images, dotmap=target, boxmaps=boxmaps)
            segmap = denmap
            # segmap = (segmap > segmap.amax(dim=(-1, -2), keepdim=True) / 10).float()
            
            # print("[Seg]:", segmap.max().item(), segmap.min().item())
            #segmap = boxmaps
            
            # segmap = union_find(denmap, config.FACTOR)
            # # print("[den]:", denmap.sum().item() / config.FACTOR, end=' -> ')
            # denmap = denmap * segmap
            # # print("[den2]:", denmap.sum().item() / config.FACTOR)
            # peakmap = F.max_pool2d(denmap, 3, stride=1, padding=1)
            # segmap = (peakmap == denmap).float() * (peakmap > 0).float()
            # segmap16 = F.adaptive_max_pool2d(segmap, (nh // 16, nw // 16))
            # # # segmap = target
            # denmap, segmap = model(images, boxmaps=segmap16, tokexp=False)


            # spa = (denmap > 1.5e-3).sum() / (denmap.sum() + 1e-2)
            # # # lrate = 4 / 3 # 16.70 | 67.75
            # # # lrate = 3 / 2 # 16.14 | 60.14
            # lrate = 2         # 15.26 | 47.46
            # # # lrate = 3     # 16.11 | 49.02
            # # # lrate = 4     # 16.40 | 50.86
            # resize_scale = max(min(49 / spa, lrate), 1 / lrate)
            # nh, nw = int(nh * resize_scale / 16 + 0.5) * 16, int(nw * resize_scale / 16 + 0.5) * 16
            
            # images = F.interpolate(images, (nh, nw), mode='bilinear', align_corners=False)
            # nh, nw = images.shape[-2:]
            # boxmaps = F.adaptive_avg_pool2d(masks, (nh // 16, nw // 16))
            # denmap, segmap = model(images, boxmaps=boxmaps)

            denmap = denmap.squeeze().cpu().numpy()
            segmap = segmap.squeeze().cpu().numpy()
            promap = boxmaps.squeeze().cpu().numpy()

            plt.imsave(os.path.join(tarfile), promap)
            plt.imsave(os.path.join(denfile), denmap)
            plt.imsave(os.path.join(segfile), segmap)

            image = imgs.squeeze().cpu().numpy().astype('uint8')

            canvas = np.zeros((H * 2 + 10, W * 2 + 10, 3), dtype='uint8')
            den = cv2.imread(denfile)
            den = cv2.resize(den, (W, H), interpolation = cv2.INTER_AREA)

            tar = cv2.imread(tarfile)
            tar = cv2.resize(tar, (W, H), interpolation = cv2.INTER_AREA)

            seg = cv2.imread(segfile)
            seg = cv2.resize(seg, (W, H), interpolation = cv2.INTER_AREA)

            # target
            canvas[:H, :W, :] = tar
            #image = image * 0.7 + template * 0.3 * 255
            canvas[:H, -W:, :] = image
            canvas[-H:, :W, :] = den
            canvas[-H:, -W:, :] = seg


            count = denmap.sum().item() / config.FACTOR
            gtc = target.sum().item()
            print(f"[{tags[i]}]: pd_cnt = {count} | gt_cnt={gtc}")
            cv2.imwrite(f'xdemo{epo}/{imid}_{tags[i]}-base.png', canvas)
        
        os.remove(denfile)
        os.remove(tarfile)
    # if idx > 50:
    #     break


# imid ['840']
# [text]: pd_cnt=364.47528076171875 | gt_cnt=637.0
# imid ['865']
# [text]: pd_cnt=196.86114501953125 | gt_cnt=1022.0
# imid ['935']
# [text]: pd_cnt=1830.516357421875 | gt_cnt=2092.0
# imid ['949']
# [text]: pd_cnt=798.2037353515625 | gt_cnt=1092.0
# imid ['1915']
# [text]: pd_cnt=530.2518310546875 | gt_cnt=684.0
# imid ['1956']
# [text]: pd_cnt=1368.449951171875 | gt_cnt=1229.0
# imid ['3484']
# [text]: pd_cnt=647.9129638671875 | gt_cnt=356.0
# imid ['3665']
# [text]: pd_cnt=1098.802734375 | gt_cnt=907.0
# imid ['5860']
# [text]: pd_cnt=419.60504150390625 | gt_cnt=757.0
# imid ['7656']
# [text]: pd_cnt=972.981201171875 | gt_cnt=1231.0