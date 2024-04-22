# -*- coding: utf-8 -*-
import os
from PIL import Image
import json
import torch
from torch.utils import data
import cv2
import random
from .utils import NormalSample, jpg2id

class FSC147(data.Dataset):
    def __init__(self, root_path, mode):
        super().__init__()
        with open(os.path.join(root_path, 'Train_Test_Val_FSC_147.json')) as f:
            imglist = json.load(f)[mode]
        self.imgids = [jpg2id(imgf) for imgf in imglist]
        
        self.can_h = 512
        self.can_w = 768

        with open(os.path.join(root_path, f'fsc147_{self.can_h}x{self.can_w}.json')) as f:
            samples = json.load(f)
        self.samples = {idx: samples[idx] for idx in self.imgids}
        for sid, sample in self.samples.items():
            imgpath = sample['imagepath']
            sample["maskpath"] = imgpath.replace("images", "llama2").replace("jpg", "png")
        
        self.root_path = root_path
        self.mode = mode
        self.normalfunc = NormalSample(self.can_h, self.can_w, train=(mode=='train'))
    
    def __getitem__(self, index):
        imgid = self.imgids[index]
        #print(imgid)
        return self.getSample(imgid)
        # image, boxmap, dotmap = self.getSample(imgid)

        # return image, boxmap, dotmap #, imgid

    def __len__(self):
        return len(self.imgids)
    
    def getSample(self, imgid):
        #print(imgid)
        sample = self.samples[imgid]

        impath = os.path.join(self.root_path, sample['imagepath'])
        image = Image.open(impath).convert('RGB')
        points = torch.tensor(sample['points']).round().long() # N x (w, h)

        # llama clip mask
        mapath = os.path.join(self.root_path, sample['maskpath'])
        mask = Image.open(mapath).convert('L')

        # box
        boxes = torch.tensor(sample['boxes']).round().long().reshape(-1, 4) # 3 x ((left, top), (right, bottom))
        # h = boxes[:, 3] - boxes[:, 1]
        # w = boxes[:, 2] - boxes[:, 0]
        # boxes = torch.stack((h, w), dim=-1)
        # print(boxes)
        box_sel = torch.randn((boxes.size(0), 1))
        box_sel = (box_sel == box_sel.max(dim=0, keepdim=True).values)
        boxes = (boxes * box_sel).sum(dim=0, keepdim=True)
        # print(boxes, box_sel)
        image, dotmap, boxmap, clip_mask, _ = self.normalfunc(image, dots=points, boxes=boxes, clip_mask=mask)
        
        # generate point prompt
        dotppt = (torch.rand_like(dotmap) * dotmap).flatten()
        dotsel = dotppt.argmax()
        dotppt = torch.zeros_like(dotppt)
        dotppt[dotsel] = 1
        dotppt = dotppt.view_as(dotmap)

        return image, dotmap, boxmap, dotppt, clip_mask


    @staticmethod
    def collate_fn(samples):
        samples = zip(*samples)
        samples = [torch.stack(sample, dim=0) for sample in samples]
        return samples


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor):
        mean = self.mean.to(tensor.device).view(1, 3, 1, 1)
        std = self.std.to(tensor.device).view(1, 3, 1, 1)
        return tensor * std + mean

if __name__ == '__main__':
    dataset = FSC147('/qnap/home_archive/wlin38/coey/gsc147/', 'train')
    import matplotlib.pyplot as plt
    denormal = DeNormalize(
        mean=[0.56347245, 0.50660025, 0.45908741], 
        std=[0.28393339, 0.2804536 , 0.30424776]
    )
    leng = []
    import tqdm
    # print("HE")
    for si, sample in enumerate(tqdm.tqdm(dataset)):
        if si < 5: continue
        image = denormal(sample[0].unsqueeze(0)).squeeze(0) * 255.
        cv2.imwrite('image.png', image.numpy().transpose((1, 2, 0))[:, :, ::-1])
        plt.imsave("dot.png", sample[1].squeeze())
        # print(sample[1].max(), sample[1].min())
        plt.imsave("box.png", sample[2].squeeze())
        plt.imsave("pot.png", sample[3].squeeze())
        plt.imsave("clip.png", sample[4].squeeze())
        break
