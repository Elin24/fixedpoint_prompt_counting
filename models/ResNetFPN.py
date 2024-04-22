import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetFPN(nn.Module):
    def __init__(self):
        super(ResNetFPN, self).__init__()
        net = models.resnet101(pretrained=True)
        mods = list(net.children())[:7]

        self.stage1 = nn.Sequential(*mods[:6])
        self.stage2 = nn.Sequential(*mods[6:], 
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU()
        )
        
        self.combine1 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU()
        )
        self.combine2 = nn.Sequential(
            nn.Conv2d(1024, 512, 5, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        x8 = self.stage1(x)
        x16 = self.stage2(x8)
        fea1 = self.combine1(torch.cat((
            x8,
            F.interpolate(x16, scale_factor=2, mode='bilinear', align_corners=False)
        ), dim=1))
        fea2 = self.combine2(torch.cat((
            F.avg_pool2d(x8, 2, stride=2),
            x16
        ), dim=1))
        return fea1, fea2
    
    def outdim(self):
        return 512

if __name__ == '__main__':
    mod = ResNetFPN().cuda()
    x = torch.randn(1, 3, 64, 64).cuda()
    f1, f2 = mod(x)
    print(f1.shape, f2.shape)