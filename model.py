import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.autograd as dif
import torchvision.models

""" -------------------------------------------------------------------------"""
""" Classifier """
class Custom(nn.Module):
    def __init__(self, params):
        super(Custom, self).__init__()
        self.ndf = params.ndf
        self.nc = params.nc
        self.nClass = params.nClass
        self.conv1 = nn.Sequential(
            ## input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias = False),
            nn.ReLU(True),
            ## state size. (ndf) x 32 x 32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.ReLU(True),
            ## state size. (ndf*2) x 16 x 16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.ReLU(True),
            ## state size. (ndf*4) x 8 x 8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.ReLU(True),
            ## state size. (ndf*8) x 4 x 4
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(self.ndf * 8, self.ndf, 4, 1, 0, bias = False),
            nn.ReLU(True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.ndf, self.nClass),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, self.ndf)
        x = self.fc1(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, params):
        super(MobileNet, self).__init__()
        self.nClass = params.nClass
        self.resnet = nn.Sequential(
            torchvision.models.mobilenet_v2(pretrained=True, progress=True),
            nn.ReLU(True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128, self.nClass),
        )
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        return x

class ResNet(nn.Module):
    def __init__(self, params):
        super(ResNet, self).__init__()
        self.nClass = params.nClass
        self.resnet = nn.Sequential(
            torchvision.models.resnet18(pretrained=True, progress=True),
            nn.ReLU(True)
        )
        # net = nn.Sequential(*list(net.children())[:2])
        self.fc1 = nn.Sequential(
            nn.Linear(128, self.nClass),
        )
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        return x