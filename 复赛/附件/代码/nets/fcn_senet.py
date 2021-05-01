# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.nn import init


def Conv1(in_planes, places):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=1, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

class SE_Module(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SE_Module, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=channel // ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=channel // ratio, out_features=channel),
                nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)

class SE_ResNetBlock(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(SE_ResNetBlock,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion))
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # SE layers
        self.fc1 = nn.Conv2d(places*self.expansion, places*self.expansion//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(places*self.expansion//16, places*self.expansion, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))        
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class SE_ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(SE_ResNet,self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=3, places=64)
        self.layer1 = self.make_layer(in_places=64,  places=64, block=blocks[0], stride=2)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(SE_ResNetBlock(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(SE_ResNetBlock(places*self.expansion, places))
        return nn.Sequential(*layers)

    def forward(self, x):
        out_put = []
        x = self.conv1(x)
        out_put.append(x)
        x = self.layer1(x)
        out_put.append(x)
        x = self.layer2(x)
        out_put.append(x)
        x = self.layer3(x)
        out_put.append(x)
        x = self.layer4(x)
        out_put.append(x)
        #x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        #x = self.fc(x)
        return out_put  

def SE_ResNet50():
    return SE_ResNet([3, 4, 6, 3])

def SE_ResNet101():
    return SE_ResNet([3, 4, 23, 3])


class FCN1s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output[4]  # [None,2048,15,15]
        x4 = output[3]  # [None,1024,30,30]
        x3 = output[2]  # [None,512,60,60]
        x2 = output[1]  # [None,256,120,120]
        x1 = output[0]  # [None,64,240,240]
        score = self.relu(self.deconv1(x5))              
        score = self.bn1(score+x4)  
        #print(score.shape)                         
        score = self.relu(self.deconv2(score))               
        score = self.bn2(score+x3)  
        #print(score.shape)                          
        score = self.relu(self.deconv3(score))               
        score = self.bn3(score+x2)  
        #print(score.shape)
        score = self.relu(self.deconv4(score))               
        score = self.bn4(score+x1) 
        #print(score.shape)
        score = self.relu(self.deconv5(score))  
        #print(score.shape)                             
        score = self.classifier(score)                       
        return score


if __name__=='__main__':
    model = SE_ResNet50()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)





