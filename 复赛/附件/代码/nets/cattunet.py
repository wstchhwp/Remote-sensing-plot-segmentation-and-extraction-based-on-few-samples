# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x = self.up(x)
        return x

class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CoordAtt, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 1),
            nn.BatchNorm2d(channels//reduction),
            nn.ReLU(inplace=True))
        self.xfc = nn.Conv2d(channels//reduction, channels, 1)
        self.yfc = nn.Conv2d(channels//reduction, channels, 1)

    def forward(self, x):
        B, _, H, W = x.size()
        # X Avg Pool and Y Avg Pool
        xap = F.adaptive_avg_pool2d(x, (H, 1))
        yap = F.adaptive_avg_pool2d(x, (1, W))
        # Concat+Conv2d+BatchNorm+Non-linear
        mer = torch.cat([xap.transpose_(2, 3), yap], dim=3)
        fc1 = self.fc1(mer)
        # split
        xat, yat = torch.split(fc1, (H, W), dim=3)
        # Conv2d-Sigmoid and Conv2d-Sigmoid
        xat = torch.sigmoid(self.xfc(xat))
        yat = torch.sigmoid(self.yfc(yat))
        # Attention Multiplier
        out = x * xat * yat
        return out


class CAttU_Net(nn.Module):
    def __init__(self, num_classes):
        super(CAttU_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=3, ch_out=64)
        self.ECAtt1 = CoordAtt(channels=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.ECAtt2 = CoordAtt(channels=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.ECAtt3 = CoordAtt(channels=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.ECAtt4 = CoordAtt(channels=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.ECAtt5 = CoordAtt(channels=1024)
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.DCAtt5 = CoordAtt(channels=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.DCAtt4 = CoordAtt(channels=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.DCAtt3 = CoordAtt(channels=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.DCAtt2 = CoordAtt(channels=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码通道
        x1 = self.Conv1(x)
        x1 = self.ECAtt1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x2 = self.ECAtt2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x3 = self.ECAtt3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x4 = self.ECAtt4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.ECAtt5(x5)
        # 解码通道
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        x4 = self.DCAtt5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        x3 = self.DCAtt4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        x2 = self.DCAtt3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        x1 = self.DCAtt2(d2)

        d1 = self.Conv_1x1(d2)
        #d1 = self.sigmoid(d1)
        return d1


if __name__=='__main__':
    input = torch.randn(1, 3, 480, 480)
    model = CAttU_Net(num_classes=2)
    print(model)
    out = model(input)
    print(out.shape)
