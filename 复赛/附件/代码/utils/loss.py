import torch
import torch.nn.functional as F  
import numpy as np
from torch import nn
from torch.autograd import Variable
from random import shuffle
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
import cv2


def CE_Loss(inputs, target, num_classes=21):
    """
    Cross Entropy Loss(CE)
    对数据进行softmax,再log，再进行NLLLoss。其与nn.NLLLoss的关系可以描述为：
    softmax(x)+log(x)+nn.NLLLoss====>nn.CrossEntropyLoss
    
    torch.nn.NLLLoss()
    个人理解：感觉像是把target转换成one-hot编码，然后与input点乘得到的结果 
    常用于多分类任务,NLLLoss函数输入input之前，需要对input进行log_softmax处理，即将input转换成概率分布的形式，并且取对数，底数为e
    yk​表示one_hot编码之后的数据标签,实际使用NLLLoss()损失函数时，传入的标签，无需进行one_hot编码
    损失函数运行的结果为yk与经过log_softmax运行的数据相乘，求平均值，在取反。
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)
    # ignore_index(int)- 忽略某一类别，不计算其loss，其loss会为0，并且，在采用size_average时，不会计算那一类的loss，除的时候的分母也不会统计那一类的样本。
    CE_loss  = nn.NLLLoss(ignore_index=num_classes)(F.log_softmax(temp_inputs, dim=-1), temp_target)
    return CE_loss

def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
    # 计算dice loss
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss