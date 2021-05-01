import numpy as np
import torch
import torch.nn.functional as F  


def generate_matrix(label_true, label_pred, n_class):  
    """
    计算混淆矩阵
    """
    # ground truth中所有正确(值在[0, classe_num])的像素label的mask
    mask = (label_true >= 0) & (label_true < n_class)
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    confusion_matrix = np.bincount(n_class*label_true[mask].astype(int)+label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return confusion_matrix

def accuracy_score(label_trues, label_preds, n_class=21):
    confusion_matrix = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        confusion_matrix += generate_matrix(lt.flatten(), lp.flatten(), n_class)  
    # Pixel Accuracy (PA) 正确的像素占总像素的比例
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    # Mean Pixel Accuracy (MPA) 分别计算每个类分类正确的概率
    acc_cls = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    # Mean Intersection over Union (MIoU) 对于每个类别计算出的IoU求和取平均
    iu = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))
    mean_iu = np.nanmean(iu)
    # Frequency Weighted Intersection over Union (FWIoU) 可以理解为根据每一类出现的频率对各个类的IoU进行加权求和
    freq = confusion_matrix.sum(axis=1) / confusion_matrix.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc

def dice_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)
    # [1, 230400, 2]
    # 计算dice系数
    # torch.gt()
    # 逐元素比较input和other, 即是否(input > other)。 如果两个张量有相同的形状和元素值，则返回True,否则False。 
    # 第二个参数可以为一个数或与第一个参数相同形状和类型的张量。
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    # temp_target[...,:-1]意思是去掉最后一个通道
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    
    return score