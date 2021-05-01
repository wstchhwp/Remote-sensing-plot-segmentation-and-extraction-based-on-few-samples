# -*- coding:utf-8 -*
from PIL import Image
import numpy as np
import random
import copy
import os
import cv2


def generate_matrix(num_class, gt_image, pre_image):
    '''
    计算混淆矩阵
    '''
    # ground truth中所有正确(值在[0, classe_num])的像素label的mask
    mask = (gt_image >= 0) & (gt_image < num_class)
    # print("mask",mask.shape)
    # print("gt_image[mask]",gt_image[mask].shape)
 
    label = num_class * gt_image[mask].astype('int') + pre_image[mask] 
    # print("*****",label)
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class**2)
    # print(count.shape)
    # 21 * 21(for pascal)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix

def Pixel_Accuracy(confusion_matrix):
    '''
    Pixel Accuracy (PA)
    正确的像素占总像素的比例
    '''
    Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return Acc

def Pixel_Accuracy_Class(confusion_matrix):
    '''
    Mean Pixel Accuracy (MPA)
    分别计算每个类分类正确的概率
    '''
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    Acc = np.nanmean(Acc)
    return Acc

def Class_IOU(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    return MIoU   
 
def Mean_Intersection_over_Union(confusion_matrix, background):
    '''
    Mean Intersection over Union (MIoU)
    对于每个类别计算出的IoU求和取平均
    '''
    MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    if background == False:
        MIoU = np.nanmean(MIoU[1:]) # 跳过0值求mean,shape:[21]
    else:
        MIoU = np.nanmean(MIoU)
    return MIoU

def Frequency_Weighted_Intersection_over_Union(confusion_matrix, background):
    '''
    Frequency Weighted Intersection over Union (FWIoU)
    可以理解为根据每一类出现的频率对各个类的IoU进行加权求和
    '''
    if background == False:
        confusion_matrix = confusion_matrix[1:4,1:4]
        # print(confusion_matrix)
        freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
  
        iu = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()

    else:
        freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
        iu = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU
  
    
num_class = 2
# 类别0-->背景-->黑色
# 类别1-->圆形-->红色
# 类别2-->方形-->绿色
# 类别3-->三角形-->蓝色
img_label_true_path = "图像分割/FCN/提交结果/SENet+ResNet101+FCN_true/"
model_pred_path = "图像分割/FCN/提交结果/SENet+ResNet101+FCN_pred/"
confusion_matrix = np.zeros((num_class,)*2)


print("正在处理：SENet+ResNet101+FCN")
for true_img in os.listdir(img_label_true_path):
    pred_img = Image.open(model_pred_path + true_img)
    true_img = Image.open(img_label_true_path + true_img)
    H = np.array(true_img).shape[0]
    W = np.array(true_img).shape[1]
    im_size = pred_img.resize((W,H),Image.BILINEAR)
    # print("预测形状：", np.array(im_size).shape)
    # print("真实形状：", np.array(true_img).shape)
    pre_image = np.array(im_size)
    gt_image = np.array(true_img)
    matrix = generate_matrix(num_class, gt_image.flatten(), pre_image.flatten())
    # 矩阵相加是各个元素对应相加,即4*4的矩阵进行pixel-wise加和
    confusion_matrix += matrix
# print("confusion_matrix:\n", confusion_matrix)
PA = Pixel_Accuracy(confusion_matrix)
print("PA:\n ", PA)
MPA = Pixel_Accuracy_Class(confusion_matrix)
print("MPA:\n ", MPA)  
Class_IoU = Class_IOU(confusion_matrix) 
# print("Class_IoU:\n ", Class_IoU)    
N_MIoU = Mean_Intersection_over_Union(confusion_matrix,background=False)
MIoU = Mean_Intersection_over_Union(confusion_matrix,background=True)
# print("不包含背景 MIoU:\n ", N_MIoU)  
print("包含背景 MIoU:\n ", MIoU)  
    
# N_FWIoU = Frequency_Weighted_Intersection_over_Union(confusion_matrix,background=False)
# print("不包含背景 FWIoU:\n ", N_FWIoU)
FWIoU = Frequency_Weighted_Intersection_over_Union(confusion_matrix,background=True)
print("包含背景 FWIoU:\n", FWIoU)