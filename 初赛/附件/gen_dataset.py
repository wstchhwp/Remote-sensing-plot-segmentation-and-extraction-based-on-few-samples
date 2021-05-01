# coding=utf-8
import cv2
import random
import os
import numpy as np
from tqdm import tqdm


img_w, img_h = 600, 500
 
def rotate(src_img,label_img, angle):
    """
    旋转
    """
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    src_img = cv2.warpAffine(src_img, M_rotate, (img_w, img_h))
    label_img = cv2.warpAffine(label_img, M_rotate, (img_w, img_h))
    return src_img, label_img

def gamma_transform(src_img, gamma):
    """
    光照增强
    """
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(src_img, gamma_table)

def random_gamma_transform(src_img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(src_img, gamma)

def blur(src_img):
    """
    模糊--均值滤波
    """
    src_img = cv2.blur(src_img, (3, 3));
    return src_img

def add_noise(src_img):
    """
     添加点噪声
    """
    for i in range(200):
        temp_x = np.random.randint(0,src_img.shape[0])
        temp_y = np.random.randint(0,src_img.shape[1])
        src_img[temp_x][temp_y] = 255
    return src_img
    
def data_augment(src_img, label_img):
    # 旋转90°
    if np.random.random() < 0.20:
        src_img,label_img = rotate(src_img,label_img,90)
    # 旋转180°
    if np.random.random() < 0.35:
        src_img,label_img = rotate(src_img,label_img,180)
    # 旋转270°
    if np.random.random() < 0.25:
        src_img,label_img = rotate(src_img,label_img,270)
    # flipcode > 0：沿y轴翻转
    if np.random.random() < 0.50:
        src_img = cv2.flip(src_img, 1)  
        label_img = cv2.flip(label_img, 1)
    # 光照调整
    if np.random.random() < 0.25:
        src_img = random_gamma_transform(src_img,1.0)
    # 模糊
    if np.random.random() < 0.25:
        src_img = blur(src_img)
    # 增加噪声（高斯噪声，椒盐噪声）
    if np.random.random() < 0.2:
        src_img = add_noise(src_img) 
    return src_img, label_img

def creat_dataset(image_num=3000):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        # image ->3 channels
        src_img = cv2.imread('dataset/src/' + image_sets[i])  
        # label ->single channel
        label_img = cv2.imread('dataset/label/' + image_sets[i], cv2.IMREAD_GRAYSCALE)  
        X_height, X_width, _ = src_img.shape
        while count < image_each:
            src_aug, label_aug = data_augment(src_img, label_img)
            visualize = np.zeros((256,256)).astype(np.uint8)
            visualize = label_aug *50
            cv2.imwrite(('/home/wp/pytorch/FCN/VOCdevkit/VOC2012/JPEGImages/%d.png' % g_count), src_aug)
            cv2.imwrite(('/home/wp/pytorch/FCN/VOCdevkit/VOC2012/SegmentationClass/%d.png' % g_count), label_aug)
            count += 1 
            g_count += 1

if __name__=='__main__':  
    image_sets = ['Data1.png','Data2.png','Data3.png','Data4.png','Data5.png','Data6.png','Data7.png','Data8.png']
    creat_dataset()
