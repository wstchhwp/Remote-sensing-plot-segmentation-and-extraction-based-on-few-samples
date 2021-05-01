import numpy as np
import cv2
import os

img = cv2.imread("图像分割/FCN/最终提交/Test3.png",cv2.IMREAD_GRAYSCALE)
#img = cv2.resize(img,(600,500)) 
# print(img)
print(np.unique(img))
# img[img==255] = 1
print(img.shape)
print(np.sum(img==0))
print(np.sum(img==1))
print(np.sum(img==1)/(600*500))

# img_label_true_path = "图像分割/FCN/提交结果/集成学习/"
# for true_img in os.listdir(img_label_true_path):
#     img = cv2.imread(img_label_true_path+true_img, cv2.IMREAD_GRAYSCALE)
#     img[img==1] = 255
#     # 保存图像在指定的路径下
#     cv2.imwrite(img_label_true_path+true_img, img)
