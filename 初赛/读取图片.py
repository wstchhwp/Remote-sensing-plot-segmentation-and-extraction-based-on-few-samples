

import numpy as np
import cv2 
from PIL import Image
img = cv2.imread("SE_ResNet101_fcn_2.png",1)
img = cv2.resize(img,(600,500))
cv2.imwrite("xu2.png",np.uint8(img),[cv2.IMWRITE_PNG_COMPRESSION, 0])
# #第二个参数是通道数和位深的参数，
# #IMREAD_UNCHANGED = -1#不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
# #IMREAD_GRAYSCALE = 0#进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
# #IMREAD_COLOR = 1#进行转化为RGB三通道图像，图像深度转为8位
# #IMREAD_ANYDEPTH = 2#保持图像深度不变，进行转化为灰度图。
# #IMREAD_ANYCOLOR = 4#若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位
img = img[:,:,0] 
print(img.shape)

# # img01[img01==1] = 255
# # print(img01)
# # cv2.imwrite("end2.png",np.uint8(img01),[cv2.IMWRITE_PNG_COMPRESSION, 0])

mask = np.unique(img) 
tmp = {} 
for v in mask: 
    tmp[v] = [np.sum(img == v)] + [np.sum(img == v)/len(img.flatten())] 
print("mask值为：") 
print(mask) 
print("统计结果：") 
print(tmp)

# z16 = (img01.astype(np.uint16))
# im = Image.fromarray(img.astype(np.uint16))                      
# im.save('image2.png')  