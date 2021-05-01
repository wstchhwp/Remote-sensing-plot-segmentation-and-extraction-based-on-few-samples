import cv2 
import numpy as np
img = cv2.imread("end/B224Test2.tif",1)
#第二个参数是通道数和位深的参数，
#IMREAD_UNCHANGED = -1#不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
#IMREAD_GRAYSCALE = 0#进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
#IMREAD_COLOR = 1#进行转化为RGB三通道图像，图像深度转为8位
#IMREAD_ANYDEPTH = 2#保持图像深度不变，进行转化为灰度图。
#IMREAD_ANYCOLOR = 4#若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位
#print (img)
print (img.shape)
print (img.dtype) 
print (img.min())
print (img.max())
# img[img==1] = 255
# print(img)
# cv2.imwrite("8888888888.png",np.uint8(img),[cv2.IMWRITE_PNG_COMPRESSION, 0])

# img = img[:,:,0] 
mask = np.unique(img) 
tmp = {} 
for v in mask: 
    tmp[v] = [np.sum(img == v)] + [np.sum(img == v)/len(img.flatten())] 
print("mask值为：") 
print(mask) 
print("统计结果：") 
print(tmp)
# # #创建窗口并显示图像
# # cv2.namedWindow("Image")
# # cv2.imshow("Image",img)
# # cv2.waitKey(0)
# # #释放窗口
# # cv2.destroyAllWindows() 
