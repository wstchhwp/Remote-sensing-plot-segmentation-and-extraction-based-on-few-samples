# -*- coding:utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft Yahei', 'SimHei', 'sans-serif'] #  全局设置支持中文字体，默认 sans-serif
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# plt.rcParams['savefig.dpi'] = 100 #图片像素
# plt.rcParams['figure.dpi'] = 120 #分辨率

# x = ['U-Net','AttU-Net','CAttU-Net','SENet+ResNet50+FCN','SENet+ResNet101+FCN']
# y_AP =  [0.8972, 0.9786, 0.9794, 0.9836, 0.9820]
# y_MPA = [0.9266, 0.9842, 0.9847, 0.9875, 0.9864]
# y_MIoU =  [0.7937, 0.9495, 0.9513, 0.9608, 0.9572]
# y_FWIoU = [0.8219, 0.9586, 0.9602, 0.9680, 0.9651]

x = ['CAttU-Net','SENet+ResNet50+FCN','SENet+ResNet101+FCN']
y_AP =  [0.9794, 0.9836, 0.9820]
y_MPA = [0.9847, 0.9875, 0.9864]
y_MIoU =  [0.9513, 0.9608, 0.9572]
y_FWIoU = [0.9602, 0.9680, 0.9651]
plt.figure(figsize=(10,8))
plt.plot(np.arange(len(x)),
        y_AP, 
        linestyle="-",
        color="red",
        marker=".",
        markersize=6,
        label="AP")

plt.plot(np.arange(len(x)),
        y_MPA, 
        linestyle="-",
        color="blue",
        marker="o",
        markersize=6,
        label="MAP")

plt.plot(np.arange(len(x)),
        y_MIoU, 
        linestyle="-",
        color="black",
        marker="s",
        markersize=6,
        label="MIoU")

plt.plot(np.arange(len(x)),
        y_FWIoU, 
        linestyle="-",
        color="green",
        marker="*",
        markersize=6,
        label="FWIoU")



plt.xticks(range(len(x)),x)
# 添加x轴标签
plt.xlabel('model',size=15)
plt.ylabel('score',size=15)

# # 添加图形标题
#plt.title("Performance of different deep learning semantic segmentation models on evaluation indicators",size=15)
# 添加图例
plt.legend()
plt.savefig("图像分割/FCN/results.png")
# 显示图形
plt.show()