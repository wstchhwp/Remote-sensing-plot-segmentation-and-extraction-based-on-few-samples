import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft Yahei', 'SimHei', 'sans-serif'] #  全局设置支持中文字体，默认 sans-serif
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


filePath = "/home/wp/pytorch/FCN/logs/"
dir_list = os.listdir(filePath)

loss = {}
epoch_list = []
train_loss = []
val_loss = []
for i in dir_list:
    epoch_loss = []
    epoch = int(i.split("Epoch")[1].split("-Train_Loss")[0])
    epoch_train_loss = float(i.split("Train_Loss")[1].split("-")[0])
    epoch_loss.append(epoch_train_loss)
    epoch_val_loss = float(i.split("Val_Loss")[1].split(".pth")[0])
    epoch_loss.append(epoch_val_loss)
    loss[epoch] = epoch_loss
for j in sorted(loss):
    epoch_list.append(j)
    train_loss.append(loss[j][0])
    val_loss.append(loss[j][1])
print(epoch_list)
print(train_loss)
print(val_loss)

plt.figure(figsize=(10,8))
# 子图1

plt.xlabel('epoch',fontsize=15)
plt.ylabel('train_loss',fontsize=15)
plt.plot(epoch_list, val_loss, color='red')
plt.title('Attention U-Net-Val',fontsize=20)  
plt.savefig("Attention U-Net-Val.png",dpi=1500)



