'''
@,@Author: ,: your name
@,@Date: ,: 2021-01-03 13:42:52
@,@LastEditTime: ,: 2021-01-07 16:56:32
@,@LastEditors: ,: your name
@,@Description: ,: In User Settings Edit
@,@FilePath: ,: \PSPNet-pytorch\VOCdevkit\voc2pspnet.py
'''
import os
import random 
 
segfilepath=r'./VOCdevkit/VOC2012/SegmentationClass'
saveBasePath=r"./VOCdevkit/VOC2012/ImageSets/Segmentation/"
 
trainval_percent=1
train_percent=0.85

temp_seg = os.listdir(segfilepath)
total_seg = []
for seg in temp_seg:
    if seg.endswith(".png"):
        total_seg.append(seg)

num=len(total_seg)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("traub suze",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
for i  in list:  
    name=total_seg[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()
