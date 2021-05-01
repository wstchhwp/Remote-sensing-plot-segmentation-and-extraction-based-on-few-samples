# -*- coding: UTF-8 -*-
import copy
import numpy as np
import cv2 
from PIL import Image
import colorsys
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from nets.fcn_senet import SE_ResNet50, SE_ResNet101, FCN1s
from nets.unet import U_Net
from nets.attunet import AttU_Net
from nets.cattunet import CAttU_Net   
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class SEResNet50(object):
    #--------------------------------------------------------------#
    #   注意修改model_path、num_classes和backbone,使其符合自己的模型
    #--------------------------------------------------------------#
    _defaults = {
        "model_path"        :   '图像分割/FCN/logs_fcn_resnet50/Epoch100-Train_Loss0.5064-Val_Loss0.4850.pth',
        "model_image_size"  :   (480, 480, 3),
        "num_classes"       :   2,
        "cuda"              :   True,
        "blend"             :   False, #True
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    def generate(self):
        #*******************ResNet50+SENet+FCN***********
        backbone = SE_ResNet50() 
        self.net = FCN1s(pretrained_net=backbone, n_class=2)  
        # 加载训练好的模型
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict, strict=False)
        self.net = self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        if self.num_classes <= 21: # (128, 0, 0)
            self.colors = [(0, 0, 0), (255, 255, 255), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                    (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                    (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def letterbox_image(self, image, size):
        # 一般来说，图片尺寸比网络输入尺寸要大
        # 图片尺寸
        iw, ih = image.size
        # 网络输入尺寸
        w, h = size
        # 转换的最小比例，以最小比例为准，进行放缩，维持图片的长宽比不变
        scale = min(w/iw, h/ih)
        # 保证长或宽，至少一个符合目标图像的尺寸
        nw = int(iw*scale)
        nh = int(ih*scale)
        # 重采样–可选的重采样过滤器
        # PIL.Image.NEAREST（使用最近的邻居），PIL.Image.BILINEAR（线性插值），
        # PIL.Image.BICUBIC（三次样条插值），PIL.Image.LANCZOS（高质量的下采样滤波器）
        image = image.resize((nw,nh), Image.BICUBIC)
        # 新建网络尺寸大小的新图片
        new_image = Image.new('RGB', size, (128,128,128))
        # 将图像填充为中间图像，两侧为灰色的样式
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image, nw, nh
  
    def detect_image(self, image):
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # 添加灰度条
        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        images = [np.array(image)/255] # 增加batchsize维度
        images = np.transpose(images,(0,3,1,2)) 
        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            # 单通道
            pr = F.softmax(pr.permute(1,2,0),dim=-1).cpu().numpy().argmax(axis=-1)
            # 去掉灰度条，先高再宽
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
        #print(pr.shape)
        cv2.imwrite("图像分割/FCN/提交结果/SENet+ResNet50+FCN_2996.png",np.uint8(pr),[cv2.IMWRITE_PNG_COMPRESSION, 0])
        # input_shape[0]表示高h input_shape[1]表示宽w
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        # 为预测结果图片上色
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c)*(self.colors[c][0])).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c)*(self.colors[c][1])).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c)*(self.colors[c][2])).astype('uint8')
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        # 显示结果融合
        if self.blend:
            image = Image.blend(old_img,image,0.5)
        return image