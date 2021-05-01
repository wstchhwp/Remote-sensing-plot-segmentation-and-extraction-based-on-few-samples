from fcn_senet import FCN1s as fcn
from fcn_senet import SE_ResNet50
from unet_senet import AttU_Net
from unet import UNet
from torch import nn
from PIL import Image
from torch.autograd import Variable
import models
import torch.nn.functional as F  
import numpy as np
import colorsys
import torch
import copy
import cv2 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class FCN1s(object):
    #--------------------------------------------------------------#
    #   注意修改model_path、num_classes和backbone,使其符合自己的模型
    #--------------------------------------------------------------#
    _defaults = {
        "model_path"        :   '/home/wp/pytorch/FCN/logs/Epoch100-Train_Loss0.5110-Val_Loss0.5056.pth',
        "model_image_size"  :   (480, 480, 3),
        "num_classes"       :   2,
        "cuda"              :   True,
        "blend"             :   False, #True
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()

    def generate(self):
        # 获得所有的分类
        backbone = SE_ResNet50()        
        #self.net = fcn(pretrained_net=backbone, n_class=self.num_classes)  
        self.net = AttU_Net(img_ch=3, output_ch=2)
        #self.net = UNet(num_classes=2)
        self.net = self.net.eval()
        # 加载训练好的模型
        state_dict = torch.load(self.model_path)
        self.net.load_state_dict(state_dict, strict=False)
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
            # 画框设置不同的颜色
            hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def letterbox_image(self, image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image,nw,nh
  
    def detect_image(self, image):
        """
        检测图片
        """
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        images = [np.array(image)/255] # 增加batchsize维度
        images = np.transpose(images,(0,3,1,2)) 
        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images =images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.model_image_size[0]-nh)//2):int((self.model_image_size[0]-nh)//2+nh), int((self.model_image_size[1]-nw)//2):int((self.model_image_size[1]-nw)//2+nw)]
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        print(pr.shape)
 
        cv2.imwrite("/home/wp/pytorch/FCN/results/Attention U-Net_2.png",np.uint8(pr),[cv2.IMWRITE_PNG_COMPRESSION, 0])
        for c in range(self.num_classes):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( self.colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( self.colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( self.colors[c][2] )).astype('uint8')
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        if self.blend:
            image = Image.blend(old_img,image,0.5)
        return image

