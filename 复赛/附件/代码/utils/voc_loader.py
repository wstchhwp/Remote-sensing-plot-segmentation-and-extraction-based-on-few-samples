import collections
import os.path as osp
import numpy as np
import PIL.Image
from PIL import Image
# import scipy.io
import torch
from torch.utils import data
import cv2
import random
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb



def letterbox_image(image, label, size):
    label = Image.fromarray(np.array(label))
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))

    return new_image, new_label

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/datasets/voc.py

class VOC2012ClassSeg(data.Dataset):
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])

    def __init__(self, root, split, input_size, num_classes, random_data):
        self.root = root
        self.split = split
        self.input_size = input_size  # 网络输入的尺寸
        self.num_classes = num_classes
        self.random_data = random_data  # 是否使用数据增强
        # 数据集路径
        dataset_dir = osp.join(self.root, 'VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split_file in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, 'ImageSets/Segmentation/%s.txt' % split_file)
            for img_name in open(imgsets_file):
                img_name = img_name.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.png' % img_name)
                lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % img_name)
                self.files[split_file].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def get_random_data(self, image, label, input_size, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))
        h, w = input_size
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2
        scale = rand(0.5,1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        label = label.convert("L")
        # flip image or not
        flip = rand()<.8
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label
        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        return image_data,label

    def __getitem__(self, index):
        data_file = self.files[self.split][index]    
        # 从文件中读取图片
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        # 从文件中读取label        
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        if self.random_data:
            img, lbl = self.get_random_data(img, lbl, (int(self.input_size[1]), int(self.input_size[0])))
        else:
            img, lbl = letterbox_image(img, lbl, (int(self.input_size[1]), int(self.input_size[0])))

        img = np.array(img,dtype='float32')
        lbl = np.array(lbl,dtype='float32')
        lbl[lbl >= self.num_classes] = self.num_classes
        img = np.transpose(np.array(img),[2,0,1])/255

        return img, lbl


"""
vocbase = VOC2012ClassSeg(root="/home/yxk/Downloads/")

print(vocbase.__len__())
img, lbl = vocbase.__getitem__(0)
img = img[:, :, ::-1]
img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
print(np.shape(img))
print(np.shape(lbl))

"""


