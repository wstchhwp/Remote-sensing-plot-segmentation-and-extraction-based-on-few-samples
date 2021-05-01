# -*- coding: UTF-8 -*-
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import numpy as np 
from tqdm import tqdm
from torchvision import models
from torch.autograd import Variable
from PIL import Image
from torch import nn
import models
from fcn_training import CE_Loss,Dice_loss
from utils.metrics import f_score
from torch.utils.data import DataLoader
from utils.dataloader import fcn_dataset_collate, FCNDataset
from fcn_senet import SE_ResNet50, FCN1s
from unet_senet import AttU_Net
from unet import UNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_lr(optimizer):
    """
    学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, epoch, epoch_size_train, epoch_size_val, gen_train ,gen_val, Epoch, cuda):
    """
    训练一个世代epoch
    net: 网络模型
    epoch: 一个世代epoch
    epoch_size_train: 训练迭代次数iters
    epoch_size_val: 验证迭代次数iters
    gen_train: 训练数据集
    gen_val: 验证数据集
    Epoch: 总的迭代次数Epoch
    cuda: 是否使用GPU
    """
    train_total_loss = 0
    train_total_f_score = 0
    val_total_loss = 0
    val_total_f_score = 0
    # 开启训练模式
    net.train()
    print('Start Training')
    start_time = time.time()
    with tqdm(total=epoch_size_train, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_train):
            if iteration >= epoch_size_train: 
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
            # 梯度初始化置零
            optimizer.zero_grad()
            # 前向传播，网络输出
            outputs = net(imgs)
            # 计算损失 一次iter即一个batchsize的loss
            loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # 计算f_score
            with torch.no_grad():
                _f_score = f_score(outputs, labels)
            # loss反向传播求梯度
            loss.backward()
            # 更新所有参数
            optimizer.step()
            train_total_loss += loss.item()
            train_total_f_score += _f_score.item()
            waste_time = time.time() - start_time
            pbar.set_postfix(**{'train_loss': train_total_loss / (iteration + 1), 
                                'f_score'   : train_total_f_score / (iteration + 1),
                                's/step'    : waste_time,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            start_time = time.time()
    print('Finish Training')
    # 开启验证模式
    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                outputs  = net(imgs)
                val_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    val_loss  = val_loss + main_dice
                # 计算f_score
                _f_score = f_score(outputs, labels)
                val_total_loss += val_loss.item()
                val_total_f_score += _f_score.item()
            pbar.set_postfix(**{'val_loss'  : val_total_loss / (iteration + 1),
                                'f_score'   : val_total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Train Loss: %.4f || Val Loss: %.4f ' % (train_total_loss/(epoch_size_train+1), val_total_loss/(epoch_size_val+1)))
    print('Saving state, epoch:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Train_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1), train_total_loss/(epoch_size_train+1), val_total_loss/(epoch_size_val+1)))


if __name__ == "__main__":
    inputs_size = [480,480,3]
    log_dir = "logs/"   
    #---------------------#
    # 分类个数+1
    #---------------------#
    NUM_CLASSES = 2
    #--------------------------------------------------------------------#
    # 建议选项：
    # 种类少（几类）时，设置为True
    # 种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    # 种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = True
    #-------------------------------#
    #-------------------------------#
    # Cuda的使用
    #-------------------------------#
    Cuda = True
    # 模型加载
    #-------ResNet50+SENet+FCN----------------
    #backbone = SE_ResNet50()
    #model = FCN1s(pretrained_net=backbone, n_class=NUM_CLASSES)
    # ---------unet-------------------
    #model = UNet(num_classes=2)
    # ---------AttU_Net-----------------
    model = AttU_Net(img_ch=3, output_ch=2)
    #--------------------------------------------------------#
    # 权值文件的下载请看README, 权值和主干特征提取网络一定要对应
    #--------------------------------------------------------#
    model_path = "/home/wp/pytorch/FCN/logs/Epoch81-Train_Loss0.5137-Val_Loss0.5087.pth"
    # 加快模型训练的效率
    print('正在加载预训练模型权重...')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('预训练模型权重加载完成!')
    # 使用多gpu训练
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    # 打开训练数据集的txt
    with open(r"VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt","r") as f:
        train_lines = f.readlines()
    # 打开验证集数据集的txt
    with open(r"VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt","r") as f:
        val_lines = f.readlines() 
        
    if True:
        lr = 1.03e-5
        Interval_Epoch = 81
        Epoch = 100
        Batch_size = 8
        optimizer = optim.Adam(model.parameters(),lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.9)
        # 加载数据集
        train_dataset = FCNDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = FCNDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen_train = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, collate_fn=fcn_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                drop_last=True, collate_fn=fcn_dataset_collate)
        epoch_size_train = max(1, len(train_lines)//Batch_size)
        epoch_size_val = max(1, len(val_lines)//Batch_size)
        # 打开骨干网络
        # for param in model.backbone.parameters():
        #     param.requires_grad = True
        for epoch in range(Interval_Epoch,Epoch):
            fit_one_epoch(model, epoch, epoch_size_train, epoch_size_val, gen_train, gen_val, Epoch, Cuda)
            lr_scheduler.step()




