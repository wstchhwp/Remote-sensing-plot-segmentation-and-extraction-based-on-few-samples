# -*- coding: UTF-8 -*-
# 对单张图片进行预测
from PIL import Image
from unet_predict import UNet
from cattunet_predict import CAttUNet
from attunet_predict import AttUNet
from fcn_resnet50_predict import SEResNet50
from fcn_resnet101_predict import SEResNet101

#cattunet = CAttUNet()
#attunet = AttUNet()
#unet = UNet()
#seresnet50 = SEResNet50()
seresnet101 = SEResNet101()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        #r_image = unet.detect_image(image)
        r_image = seresnet101.detect_image(image)
        #r_image.show()
        #r_image.save(img.split('.png')[0]+'x.png')
