#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from fcn import FCN1s
from PIL import Image

fcn = FCN1s()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = fcn.detect_image(image)
        #r_image.show()
        r_image.save("SENet+ResNet50+FCN 2.png")
