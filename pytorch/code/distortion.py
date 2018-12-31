import cv2
from PIL import Image, ImageFilter
imgs = ['01.jpeg', '02.jpeg', '03.png', '04.jpg', '05.JPG', '06.jpg']
root = '/data2/gaofuxun/liveness/Head-Pose/pytorch/data/'

for img in imgs:
    im01 = cv2.imread(root+img)
    im02 = Image.open(root+img)
    im02 = im02.filter(ImageFilter.Kernel((3,3),(1,1,1,0,0,0,2,0,2)))
    im02.save('../figs/blur_'+img.split('.')[0]+'.jpg')

