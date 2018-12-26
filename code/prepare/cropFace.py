# -*- coding:utf-8 -*-

"""
使用 MTCNN crop 出人脸位置，要适当扩大区域以完全 crop 出头部
"""

from mtcnn import usemtcnn
import cv2
import os

# train
rankPosetxt = '../data/300W_LP_rank.txt'
savepath = '../data/300W_LP_bbox.txt'
# validation
# rankPosetxt = '../data/AFLW2000_rank.txt'
# savepath = '../data/AFLW2000_bbox.txt'

# 存放数据的根目录
rootpath = '/data2/gaofuxun/data/head-pose/'

trainset = os.path.join(rootpath, 'train')
if not os.path.exists(trainset):
    os.makedirs(trainset)

# crop 的放缩尺寸
scale_rate = 0.35

saver = open(savepath, 'w')
data = open(rankPosetxt, 'r')
num = 0
fail = 0
for line in data:
    imgpath = line.split(' ')[0]
    imgName = imgpath.split('/')[-1]
    imgpath = os.path.join(rootpath, imgpath)
    img = cv2.imread(imgpath)
    try:
        face, bbox, landmark = usemtcnn(img)
    except:
        fail += 1
        continue
    # face
    x_min = bbox[:, 0]
    y_min = bbox[:, 1]
    x_max = bbox[:, 2]
    y_max = bbox[:, 3]

    saver.write(os.path.join(trainset, imgName)+' '+str(float(x_min))+' '
                +str(float(x_max))+' '+str(float(y_min))+' '+str(float(y_max))+'\n')
    # head
    x_min -= 0.6 * scale_rate * abs(x_max - x_min)
    y_min -= scale_rate * abs(y_max - y_min)
    x_max += 0.6 * scale_rate * abs(x_max - x_min)
    y_max += 0.6 * scale_rate * abs(y_max - y_min)
    head = img[int(y_min):int(y_max), int(x_min):int(x_max)]

    cv2.imwrite(os.path.join(trainset, imgName), head)
    num += 1
    print('success {} fail {}'.format(num, fail))
