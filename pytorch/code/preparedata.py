# -*- coding:utf-8 -*- 

import os

rootpath = '/data2/gaofuxun/data/head-pose/'

trainfile = '../data/300W-LP.txt'
testfile = '../data/AFLW2000.txt'
train = open(trainfile, 'w')
test = open(testfile, 'w')

# 300W-LP
filedirs = os.listdir(os.path.join(rootpath, '300W_LP'))
for filedir in filedirs:
    if filedir is None:
        break
    imgpath = os.path.join(rootpath,'300W_LP', filedir)
    imgs = os.listdir(imgpath)
    for img in imgs:
        if img is None:
            break
        if img.endswith('.jpg'):
            filename_list = os.path.join('300W_LP', filedir, img.split('.')[0])
            train.write(filename_list+'\n')

# AFLW2000
dirs = os.listdir(os.path.join(rootpath, 'AFLW2000'))
for dir_ in dirs:
    if dir_ is None:
        break
    if dir_.endswith('.jpg'):
        filename_list = os.path.join('AFLW2000', dir_.split('.')[0])
        test.write(filename_list+'\n')

