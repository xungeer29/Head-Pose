# -*- coding:utf-8 -*- 

"""
提取图像路径及头部三维角度，保存在txt中
"""

import scipy.io as scio
import os
import argparse

def parse_args():
    description = 'Extract pose data from mat file'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_file', type=str,
            default='AFW', help='The path of input file')
    # parser.add_argument('')

    args = parser.parse_args()
    return args

def extractPose(rootpath):
    for datafile in os.listdir(rootpath):
        if datafile.endswith('.mat'):
            data = scio.loadmat(os.path.join(rootpath, datafile))
            print data['Pose_Para'][0][0:3]
            pitch, yaw, roll = data['Pose_Para'][0][0:3]
            imagefile = datafile[0:-4]+'.jpg'
            saver.write(os.path.join(rootpath,imagefile)+' '+str(yaw)
                        +' '+str(pitch)+' '+str(roll)+'\n')

if __name__ == '__main__':
    args = parse_args()
    #savefile = args.input_file.split('/')[-2]+'.txt'
    #saver = open(savefile, 'a')
    saver = open('AFLW2000.txt', 'a')
    extractPose(args.input_file)
