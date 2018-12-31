# -*- coding:utf-8 -*- 

"""
提取图像路径及头部三维角度，保存在txt中
"""

import scipy.io as scio
import os
import argparse
import numpy as np

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
            bbox = data['pt2d']
            xmin = min(bbox[0,:])
            ymin = min(bbox[1,:])
            xmax = max(bbox[0,:])
            ymax = max(bbox[1,:])

            # k=0.2 to 0.4
            k = np.random.random_sample() * 0.2 + 0.2
            x_min -= 0.6 * k * abs(x_max - x_min)
            y_min -= 2 * k * abs(y_max - y_min)
            x_max += 0.6 * k * abs(x_max - x_min)
            y_max += 0.6 * k * abs(y_max - y_min)

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
