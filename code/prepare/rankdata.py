# -*- coding:utf-8 -*-

"""
数据预处理: 将角度分成不同的类
"""

import numpy as np
import os

# trainset
rootpath = '300W_LP.txt'
savepath = '300W_LP_rank.txt'

# validation set
# rootpath = 'AFLW2000.txt'
# savepath = 'AFLW2000_rank.txt'

def value2cls(pose_file):
    pose_data = open(pose_file, 'r')
    num = 0
    failNum = 0
    for line in pose_data:
        # print line
        try:
            img_path, yaw, pitch, roll = line.split(' ')
        except:
            failNum += 1
            continue
        # convert to degree.
        yaw = float(yaw) * 180.0 / np.pi
        pitch = float(pitch) * 180.0 / np.pi
        roll = float(roll) * 180.0 / np.pi
        # value convert to cls
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins)

        saver.write(img_path+' '+str(binned_pose[0])+' '+
                    str(binned_pose[1])+' '+str(binned_pose[2])+'\n')
        num += 1
        print('Success {} fail {}'.format(num, failNum))


if __name__ == "__main__":
    saver = open(savepath, 'w')
    value2cls(rootpath)
    saver.close()

