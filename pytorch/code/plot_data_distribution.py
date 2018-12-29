# -*- coding:utf-8 -*- 

import os
import numpy as np
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
plt.switch_backend('agg')

root = '/data2/gaofuxun/data/head-pose'
trainfile = '../data/300W-LP.txt'
testfile = '../data/AFLW2000.txt'

#savetrain = '../data/data_distribution_300w-lp.txt'
#savetest = '../data/data_distribution_aflw2000.txt'
save = '../data/data_distribution.txt'

traindata = open(trainfile, 'r')
testdata = open(testfile, 'r')

#writetrain = open(savetrain, 'w')
#wrritetest = open(savetest, 'w')
saver = open(save, 'w')

bins = np.array(range(-99, 102, 3))

pitch_train, yaw_train, roll_train = [], [], []
for line in traindata.readlines():
    if line is None:
        break
    line = line.strip()
    mat = sio.loadmat(os.path.join(root, line+'.mat'))
    pre_pose_params = mat['Pose_Para'][0]
    pose_params = pre_pose_params[:3]
    #print pose_params[0], pose_params[1], pose_params[2]
    pitch_train.append(pose_params[0]*180/np.pi)
    yaw_train.append(pose_params[1]*180/np.pi)
    roll_train.append(pose_params[2]*180/np.pi)
#print yaw_train, pitch_train, roll_train
binned_pose_train = np.digitize([yaw_train, pitch_train, roll_train], bins)

pitch_test, yaw_test, roll_test = [], [], []
for line in testdata.readlines():
    if line is None:
        break
    line = line.strip()
    mat = sio.loadmat(os.path.join(root, line+'.mat'))
    pre_pose_params = mat['Pose_Para'][0]
    pose_params = pre_pose_params[:3]
    pitch_test.append(pose_params[0]*180/np.pi)
    yaw_test.append(pose_params[1]*180/np.pi)
    roll_test.append(pose_params[2]*180/np.pi)
binned_pose_test = np.digitize([yaw_test, pitch_test, roll_test], bins)

num_yaw_300w_lp = np.zeros(68)
num_pitch_300w_lp = np.zeros(68)
num_roll_300w_lp = np.zeros(68)
# yaw
for i in binned_pose_train[0,:]:
    num_yaw_300w_lp[i] += 1
# pitch
for i in binned_pose_train[1,:]:
    num_pitch_300w_lp[i] += 1
# roll
for i in binned_pose_train[2,:]:
    num_roll_300w_lp[i] += 1

num_yaw_aflw = np.zeros(68)
num_pitch_aflw = np.zeros(68)
num_roll_aflw = np.zeros(68)
# yaw
for i in binned_pose_test[0,:]:
    #print i
    num_yaw_aflw[i] += 1
# pitch
for i in binned_pose_test[1,:]:
    num_pitch_aflw[i] += 1
# roll
for i in binned_pose_test[2,:]:
    num_roll_aflw[i] += 1

saver.write('300W-LP\nyaw:{}\npitch:{}\nroll:{}\nAFLW\nyaw:{}\npitch:{}\nroll:{}\n'.format(
            num_yaw_300w_lp, num_pitch_300w_lp, num_roll_300w_lp, 
            num_yaw_aflw, num_pitch_aflw, num_roll_aflw))

font1 = {'family': 'Times New Roman',  
         'weight': 'normal',  
         'size': 9,  
}
x = [x for x in range(68)]
#plt.plot([num_yaw_300w_lp, num_pitch_300w_lp, num_roll_300w_lp],color=('b','g','r'))
plt.plot(x, num_yaw_300w_lp,color='b', label='yaw')
plt.plot(x, num_pitch_300w_lp,color='g', label='pitch')
plt.plot(x, num_roll_300w_lp,color='r', label='roll')
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.xlabel('bins')
plt.ylabel('number')
plt.savefig('../figs/plot_300w_lp.jpg')
plt.savefig('../figs/plot_300w_lp.eps')
plt.savefig('../figs/plot_300w_lp.tif')

plt.figure()
plt.plot(x, num_yaw_aflw, color='b', label='yaw')
plt.plot(x, num_pitch_aflw, color='g', label='pitch')
plt.plot(x, num_roll_aflw, color='r', label='roll')
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.xlabel('bins')
plt.ylabel('number')
plt.savefig('../figs/plot_aflw.jpg')
plt.savefig('../figs/plot_aflw.eps')
plt.savefig('../figs/plot_aflw.tif')
