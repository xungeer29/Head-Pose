# -*- coding:utf-8 -*-

"""
绘制模型迭代曲线图
"""
import matplotlib.pyplot as plt
plt.switch_backend('agg')

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 9,
         }

logdir = '../log/hopenet.txt'
log = open(logdir, 'r')
lines = log.readlines()
cls_yaw, cls_pitch, cls_roll = [], [], []
mse_yaw, mse_pitch, mse_roll = [], [], []
total_yaw, total_pitch, total_roll = [], [], []
num = 0
for line in lines:
    num += 1
    if num < 2359:
        continue
    try:
        iters, clsloss, mseloss, totalloss = line.split('|')
    except:
        continue
        #break
    
    # clsloss
    yaw, pitch, roll = clsloss.split(',')
    yaw = yaw.split(' ')[-1]
    pitch = pitch.split(' ')[-1]
    roll = roll.strip().split(' ')[-1]
    print yaw, pitch, roll
    cls_yaw.append(float(yaw))
    cls_pitch.append(float(pitch))
    cls_roll.append(float(roll))

def smooth(scalar, weight):
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

cls_yaw = smooth(cls_yaw, 0.9)
cls_pitch = smooth(cls_pitch, 0.9)
cls_roll = smooth(cls_roll, 0.9)

x = [x for x in range(len(cls_yaw))]
plt.plot(x, cls_yaw, color='b', label='yaw')
plt.plot(x, cls_pitch, color='g', label='pitch')
plt.plot(x, cls_roll, color='r', label='roll')
plt.legend(loc='upper right', prop=font1, frameon=False)
plt.xlabel('iterationos')
plt.ylabel('loss')
plt.savefig('../figs/clsloss.jpg')
plt.savefig('../figs/clsloss.eps')
plt.savefig('../figs/clsloss.tif')

