from utils import *
import numpy as np
import cv2

#imgpath = '/data2/gaofuxun/data/head-pose/300W_LP/AFW/AFW_1051618982_1_10'
#imgpath = '/data2/gaofuxun/data/head-pose/300W_LP/LFPW/LFPW_image_test_0001_11'
#imgpath = '/data2/gaofuxun/data/head-pose/300W_LP/HELEN/HELEN_100040721_2_4'
imgpath = '/data2/gaofuxun/data/head-pose/300W_LP/IBUG/IBUG_image_014_02_10'
im = cv2.imread(imgpath+'.jpg')
mat = sio.loadmat(imgpath+'.mat')
pre_pose_params = mat['Pose_Para'][0]
pitch, yaw, roll = pre_pose_params[:3]
pitch = pitch*180/np.pi
yaw = yaw*180/np.pi
roll = roll*180/np.pi

pt2d = mat['pt2d']
xmin = min(pt2d[0,:])
ymin = min(pt2d[1,:])
xmax = max(pt2d[0,:])
ymax = max(pt2d[1,:])

k = np.random.random_sample() * 0.2 + 0.2
#x_min -= 0.6 * k * abs(xmax-xmin)
#y_min -= 2 * k * abs(ymax - ymin)
#x_max += 0.6 * k * abs(xmax - xmin)
#y_max += 0.6 * k * abs(ymax - ymin)

print '{} {} {} {} {} {} {}'.format(yaw, pitch, roll, xmin, ymin, xmax, ymax)
draw_axis(im, yaw, pitch, roll, (xmin+xmax)/2, (ymin+ymax)/2, (ymax-ymin)/2)
cv2.imwrite('300w-lp_006.jpg', im)
cv2.imwrite('300w-lp_006.tif', im)
#cv2.imwrite('300w-lp_001.eps', im)
