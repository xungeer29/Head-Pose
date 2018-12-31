# -*- coding:utf-8 -*-
# 对iqiyi的part1的train进行姿态检测，输出在txt中
# txt：图像路径 yaw pitch roll 图像质量
import platform
print(platform.python_version())
# CUDA_VISIBLE_DEVICES=3 python test_on_video_dlib.py --snapshot /data2/gaofuxun/liveness/deep-head-pose-master/snapshot/ --face_model /data2/gaofuxun/liveness/deep-head-pose-master/dlib_model/shape_predictor_5_face_landmarks.dat --video /data6/shentao/IQIYI_VID_DATA_Part2/IQIYI_VID_TRAIN/IQIYI_VID_TRAIN_0045586.mp4 --output_string result --n_frames 97 --fps 25
import sys, os, argparse
import numpy as np
import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, network, utils

#from skimage import io
#import dlib
#from mtcnn import usemtcnn
from math import cos,sin
import cv2

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=6, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='../models/hopenet_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='../mmod_human_face_detector.dat', type=str)
    # parser.add_argument('--images_path', dest='images_path', help='Path of images', 
	#   default='/data2/gaofuxun/liveness/deep-head-pose-master/IQIYI_Aligned_Face/VAL/')
#    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
#    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
#    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = parser.parse_args()
    return args

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch)*sin(yaw)) + tdy
    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) *sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    # out_dir = '../IQIYI_AlignedFace_POSE/VAL/'
    #images_path = args.images_path
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    # if not os.path.exists(args.images_path):
    #     sys.exit('images does not exist')

    # ResNet50 structure
    model = network.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # Dlib face detection model
#    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    # txt_out = open('pitch.txt', 'w')
    f_out = open('/data2/gaofuxun/data/RankIQA/iqiyi_part2_tid2013_128_ten_crop_train.txt', 'w')
    data_dir = '/data2/gaofuxun/data/RankIQA/iqiyi_part2_train_aligned_qua/'    
    num = 1
    fail = 1
    total_images = len(os.listdir(data_dir))
    fail_image = []
    for image_path in os.listdir(data_dir):
        image_row = cv2.imread(os.path.join(data_dir, image_path))
        img_gray = cv2.imread(os.path.join(data_dir, image_path), 0)
        # print os.path.join(data_dir, image_path)
        # print image_row

        # image = image_row[int(y):(int(y)+int(w)), int(x):(int(x)+int(h))]
        image = image_row
        cv2_frame = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = cv2_frame
        img = Image.fromarray(img)
		
        # Transform
        img = transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(gpu)
		
        yaw, pitch, roll = model(img)

        yaw_predicted = F.softmax(yaw)
        pitch_predicted = F.softmax(pitch)
        roll_predicted = F.softmax(roll)
	    # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

        # x, y = img_gray.shape
        # draw_axis(image_row, yaw_predicted, pitch_predicted, roll_predicted, tdx =  x/2, tdy = y/2, size = y/2)
        # cv2.imwrite('/data2/gaofuxun/data/RankIQA/iqiyi_pose/'+image_path, image_row)

	    # Print new image with cube and axis
        qua_score = image_path[0:-4]
        f_out.write(os.path.join(data_dir, image_path)+' '+str(yaw_predicted)+
                    ' '+str(pitch_predicted)+' '+str(roll_predicted)+' '+qua_score+'\n')
        print('finish {}/{}'.format(num, total_images))
        num += 1
    f_out.close()
