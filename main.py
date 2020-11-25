import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import os
import cv2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 50
BATCH_SIZE = 16
LR = 0.0001

# 读取 图像-种类 对应表，并将图像名称、种类分别提取为list
img2label = pd.read_csv('train.csv')
img2label_imgs = list(img2label['image_id'])
img2label_labels = list(img2label['label'])

filePath = 'C:/Users/lab/Desktop/taskey/kaggle/Cassava_data/train_images'

################################################################
# 首次运行时使用，用于将图片进行不同文件夹分类
# # 创建五个新文件夹，存放不同种类的图像 
# for i in range(5):
#     os.makedirs('C:/Users/lab/Desktop/taskey/kaggle/Cassava_data' + '/sample/' + str(i), exist_ok = True)

# for name in img2label_imgs:
#     img = cv2.imread(filePath + '/' + name)
#     label = img2label_labels[img2label_imgs.index(name)]
#     cv2.imwrite('C:/Users/lab/Desktop/taskey/kaggle/Cassava_data' + 'sample/' + str(label) + '/' + name, img)
################################################################
    
transforms_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor()])

full_dataset = torchvision.datasets.ImageFolder(root = 'sample', transform = transforms_train)

train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
data_train, data_valid = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

train_data = torch.utils.data.DataLoader(dataset = data_train, batch_size = BATCH_SIZE, shuffle = True)
valid_data = torch.utils.data.DataLoader(dataset = data_valid, batch_size = BATCH_SIZE, shuffle = True)
