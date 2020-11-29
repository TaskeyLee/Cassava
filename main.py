import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import os
import cv2
import model
import freeze

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCH = 50
BATCH_SIZE = 64
LR = 0.0001

# 读取 图像-种类 对应表，并将图像名称、种类分别提取为list
img2label = pd.read_csv('train.csv')
img2label_imgs = list(img2label['image_id'])
img2label_labels = list(img2label['label'])

filePath = 'C:/Users/lab/Desktop/taskey/kaggle/Cassava_data/train_images'

################################################################
def img_sort(img2label_labels):
    # 创建五个新文件夹，存放不同种类的图像 
    for i in range(5):
        os.makedirs('C:/Users/lab/Desktop/taskey/kaggle/Cassava_data' + '/sample/' + str(i), exist_ok = True)
    
    for name in img2label_imgs:
        img = cv2.imread(filePath + '/' + name)
        label = img2label_labels[img2label_imgs.index(name)]
        cv2.imwrite('C:/Users/lab/Desktop/taskey/kaggle/Cassava_data' + 'sample/' + str(label) + '/' + name, img)
# img_sort(img2label_labels) # 首次运行时使用，用于将图片进行不同文件夹分类
################################################################

def data_preprocess():
    # 数据增强：随机翻转、规范化、转为tensor
    transforms_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            torchvision.transforms.Resize([160, 120]),
            # transforms.RandomResizedCrop([160, 120]),
            # transforms.Normalize(mean=0.437, std=0.244),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
    # 从img_sort函数生成的文件夹中导入数据集
    full_dataset = torchvision.datasets.ImageFolder(root = 'C:/Users/lab/Desktop/taskey/kaggle/Cassava_data/sample', transform = transforms_train)
    # 将数据集转为cpu上的array，求其均值、方差，用于后续预处理
    # data = []
    # count = 1
    # for d in full_dataset:
    #     data.append(d[0].data.cpu().numpy())
    #     print('Count: {}'.format(count))
    #     count += 1
    # # 计算均值、方差
    # mean, std = np.mean(data), np.std(data)
    # 按照0.8的比例将数据集分为train和valid
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    data_train, data_valid = torch.utils.data.random_split(full_dataset, [train_size, valid_size])
    # 转为torch使用的dataloader
    train_data = torch.utils.data.DataLoader(dataset = data_train, batch_size = BATCH_SIZE, shuffle = True)
    valid_data = torch.utils.data.DataLoader(dataset = data_valid, batch_size = BATCH_SIZE, shuffle = True)
    return train_data, valid_data

if __name__ == '__main__':
    writer1 = SummaryWriter('runs/exp')
    
    train_data, valid_data = data_preprocess()
    
    my_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet101', pretrained=True)  
    
    num_fc_in = my_model.fc.in_features
    
    model.weights_init(my_model)
    # 改变全连接层，5分类问题，out_features = 5
    my_model.fc = nn.Linear(num_fc_in, 5)
    
    # 冻结部分layer
    # freeze.set_freeze_by_names(my_model, 'conv1')
    # freeze.set_freeze_by_names(my_model, 'layer1')
    # freeze.set_freeze_by_names(my_model, 'layer2')
    # freeze.set_freeze_by_names(my_model, 'layer3')
    # freeze.set_freeze_by_names(my_model, 'layer4')
    
    # 模型迁移到CPU/GPU
    my_model = my_model.to(DEVICE)
    
    # 选择优化方法
    optimizer = optim.Adam(my_model.parameters(), lr = LR)
    
    for epoch in range(EPOCH):
        model.transfer_train(my_model, train_data, DEVICE, epoch, optimizer, writer1)
        model.test_model(my_model, valid_data, DEVICE, epoch, writer1)
