import pandas as pd
import json
import numpy as np
from bs4 import BeautifulSoup


import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches 
from PIL import Image 
import albumentations as A
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor
import torchvision
from torchvision import tv_tensors

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torchvision import models

from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

# from engine import train_one_epoch, evaluate
# import utils
# from utils import collate_fn
import warnings
warnings.filterwarnings('ignore')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4)  # 32*(4,4) filter ==> 221*221*32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # pool (2,2) ==> 110*110*32
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, 3)  # 64*(3,3) filter ==> 108*108*64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # pool (2,2) ==> 54*54*64
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 2)  # 128*(2,2) filter ==> 53*53*128
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # pool (2,2) ==> 26*26*128
        self.dropout3 = nn.Dropout(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 1)  # 256*(1,1) filter ==> 26*26*256
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # pool (2,2) ==> 13*13*256
        self.dropout4 = nn.Dropout(p=0.4)

        self.fc1 = nn.Linear(82944, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.dropout5 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(1000, 500)
        self.bn6 = nn.BatchNorm1d(500)
        self.dropout6 = nn.Dropout(p=0.6)

        self.fc3 = nn.Linear(500, 8)

        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)

    def forward(self, x):
        # Defining the feedforward behavior of this model
        # x is the input image and, as an example, here we may choose to include a pool/conv step:

        x = self.dropout1(self.pool1(F.elu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.elu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.elu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.elu(self.bn4(self.conv4(x)))))
        # flatten
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.bn5(x)
        x = self.dropout5(F.elu(x))
        x = self.dropout6(F.elu(self.bn6(self.fc2(x))))
        x = self.fc3(x)

        return x
    


class KeyPointDetectionDS(Dataset):

    def __init__(self, path2json,file_base_path, type = "train",transform=True):
        self.path2json = path2json
        self.file_base_path = file_base_path
        self.dataset_type = type
        self.transform = transform
        #print(path2json)
        with open(path2json, 'r') as file:
            self.data = json.load(file)

        

    def __len__(self):
        return len(self.data)
    

    def show_load(self, image_path, points,image_i_dict):
        img = cv2.imread(image_path) 
        fig, ax = plt.subplots(1) 
        ax.imshow(img) 
        rect3 = patches.Rectangle((float(image_i_dict["xtl"]), #X top left
                              float(image_i_dict["ytl"])), # Y top_left
                             float(image_i_dict["xwidth"]), #Width
                             float(image_i_dict["yheight"]), #hight
                             linewidth=1, edgecolor='g', facecolor="none") 
        # Add the patch to the Axes 
        ax.add_patch(rect3) 
        for p in points:
            plt.plot(p[0],p[1],'ro') 

        plt.show() 
        return None
    
    def __getitem__(self, idx, show = False,transform_show=False):
        image_i_dict = self.data[str(idx)]
        image_path = self.file_base_path + r'/'+ image_i_dict["image_path"]



        # Process Image
        img = cv2.imread(image_path)
        bbox =  tv_tensors.BoundingBoxes(torch.tensor([[float(image_i_dict["xtl"]),float(image_i_dict["ytl"]),float(image_i_dict["xbr"]),float(image_i_dict["ybr"])]]),
                                         format="XYXY",
                                         canvas_size = (float(image_i_dict["width"]), float(image_i_dict["height"])))
        key_points = torch.tensor([float(i) for i in image_i_dict["points_points"].replace(";", ",").split(",")], dtype=torch.int).reshape((4,2))
        #print(key_points)


        if show:
            self.show_load(image_path,key_points, image_i_dict)

        if self.transform:
            transform = A.Compose([A.Resize(width=500, height=500),
                                    A.VerticalFlip(p=0.4),
                                   #A.Rotate(p=0.5),
                                   #A.RandomCrop(width=1500, height=1000),
                                   A.RandomBrightnessContrast(p=0.2),],
                                  keypoint_params=A.KeypointParams(format="xy"),)
            transformed = transform(image=img, keypoints=key_points)
            img = transformed["image"]
            key_points = torch.tensor(transformed["keypoints"])
            #print(transformed["keypoints"])
            if transform_show:
                plt.imshow(img) 
                for p in transformed["keypoints"]:
                    plt.plot(p[0],p[1],'ro') 

                plt.show() 
            #print(transformed["keypoints"])

        img_original = torchvision.transforms.functional.to_tensor(img)

        visibility = torch.ones((key_points.size()[0],1), dtype=torch.int)

        item_dict = {}
        #item_dict["input"] = img_original
        item_dict["boxes"] = bbox
        item_dict["keypoints"] = key_points
        #print(f'Shape: Keypoints: {key_points.shape} image_id  {str(idx)}')
        item_dict["labels"] = torch.ones(1, dtype=int)
        #item_dict["image_id"] = torch.tensor([idx])
        item_dict["image_id"] = str(idx)
        item_dict["area"] =torch.absolute(bbox[:, 3] - bbox[:, 1]) * torch.absolute(bbox[:, 2] - bbox[:, 0])
        item_dict["iscrowd"] =torch.zeros(len(bbox), dtype=torch.int64)
        return img_original, item_dict#, img, transformed["keypoints"]