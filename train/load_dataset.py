from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
import pandas as pd
import xlrd
import numpy as np
import random
from PIL import Image

class VAPDataset(Dataset):

    def __init__(self,t_duration=9,t_skip=3,transform=None):
        f = open("/hdd/sdd/lzq/dianwang/dataset/items.txt","r")
        self.items = eval(f.read())
        self.items=self.items[:len(self.items)//10*9]
        f.close()
        self.transform = transform
        self.t_duration=t_duration
        self.t_skip=t_skip

    def __len__(self):
        return len(self.items)

    def openimg(self,img_path):
        img=np.array(Image.open(img_path).convert("RGB"))
        img=img.astype(np.float32)
        img/=225.0
        img_torch=torch.Tensor(np.transpose(img,(2,0,1)))
        return img_torch

    def onehot(self,label,class_num):
        label_long=torch.tensor([label])
        y=torch.zeros(class_num)
        y.scatter_(0,label_long,1)
        return y

    def openpose(self,pose_path):
        poses=sorted(os.listdir(pose_path+'/normal_pose'))
        rd=random.randint(0,len(poses)-self.t_duration*self.t_skip-1)
        ps=[]
        for i in range(rd,rd+self.t_duration*self.t_skip,self.t_skip):
            with open(pose_path+'/normal_pose/'+poses[i]) as f:
                p=eval(f.read())
                ps.append(p)
        ps=torch.Tensor(ps)
        ps=ps.permute(2,0,1).contiguous()
        return ps
        
            
    def __getitem__(self, idr):
        item=self.items[idr]
        rd=random.randint(0,len(item)-2)       
        sample = {'video': self.openimg(item[rd][0]), 'audio': self.openimg(item[rd][1]),'pose':self.openpose(item[rd][2]),'label':int(item[-1])}
        #print(idr, sample['video'], sample['audio'],sample['label'])
        if self.transform:
            sample = self.transform(sample)

        return sample

class VAPTestDataset(Dataset):

    def __init__(self,t_duration=9,t_skip=3,transform=None):
        f = open("/hdd/sdd/lzq/dianwang/dataset/items.txt","r")
        self.items = eval(f.read())
        self.items=self.items[len(self.items)//10*9:]
        f.close()
        self.transform = transform
        self.t_duration=t_duration
        self.t_skip=t_skip

    def __len__(self):
        return len(self.items)

    def openimg(self,img_path):
        img=np.array(Image.open(img_path).convert("RGB"))
        img=img.astype(np.float32)
        img/=255.0
        img_torch=torch.Tensor(np.transpose(img,(2,0,1)))
        #print(img)
        return img_torch

    def onehot(self,label,class_num):
        label_long=torch.tensor([label])
        y=torch.zeros(class_num)
        y.scatter_(0,label_long,1)
        return y

    def openpose(self,pose_path):
        poses=sorted(os.listdir(pose_path+'/normal_pose'))
        rd=random.randint(0,len(poses)-self.t_duration*self.t_skip-1)
        ps=[]
        for i in range(rd,rd+self.t_duration*self.t_skip,self.t_skip):
            with open(pose_path+'/normal_pose/'+poses[i]) as f:
                p=eval(f.read())
                ps.append(p)
        ps=torch.Tensor(ps)
        ps=ps.permute(2,0,1).contiguous()
        return ps
        
            
    def __getitem__(self, idr):
        item=self.items[idr]
        rd=random.randint(0,len(item)-2)       
        sample = {'video': self.openimg(item[rd][0]), 'audio': self.openimg(item[rd][1]),'pose':self.openpose(item[rd][2]),'label':int(item[-1])}
        #print(idr, sample['video'], sample['audio'],sample['label'])
        if self.transform:
            sample = self.transform(sample)

        return sample

# vadataset=VADataset('/hdd/sdd/lzq/dianwang/dataset/labels.xlsx')
# for i in range(len(vadataset)):
#     sample = vadataset[i]
#     print(i, sample['video'].shape, sample['audio'].shape,sample['label'])
# labels=pd.read_excel('/hdd/sdd/lzq/dianwang/dataset/labels.xlsx',sheet_name='Sheet1')
# print(labels.iloc[2])