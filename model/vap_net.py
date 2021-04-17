import torch
import torch.nn as nn
from model.attention import CoAttention
from model.stream import Stream
from model.st_gcn import SSTGCN

class VANET_SUB(nn.Module):
    def __init__(self,class_num=4,in_planes=128,video_embed=128,audio_embed=128,is_attention=True,is_coa=True):
        super(VANET_SUB, self).__init__()
        self.video=Stream(video_embed,is_attention=is_attention)
        self.audio=Stream(audio_embed,is_attention=is_attention)
        self.coa=CoAttention(in_planes)
        self.is_coa=is_coa
        #self.fc=nn.Linear(in_planes*2,class_num)
        #self.softmax=nn.Softmax(dim=1)

    def forward(self,x,y):
        v=self.video(x)
        a=self.audio(y)
        if self.is_coa:
            v,a=self.coa(v,a)
            v=v.squeeze()
            a=a.squeeze()
        return v,a
        # inp=torch.cat([v,a],1)
        # out=self.fc(inp)
        # out=self.softmax(out)
        # return out

class VANET(nn.Module):
    def __init__(self,class_num=4,in_planes=128,is_attention=True,is_coa=True):
        super(VANET, self).__init__()
        self.vanet=VANET_SUB(is_attention=is_attention,is_coa=is_coa)
        self.bn=nn.BatchNorm1d(in_planes*2)
        self.fc=nn.Linear(in_planes*2,class_num)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x,y):
        v,a=self.vanet(x,y)
        inp=torch.cat([v,a],1)
        inp=self.bn(inp)
        out=self.fc(inp)
        out=self.softmax(out)
        return out

class VAPNET(nn.Module):
    def __init__(self,class_num=4,in_planes=128,is_attention=True,is_coa=True):
        super(VAPNET, self).__init__()
        self.vanet=VANET_SUB(is_attention=is_attention,is_coa=is_coa)
        self.pnet=SSTGCN()
        self.bn=nn.BatchNorm1d(in_planes*3)
        self.fc=nn.Linear(in_planes*3,class_num)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x,y,z):
        v,a=self.vanet(x,y)
        p=self.pnet(z)
        inp=torch.cat([v,a,p],1)
        inp=self.bn(inp)
        out=self.fc(inp)
        out=self.softmax(out)
        return out

class PNET(nn.Module):
    def __init__(self,class_num=4,in_planes=128):
        super(PNET, self).__init__()
        self.pnet=SSTGCN()
        self.bn=nn.BatchNorm1d(in_planes)
        self.fc=nn.Linear(in_planes,class_num)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,z):
        p=self.pnet(z)
        inp=self.bn(p)
        out=self.fc(inp)
        out=self.softmax(out)
        return out

# x=torch.FloatTensor(100,3,128,128)
# y=torch.FloatTensor(100,3,128,128)
# model=VANET(3)
# #print(model)
# x,y=model(x,y)
# print(x,y)
# print(x.shape,y.shape)