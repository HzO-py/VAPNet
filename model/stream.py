import torch
import torch.nn as nn
import torchvision
from model.attention import ChannelAttention,SpatialAttention

class Stream(nn.Module):
    def __init__(self,embedding_size=128,is_attention=True):
        super(Stream, self).__init__()
        resnet=torchvision.models.resnet34(pretrained=True)
        self.is_attention=is_attention
        self.resnet=nn.Sequential(*list(resnet.children())[:-2])
        self.ca=ChannelAttention(resnet.fc.in_features)
        self.nor1=nn.BatchNorm2d(resnet.fc.in_features)
        self.sa=SpatialAttention()
        self.nor2=nn.BatchNorm2d(resnet.fc.in_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(resnet.fc.in_features, embedding_size)

    
    def forward(self,x):
        x=self.resnet(x)
        if self.is_attention:
            ca_att=self.ca(x)
            x=ca_att*x
            x=self.nor1(x)
            sa_att=self.sa(x)
            x=sa_att*x
            x=self.nor2(x)
        x=self.avgpool(x)
        x=torch.flatten(x,start_dim=1)
        x=self.fc(x)
        return x

# x=torch.FloatTensor(100,3,128,128)
# model=Stream()
# print(model)
# print(x.shape,model(x).shape)