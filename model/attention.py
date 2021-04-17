import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_planes=256, ratio=8,dropout=0.7):
        super(MLP, self).__init__()
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.dropout=nn.Dropout(dropout)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self,x):
        while len(x.shape)<4:
            x=torch.unsqueeze(x,dim=-1)
        x=self.fc1(x)
        x=self.relu1(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8,dropout=0.7):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp=MLP(in_planes,ratio,dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CoLayer(nn.Module):
    def __init__(self, dropout=0.7):
        super(CoLayer, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.softmax=nn.Softmax(dim=-1)
    
    def forward(self,x,y):
        yT=y.transpose(-1, -2).contiguous()
        score=self.softmax(torch.matmul(x,yT))
        score=self.dropout(score)
        att_map=torch.matmul(score,y)
        return x+att_map

class CoAttention(nn.Module):
    def __init__(self,in_planes=128):
        super(CoAttention, self).__init__()

        self.mlp_x1=MLP(in_planes)
        self.mlp_y1=MLP(in_planes)
        self.colayer_x1=CoLayer()
        self.colayer_y1=CoLayer()
        self.bn_x1=nn.BatchNorm2d(in_planes)
        self.bn_y1=nn.BatchNorm2d(in_planes)

        self.mlp_x2=MLP(in_planes)
        self.mlp_y2=MLP(in_planes)
        self.colayer_x2=CoLayer()
        self.colayer_y2=CoLayer()
        self.bn_x2=nn.BatchNorm2d(in_planes)
        self.bn_y2=nn.BatchNorm2d(in_planes)
    
    def forward(self,x,y):
        
        x=self.mlp_x1(x)
        y=self.mlp_y1(y)
        x_out=self.colayer_x1(x,y)
        y=self.colayer_y1(y,x)
        x=x_out
        x=self.bn_x1(x)
        y=self.bn_y1(y)

        
        x=self.mlp_x2(x)
        y=self.mlp_y2(y)
        x_out=self.colayer_x2(x,y)
        y=self.colayer_y2(y,x)
        x=x_out
        x=self.bn_x2(x)
        y=self.bn_y2(y)
        
        return x,y

        