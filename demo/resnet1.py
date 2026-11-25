import torch
from torch import nn

class ResidualBlock(nn.Module):
    #定义残差块的初始化函数
    def __init__(self, channels):
        super().__init__()
        #定义残差块的卷积层、批归一化层和激活函数
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        #定义第二个卷积层和批归一化层
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    #定义前向传播函数
    def forward(self, x):
        #残差连接
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #y=F(x)+x
        out = out + identity
        out = self.relu(out)

        return out
