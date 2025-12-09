#coding:utf8
import  torch
import torch.nn as nn
import numpy as np

# 使用numpy手动实现模拟一个线性层
# 搭建一个2层的神经网络模型
# 每层都是线性层


class TorchModel(nn.Module):
    def __init__(self,input_size,hidden_size,hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size) #w：3×5矩阵
        self.sig=nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_size,hidden_size2) #5×2
        self.relu=nn.ReLU()
    def forward(self,x):
        x = self.layer1(x)
        x = self.sig(x)
        y_pred=self.layer2(x)
        y_pred=self.relu(y_pred)
        return y_pred
#自定义模型
