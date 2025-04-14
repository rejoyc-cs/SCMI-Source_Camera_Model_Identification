from __future__ import print_function
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=(dim),out_channels=(dim),kernel_size=(1,1))
        self.LN1 = LayerNorm(dim, eps=1e-6)
        
        self.conv2 = nn.Conv2d(in_channels=(dim),out_channels=(dim),kernel_size=(1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_input = x
        
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.LN1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
    
        return x * x_input

class DPEC(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super(DPEC, self).__init__()
        half_dim = int(dim/2)
        self.conv11 = nn.Conv2d(in_channels=dim,out_channels=half_dim,kernel_size=(1,1))
        self.gelu11 = nn.GELU()
        
        self.conv12 = nn.Conv2d(in_channels=half_dim,out_channels=half_dim,kernel_size=(7,7),groups= half_dim,padding=3)
        self.LN12 = LayerNorm(half_dim, eps=1e-6)
        
        self.conv13 = nn.Conv2d(in_channels=half_dim,out_channels=dim,kernel_size=(1,1))
        self.gelu13 = nn.GELU()
        
        self.conv14 = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=(1,1))
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.LN14 = LayerNorm(dim, eps=1e-6)
        
        #------------------------------------------------------------------------------------
        
        self.conv21 = nn.Conv2d(in_channels=dim,out_channels=half_dim,kernel_size=(1,1))
        self.gelu21 = nn.GELU()
        
        self.conv22 = nn.Conv2d(in_channels=half_dim,out_channels=half_dim,kernel_size=(7,7),groups= half_dim,padding=3)
        self.LN22 = LayerNorm(half_dim, eps=1e-6)
        
        self.conv23 = nn.Conv2d(in_channels=half_dim,out_channels=dim,kernel_size=(1,1))
        self.gelu23 = nn.GELU()
        
        self.conv24 = nn.Conv2d(in_channels=dim,out_channels=dim,kernel_size=(1,1))
        self.LN24 = LayerNorm(dim, eps=1e-6)
        
        self.channel_attention = ChannelAttention(dim)
        
        
        
    def forward(self, x):
        x1 = x
        x2 = x
        x1 = self.conv11(x1)
        x1 = self.gelu11(x1)
        x1 = self.conv12(x1)
        x1 = self.LN12(x1)
        x1 = self.conv13(x1)
        x1 = self.gelu13(x1)
        x1 = self.conv14(x1)
        
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.gamma * x1
        x1 = x1.permute(0, 3, 1, 2)
        
        x1 = self.LN14(x1)
        
        x2 = self.conv21(x2)
        x2 = self.gelu21(x2)
        x2 = self.conv22(x2)
        x2 = self.LN22(x2)
        x2 = self.conv23(x2)
        x2 = self.gelu23(x2)
        x2 = self.conv24(x2)
        
        x2 = x2.permute(0, 2, 3, 1)
        x2 = self.gamma * x2
        x2 = x2.permute(0, 3, 1, 2)
        
        
        x2 = self.LN24(x2)
        
        x3 = x1 + x2
        x3 =self.channel_attention(x3)
        x = x + x3
        
        return x

class ConvNeXt(nn.Module):
    def __init__(self,num_classes):
        super(ConvNeXt, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=128,kernel_size=(4,4),stride=4,padding=4)
        self.LN1 = LayerNorm(128, eps=1e-6)
        
        
        self.dpec1 = DPEC(dim=128)
        self.dpec2 = DPEC(dim=128)
        self.dpec3 = DPEC(dim=128)
        self.LN2 = LayerNorm(128, eps=1e-6)
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(2,2),stride=2,padding=1)
        
        self.dpec4 = DPEC(dim=256)
        self.dpec5 = DPEC(dim=256)
        self.dpec6 = DPEC(dim=256)
        self.LN3 = LayerNorm(256, eps=1e-6)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(2,2),stride=2,padding=1)
        
        self.dpec7 = DPEC(dim=512)
        self.dpec8 = DPEC(dim=512)
        self.dpec9 = DPEC(dim=512)
        self.LN4 = LayerNorm(512, eps=1e-6)
        self.conv4 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(2,2),stride=2,padding=1)
        
        self.dpec10 = DPEC(dim=1024)
        self.dpec11 = DPEC(dim=1024)
        self.dpec12 = DPEC(dim=1024)
      
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.LN5 = LayerNorm(1024, eps=1e-6)
        
        self.fc1 = nn.Linear(1024,num_classes) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.LN1(x)
        
        x = self.dpec1(x)
    
        x = self.dpec2(x)
        x = self.dpec3(x)
        x = self.LN2(x)
        
        x = self.conv2(x)
        
        x = self.dpec4(x)
        x = self.dpec5(x)
        x = self.dpec6(x)
        x = self.LN3(x)
        x = self.conv3(x)
        
        x = self.dpec7(x)
        x = self.dpec8(x)
        x = self.dpec9(x)
        x = self.LN4(x)
        x = self.conv4(x)
        
        x = self.dpec10(x)
        x = self.dpec11(x)
        x = self.dpec12(x)
        
        
        x = self.globalpool(x)
        x = self.LN5(x)
        x = x.reshape(x.size(0), -1)
       # print(x.shape)
        x = self.fc1(x)
        x = self.softmax(x)
        
        return x