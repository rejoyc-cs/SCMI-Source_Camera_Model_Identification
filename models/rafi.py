from __future__ import print_function
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn


class RemNet(nn.Module):
    def __init__(self,num_classes):
        super(RemNet, self).__init__()
        
        self.rem1_conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        self.rem1_bn1 = nn.BatchNorm2d(64)
        self.rem1_conv2 = nn.Conv2d(in_channels=67,out_channels=64,kernel_size=(3,3),stride=1,padding=1)
        self.rem1_bn2 = nn.BatchNorm2d(64)
        self.rem1_conv3 = nn.Conv2d(in_channels=67,out_channels=3,kernel_size=(3,3),stride=1,padding=1)
        self.rem1_bn3 = nn.BatchNorm2d(3)
        
        
        self.rem2_conv1 = nn.Conv2d(in_channels=3,out_channels=128,kernel_size=(3,3),stride=1,padding=1)
        self.rem2_bn1 = nn.BatchNorm2d(128)
        self.rem2_conv2 = nn.Conv2d(in_channels=131,out_channels=128,kernel_size=(3,3),stride=1,padding=1)
        self.rem2_bn2 = nn.BatchNorm2d(128)
        self.rem2_conv3 = nn.Conv2d(in_channels=131,out_channels=3,kernel_size=(3,3),stride=1,padding=1)
        self.rem2_bn3 = nn.BatchNorm2d(3)
        
        self.rem3_conv1 = nn.Conv2d(in_channels=3,out_channels=256,kernel_size=(3,3),stride=1,padding=1)
        self.rem3_bn1 = nn.BatchNorm2d(256)
        self.rem3_conv2 = nn.Conv2d(in_channels=259,out_channels=256,kernel_size=(3,3),stride=1,padding=1)
        self.rem3_bn2 = nn.BatchNorm2d(256)
        self.rem3_conv3 = nn.Conv2d(in_channels=259,out_channels=3,kernel_size=(3,3),stride=1,padding=1)
        self.rem3_bn3 = nn.BatchNorm2d(3)
        
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(7,7),stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU()
        
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(5,5),stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        self.prelu3 = nn.PReLU()
        
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(2,2),stride=2,padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.prelu4 = nn.PReLU()
        
   
        self.avgpool1= nn.AvgPool2d(kernel_size=(4,4),stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=512,out_channels=num_classes,kernel_size=(1,1),stride=1)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1):
        
        x = x1
        x = self.rem1_conv1(x)
        x = self.rem1_bn1(x)
        x = torch.cat((x,x1),dim=1)
        x = self.rem1_conv2(x)
        x = self.rem1_bn2(x)
        x = torch.cat((x,x1),dim=1)
        x = self.rem1_conv3(x)
        x = self.rem1_bn3(x)
        
        x = x1-x
        x1 = x
        
        
    
        x = self.rem2_conv1(x)
        x = self.rem2_bn1(x)
        x = torch.cat((x,x1),dim=1)
        x = self.rem2_conv2(x)
        x = self.rem2_bn2(x)
        x = torch.cat((x,x1),dim=1)
        x = self.rem2_conv3(x)
        x = self.rem2_bn3(x)
        
        x = x1-x
        x1 = x
        
        
        
        x = self.rem3_conv1(x)
        x = self.rem3_bn1(x)
        x = torch.cat((x,x1),dim=1)
        x = self.rem3_conv2(x)
        x = self.rem3_bn2(x)
        x = torch.cat((x,x1),dim=1)
        x = self.rem3_conv3(x)
        x = self.rem3_bn3(x)
        
        x = x1 - x
        
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        
       # print(x.shape)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        
       # print(x.shape)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        
       # print(x.shape)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.prelu4(x)
        #print(x.shape)
        
        x = self.avgpool1(x)
        
       # print(x.shape)
        x = self.conv5(x)
        
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        
        x = self.softmax(x)
       
        return x