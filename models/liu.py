from __future__ import print_function
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import models


random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)  # Only for single GPU


class Res2Net(nn.Module):

    def __init__(self,num_classes):
       
        super(Res2Net, self).__init__()
        inplanes = 3
        planes = 3
        
        width = 16
        self.width_1 = 16
        scale = 4
        stride = 1
        
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        self.relu1 = nn.ReLU(inplace=True)
       # self.relu = nn.ReLU(inplace=True)
        
        self.nums = scale
        
        self.conv11 = nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(width)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False)
        self.bn12 = nn.BatchNorm2d(width)
        self.relu12 = nn.ReLU(inplace=True)
        
        
        self.conv21 = nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False)
        self.bn21 = nn.BatchNorm2d(width)
        self.relu21 = nn.ReLU(inplace=True)
        self.conv22 = nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False)
        self.bn22 = nn.BatchNorm2d(width)
        self.relu22 = nn.ReLU(inplace=True)
        
        self.conv31 = nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False)
        self.bn31 = nn.BatchNorm2d(width)
        self.relu31 = nn.ReLU(inplace=True)
        self.conv32 = nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False)
        self.bn32 = nn.BatchNorm2d(width)
        self.relu32 = nn.ReLU(inplace=True)
        
        self.conv41 = nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False)
        self.bn41 = nn.BatchNorm2d(width)
        self.relu41 = nn.ReLU(inplace=True)
        self.conv42 = nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False)
        self.bn42 = nn.BatchNorm2d(width)
        self.relu42 = nn.ReLU(inplace=True)
        

        self.conv3 = nn.Conv2d(width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        
        vgg1 =  models.vgg16(pretrained=False)
        modules1 = list(vgg1.children())[:-2]      # delete the last fc layer.
        self.vgg = nn.Sequential(*modules1)
        
        self.globpool = nn.AvgPool2d(2)
        
        self.fc1 = nn.Linear(512, num_classes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        residual = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        spx = torch.split(x, self.width_1, 1)
        
        x1 = self.conv11(spx[0])
        x1 = self.bn11(x1)
        x1 = self.relu11(x1)
        x1 = self.conv12(x1)
        x1 = self.bn12(x1)
        x1 = self.relu12(x1)
        
        x2 = x1 + spx[1]
        x2 = self.conv21(x2)
        x2 = self.bn21(x2)
        x2 = self.relu21(x2)
        x2 = self.conv22(x2)
        x2 = self.bn22(x2)
        x2 = self.relu22(x2)
        
        x3 = x2 + spx[2]
        x3 = self.conv31(x3)
        x3 = self.bn31(x3)
        x3 = self.relu31(x3)
        x3 = self.conv32(x3)
        x3 = self.bn32(x3)
        x3 = self.relu32(x3)
        
        x4 = x3 + spx[3]
        x4 = self.conv41(x4)
        x4 = self.bn41(x4)
        x4 = self.relu41(x4)
        x4 = self.conv22(x4)
        x4 = self.bn42(x4)
        x4 = self.relu42(x4)
        
        
              
        out = torch.cat((x1,x2,x3,x4), 1)
                     
        
        x = self.conv3(out)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = x - residual
        
        x = self.vgg(x)
        
        x = self.globpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        
        x = self.softmax(x)    

        return x
