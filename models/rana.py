from __future__ import print_function
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import models

class NetRana(nn.Module):
    def __init__(self,num_classes):
        super(NetRana, self).__init__()
        
        resnet1 = models.resnet50(pretrained=True)
        modules1 = list(resnet1.children())[:-1]      # delete the last fc layer.
        self.resnet1 = nn.Sequential(*modules1)

        resnet2 = models.resnet50(pretrained=True)
        modules2 = list(resnet2.children())[:-1] 
        self.resnet2 = nn.Sequential(*modules2)
        
        self.fc1 = nn.Linear(2048*2, 2048)
        self.fc2 = nn.Linear(2048,num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x1,x2):
        
        x1 = self.resnet1(x1)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.resnet2(x2)
        x2 = x2.view(x2.size(0), -1)
        
        x = torch.cat((x1,x2),dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
       
        output = self.softmax(x)
        
        return output
    