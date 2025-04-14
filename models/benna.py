from __future__ import print_function
from __future__ import print_function
from __future__ import division
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,num_classes):
        super(Net, self).__init__()
        
        self.conv11 = nn.Conv2d(in_channels=3,out_channels=96,kernel_size=(7,7),stride=2,padding=3)
        self.bn11 = nn.BatchNorm2d(96)
        self.relu11 = nn.ReLU()
        
        self.maxpool11= nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))
        
        self.conv12 = nn.Conv2d(in_channels=96,out_channels=64,kernel_size=(5,5),stride=1,padding=2)
        self.bn12 = nn.BatchNorm2d(64)
        self.relu12 = nn.ReLU()
        
        self.maxpool12= nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))
        
        self.conv13 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(5,5),stride=1,padding=2)
        self.bn13 = nn.BatchNorm2d(64)
        self.relu13 = nn.ReLU()
        
        self.maxpool13= nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))
        
        
        self.conv14 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(1,1),stride=1,padding=0)
        self.bn14 = nn.BatchNorm2d(128)
        self.relu14 = nn.ReLU()
        
        self.maxpool14= nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))
        
        self.fc1 = nn.Linear(2048,1024) 
        self.tanh1 = nn.Tanh()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024,200) 
        self.tanh2 = nn.Tanh()
        self.drop2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(200,num_classes) 
    
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        
        x = self.maxpool11(x)
        
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        
        x = self.maxpool12(x)
        
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)
        
        x = self.maxpool13(x)
       
        
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu14(x)
        
        x = self.maxpool14(x)
        
        #print(x.shape)
        x = x.reshape(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.drop2(x)
        
        x = self.fc3(x)

        x = self.softmax(x)
       
        return x
