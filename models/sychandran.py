from __future__ import print_function
from __future__ import print_function
from __future__ import division
import torch.nn as nn


class Sy_Net(nn.Module):
    def __init__(self,num_classes):
        super(Sy_Net, self).__init__()
        
        self.conv11 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride=1,padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.relu11 = nn.ReLU()
        
        self.conv12 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),stride=2,padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.relu12 = nn.ReLU()
        
        self.rconv11 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),stride=1,padding=1)
        self.rbn11 = nn.BatchNorm2d(16)
        self.rrelu11 = nn.ReLU()
        
        self.rconv12 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),stride=1,padding=1)
        self.rbn12 = nn.BatchNorm2d(16)
        self.rrelu12 = nn.ReLU()
        
 #--------------------------------------------------------------------------------------------------       
        
        self.conv21 = nn.Conv2d(in_channels=16,out_channels=8,kernel_size=(3,3),stride=2,padding=1)
        self.bn21 = nn.BatchNorm2d(8)
        self.relu21 = nn.ReLU()

        self.conv22 = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=(3,3),stride=1,padding=1)
        self.bn22 = nn.BatchNorm2d(8)
        self.relu22 = nn.ReLU()

        self.rconv21 = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=(3,3),stride=1,padding=1)
        self.rbn21 = nn.BatchNorm2d(8)
        self.rrelu21 = nn.ReLU()

        self.rconv22 = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=(3,3),stride=1,padding=1)
        self.rbn22 = nn.BatchNorm2d(8)
        self.rrelu22 = nn.ReLU()
        
#--------------------------------------------------------------------------------------------------               
        
        self.conv31 = nn.Conv2d(in_channels=8,out_channels=4,kernel_size=(3,3),stride=2,padding=1)
        self.bn31 = nn.BatchNorm2d(4)
        self.relu31 = nn.ReLU()
        
        self.conv32 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=1,padding=1)
        self.bn32 = nn.BatchNorm2d(4)
        self.relu32 = nn.ReLU()
        
        self.rconv31 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=1,padding=1)
        self.rbn31 = nn.BatchNorm2d(4)
        self.rrelu31 = nn.ReLU()
        
        self.rconv32 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=1,padding=1)
        self.rbn32 = nn.BatchNorm2d(4)
        self.rrelu32 = nn.ReLU()
        
#--------------------------------------------------------------------------------------------------       
        
        self.conv41 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=2,padding=1)
        self.bn41 = nn.BatchNorm2d(4)
        self.relu41 = nn.ReLU()
        
        self.conv42 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=1,padding=1)
        self.bn42 = nn.BatchNorm2d(4)
        self.relu42 = nn.ReLU()
        
        self.rconv41 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=1,padding=1)
        self.rbn41 = nn.BatchNorm2d(4)
        self.rrelu41 = nn.ReLU()
        
        self.rconv42 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=1,padding=1)
        self.rbn42 = nn.BatchNorm2d(4)
        self.rrelu42 = nn.ReLU()
        
#--------------------------------------------------------------------------------------------------       
        
        self.conv51 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=2,padding=1)
        self.bn51 = nn.BatchNorm2d(4)
        self.relu51 = nn.ReLU()
        
        self.conv52 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=1,padding=1)
        self.bn52 = nn.BatchNorm2d(4)
        self.relu52 = nn.ReLU()
        
        self.rconv51 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=1,padding=1)
        self.rbn51 = nn.BatchNorm2d(4)
        self.rrelu51 = nn.ReLU()
        
        self.rconv52 = nn.Conv2d(in_channels=4,out_channels=4,kernel_size=(3,3),stride=1,padding=1)
        self.rbn52 = nn.BatchNorm2d(4)
        self.rrelu52 = nn.ReLU()
        
        
 #--------------------------------------------------------------------------------------------------       
        
        self.fc1 = nn.Linear(4,num_classes) 
    
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)
        
        x1 = x
        
        x = self.rconv11(x)
        x = self.rbn11(x)
        x = self.rrelu11(x)
        
        x = self.rconv12(x)
        x = self.rbn12(x)
        x = self.rrelu12(x)
        
        x = x1+x
        
#---------------------------------------------------------------------------        
        
        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu21(x)
        
        x = self.conv22(x)
        x = self.bn22(x)
        x = self.relu22(x)
        
        x1 = x
        
        x = self.rconv21(x)
        x = self.rbn21(x)
        x = self.rrelu21(x)
        
        x = self.rconv22(x)
        x = self.rbn22(x)
        x = self.rrelu22(x)
        
        x = x1+x
       
         
        
        x = self.conv31(x)
        x = self.bn31(x)
        x = self.relu31(x)
        
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu32(x)
        
        x2 = x
        
        x = self.rconv31(x)
        x = self.rbn31(x)
        x = self.rrelu31(x)
        
        x = self.rconv32(x)
        x = self.rbn32(x)
        x = self.rrelu32(x)
        
        x = x2+x
        
#---------------------------------------------------------------------------                
        
        x = self.conv41(x)
        x = self.bn41(x)
        x = self.relu41(x)
        
        x = self.conv42(x)
        x = self.bn42(x)
        x = self.relu42(x)
        
        x3 = x
        
        x = self.rconv41(x)
        x = self.rbn41(x)
        x = self.rrelu41(x)
        
        x = self.rconv42(x)
        x = self.rbn42(x)
        x = self.rrelu42(x)
        
        x = x3+x
        #print(x.shape)
#---------------------------------------------------------------------------                


        x = self.conv51(x)
        x = self.bn51(x)
        x = self.relu51(x)
        
        x = self.conv52(x)
        x = self.bn52(x)
        x = self.relu52(x)
        
        x4 = x
        
        x = self.rconv51(x)
        x = self.rbn51(x)
        x = self.rrelu51(x)
        
        x = self.rconv52(x)
        x = self.rbn52(x)
        x = self.rrelu52(x)
        
        x = x4+x
        
#---------------------------------------------------------------------------              
    
    
        #print(x.shape)
        x = x.reshape(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.softmax(x)
       
        return x