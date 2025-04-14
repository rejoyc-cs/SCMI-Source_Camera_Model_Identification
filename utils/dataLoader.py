from __future__ import print_function
from __future__ import print_function
from __future__ import division
import torch
import torch.utils.data as data

from PIL import Image
from pathlib import Path
import os.path
import os
import scipy.io
import numpy as np
import random
from torchvision import  transforms
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)  # Only for single GPU
random.seed(random_seed)

# Ensuring deterministic behavior in convolutional operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMG_EXTENSIONS = [
   '.jpg', '.JPG', '.jpeg', '.JPEG',
   '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.mat',
]

def is_image_file(filename):
   return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
   classes = os.listdir(dir)
   classes.sort()
   class_to_idx = {classes[i]: i for i in range(len(classes))}
   return classes, class_to_idx

def make_dataset(dir, class_to_idx):
   images = []
   for target in os.listdir(dir):
       d = os.path.join(dir, target)
       if not os.path.isdir(d):
           continue

       for filename in os.listdir(d):
           if is_image_file(filename):
               path = '{0}/{1}'.format(target, filename)
               #print(path)
               item = (path, class_to_idx[target])
               images.append(item)

   return images

def default_loader(path):
   return Image.open(path).convert('RGB')

def mat_loader(path):
   return scipy.io.loadmat(path)

class ImageFolderLoader(data.Dataset):
   def __init__(self, data_path,transform=None,target_transform=None,loader=default_loader,filename_return=False):
       classes, class_to_idx = find_classes(data_path)
       imgs = make_dataset(data_path, class_to_idx)
       self.data_path = data_path
       self.imgs = imgs
       self.classes = classes
       self.class_to_idx = class_to_idx
       self.target_transform = target_transform
       self.loader = loader
       self.img_transform = transform
       self.filename_return = filename_return

   def __getitem__(self, index):
       path, target = self.imgs[index]
       filename = Path(path).stem 
       img = self.loader(os.path.join(self.data_path, path))  
       
       if self.img_transform is not None:
           img = self.img_transform(img)
        
       if self.target_transform is not None:
           target = self.target_transform(target)
        
       target = torch.eye(len(self.class_to_idx))[target]       
    
       if self.filename_return:
           return img,target,filename
       else:
           return img,target
       

   def __len__(self):
       return len(self.imgs)

class ImageFolderLoaderDual(data.Dataset):
    def __init__(self, root1, transform=None, target_transform=None, loader=default_loader, filename_return=False):
        classes1, class_to_idx1 = find_classes(root1)
        imgs1 = make_dataset(root1, class_to_idx1)
        
        self.root1 = root1
        self.imgs = imgs1
        self.classes1 = classes1
        self.class_to_idx1 = class_to_idx1
        self.transform_1 = transform
        self.target_transform = target_transform
        self.loader = loader
        
        self.img_noise = None
        self.img_rgb = None
        self.filename_return = filename_return
    
    def SRM(self):
        imgs = self.img_rgb
        
        filter2 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
        
        filter1 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
        
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, -2, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]]

        filter1 = np.asarray(filter1, dtype=float) / 12
        filter2 = np.asarray(filter2, dtype=float) / 4
        filter3 = np.asarray(filter3, dtype=float) / 2

        filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3]]
        filters = torch.FloatTensor(filters)
        
        imgs = np.array(imgs, dtype=float)
        w, h, c = imgs.shape
        imgs = imgs.reshape(1, w, h, c)
        imgs = np.einsum('klij->kjli', imgs)
        
        input_tensor = torch.tensor(imgs, dtype=torch.float32)
        op1 = F.conv2d(input_tensor, filters, stride=1, padding=2)
        
        op1 = op1.reshape(c, w, h)
        self.img_noise = op1
    
    def __getitem__(self, index):
        path, target = self.imgs[index]    
        img = self.loader(os.path.join(self.root1, path))  
        filename = Path(path).stem
        
        self.img_rgb = img
        self.SRM()
        
        if self.transform_1 is not None:
            img = self.transform_1(self.img_rgb)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        target = torch.eye(len(self.class_to_idx1))[target]
        
        if self.filename_return:
            return img, self.img_noise, target, filename
        
        return img, self.img_noise, target
    
    def __len__(self):
        return len(self.imgs)

   
def transform_data(method):
    data_transforms = None

    if method=='bennabhaktula':
        data_transforms = transforms.Compose([
        transforms.RandomCrop((128,128)),
        transforms.ToTensor(),
        ])
    elif method=='rafi':
        data_transforms = transforms.Compose([
        transforms.RandomCrop((64,64)),
        transforms.ToTensor(),
        ])
    else:
        data_transforms = transforms.Compose([
        transforms.ToTensor(),
        ])
        
    return data_transforms