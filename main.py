import argparse
import os
import torch
import torch.optim as optim
import os.path


from models.liu import Res2Net
from models.huan import ConvNeXt
from models.benna import Net
from models.rana import NetRana
from models.sychandran import Sy_Net
from models.rafi import RemNet

import utils.dataLoader as dl
from utils.train_test import Train
from utils.pla_ila import patch_level_accuracy,image_level_predictions

print(torch.cuda.is_available())

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)  # Only for single GPU

def dataset_initializer(path,data_transform,batchsize,numWorkers,shuffle=True,filename_return=False,dual=False):
    if dual:
        dataImage = dl.ImageFolderLoaderDual(path,transform=data_transform,filename_return=filename_return)
        data_loader = torch.utils.data.DataLoader(
            dataImage, batch_size=batchsize,
            shuffle=shuffle, num_workers=numWorkers
            )
        return data_loader
    else:
        dataImage = dl.ImageFolderLoader(path,transform=data_transform,filename_return=filename_return)
        data_loader = torch.utils.data.DataLoader(
            dataImage, batch_size=batchsize,
            shuffle=shuffle, num_workers=numWorkers
            )
        return data_loader

 
def modelInit(method,dev,num_classes):
    model = None
    if method == 'liu':
        model = Res2Net(num_classes=num_classes).to(dev)
    elif method == 'huan':
        model = ConvNeXt(num_classes=num_classes).to(dev)
    elif method == 'bennabhaktula':
        model = Net(num_classes=num_classes).to(dev)
    elif method == 'sychandran':
        model = Sy_Net(num_classes=num_classes).to(dev)
    elif method == 'rana':
        model = NetRana(num_classes=num_classes).to(dev)
    elif method == 'rafi':
        model = RemNet(num_classes=num_classes).to(dev)
    return model

def optimInit(method,model, lr,momentum,weightDecay):
    optimizer = None
    scheduler = None
    
    if method == 'liu':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weightDecay)
    elif method=='sychandran':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif method == 'huan':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weightDecay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=65173)
    elif method== 'bennabhaktula':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif method == 'rana':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weightDecay)
    elif method == 'rafi':
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    return optimizer,scheduler

def main(src,dest,onlyTest,method,epochs,batchsize,lr,momentum, weightDecay, modelName,log,numWorkers,device):
    train_datapath = os.path.join(src,"train")
    val_datapath = os.path.join(src,"val")
    test_datapath = os.path.join(src,"test")
    
    classes, class_to_idx = dl.find_classes(test_datapath)
    
    data_transform = dl.transform_data(method)
    
    db=False
    clust=False
    if method=='rana':
        db = True
    if method=='rafi':
        clust=True
    
    train_loader = dataset_initializer(train_datapath,data_transform,batchsize,numWorkers,dual=db)
    val_loader = dataset_initializer(val_datapath,data_transform,batchsize,numWorkers,shuffle=False,dual=db)
    
    method = method.lower()
    model = modelInit(method,device,len(classes))
    optimizer,scheduler = optimInit(method,model,lr,momentum, weightDecay)
    
    print(model)
    
    if onlyTest == False:
        print("Training started!")
        history = Train(model, epochs=epochs,optimizer = optimizer,scheduler=scheduler, device=device,train_data=train_loader, valid_data=val_loader, save_path=modelName,csv_file=log,dual=db)
    

    test_loader = dataset_initializer(test_datapath,data_transform,1,numWorkers*4,shuffle=False,filename_return=True,dual=db)
    
    model.load_state_dict(torch.load(modelName, weights_only=True))
    pla,predicted_data = patch_level_accuracy(test_loader, model, result_folder=dest,dual=db)
    ila = image_level_predictions(dest,clust)
    
    print("Patch Level Accuracy: ",pla)
    print("Image Level Accuracy: ",ila)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provide source and destination paths")

    parser.add_argument(
        "--src", 
        type=str, 
        default=r"./patches", 
        help="Source directory path (default: ./patches)"
    )
    
    parser.add_argument(
        "--dest", 
        type=str, 
        default="./Result", 
        help="Prediction Resultions CSV storing directory path (default: ./Result)"
    )
    
    parser.add_argument(
        "--onlyTest", 
        type=bool, 
        default=False, 
        help="Would you like to only test True/False (default: false)"
    )
    
    parser.add_argument(
        "--method", 
        type=str, 
        default="liu", 
        help="SCMI method to use (default: liu)"
    )

    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1, 
        help="Number of epochs (default: 1)"
    )
    
    parser.add_argument(
        "--batchsize", 
        type=int, 
        default=64, 
        help="Batch Size (default: 64)"
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.01, 
        help="Learning Rate (default: 0.01)"
    )
    
    
    parser.add_argument(
        "--momentum", 
        type=float, 
        default=0.9, 
        help="Momentum (default: 0.9)"
    )
    
    parser.add_argument(
        "--weightDecay", 
        type=float, 
        default=0.00075, 
        help="weightDecay (default: 0.00075)"
    )
    
    parser.add_argument(
        "--modelname", 
        type=str, 
        default="bestModel.pth", 
        help="Best saved model name (default: bestModel.pth)"
    )
    
    parser.add_argument(
        "--log", 
        type=str, 
        default="log.csv", 
        help="CSV log file for training (default: log.csv)"
    )
    
    parser.add_argument(
        "--numWorkers", 
        type=int, 
        default=4, 
        help="Number of workers in GPU (default: 4)"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default='cuda:0', 
        help="Device: CPU or GPU (Cuda:i) (default: cuda:0)"
    )

    args = parser.parse_args()
    main(args.src, args.dest, args.onlyTest, args.method, args.epochs, args.batchsize, args.lr, args.momentum, args.weightDecay, args.modelname,args.log,args.numWorkers,args.device)
