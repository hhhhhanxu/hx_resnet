'''
Author: hanxu
Date: 2022-03-21 09:04:37
LastEditors:  
LastEditTime: 2022-03-21 20:35:24
FilePath: /resnet/train.py
Description: 

Copyright (c) 2022 by bip_hx, All Rights Reserved. 
'''
import os
import argparse 
import wandb
import tqdm
from loguru import logger
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import langqiao_dataset
from model import My_resnet
from test import test
from utils.utils import select_device,init_torch_seeds
from utils.warmup import CosineAnnealingLRWarmup
from utils.smooth_loss import LabelSmoothingCrossEntropy


@logger.catch
def train(opt,device):
    #先写好dataloader
    train_loader = DataLoader(
        langqiao_dataset(root_path=opt.data_dir,transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                        ])),
        batch_size = opt.batch_size,
        shuffle=True,
        num_workers=8
    )

    test_loader = DataLoader(
        langqiao_dataset(root_path=opt.data_dir,train=False,transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                        ])),
        batch_size = 1,
        shuffle = False, #batchsize=1 这就没办法写成True
        num_workers =2
    )
    #加载模型
    model = My_resnet(pretrained=opt.use_pretrained,model_size=opt.model_size,num_classes=opt.num_class)
    cuda = device.type != 'cpu'
    if cuda: 
        model = model.cuda()
    #损失函数
    # criterion = LabelSmoothingCrossEntropy()  # 标签平滑
    criterion = nn.CrossEntropyLoss()
    #优化器
    optimizer = optim.SGD(model.parameters(),lr=opt.lr,momentum=opt.momentum)
    #学习率调整
    scheduler = CosineAnnealingLRWarmup(optimizer,T_max=100,eta_min=opt.lr/100,warmup_steps=15)

    # test for every epoch 
    best_acc = 0
    
    for epoch in range(opt.epochs):
        model.train()
        pbar = tqdm.tqdm(enumerate(train_loader),total=len(train_loader))
        # train one epoch
        for batch_idx ,(data,label) in pbar:
            if cuda:
                data,label = data.cuda(),label.cuda()
            optimizer.zero_grad()
            output = model(data) 
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            pbar.set_description('epoch:{}   {}/{}'.format(epoch,batch_idx+1,len(train_loader)))
        scheduler.step()
        wandb.log({'train/loss':loss})
        wandb.log({'train/lr':scheduler.get_last_lr()[0]})
        # test for every epoch 
        accuracy,cm_value = test(opt,device,model,test_loader,epoch)
        wandb.log({'test/accuracy':accuracy})
        # wandb upload cm img
        plt.imshow(cm_value,cmap=plt.cm.hot,vmin=0,vmax=1)
        plt.colorbar()
        plt.savefig(opt.save_dir+'/{}.png'.format(epoch))
        if opt.epochs - epoch < 0.1*opt.epochs:
            #最后10%的epoch上传 cm img
            wandb.log({"Confusion_matrix": [wandb.Image(str(x), caption=x.name) for x in Path(opt.save_dir).glob('*.png')]})
        if accuracy > best_acc:
            torch.save(model.state_dict(),os.path.join(opt.save_dir,'best.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='a litte effort for my resnet')
    parser.add_argument('--data_dir',type=str,default="/data_plus/dataset/my_fgvc_dataset/",required=True)
    parser.add_argument('--save_dir',type=str,default='result/',required=True)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--momentum',type=float,default=0.937)
    parser.add_argument('--num_class',type=int,default=3)
    parser.add_argument('--device', default='0,1,2,3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--use_pretrained',action='store_true',help='use pretrained resnet model')
    parser.add_argument('--model_size', type=int, default=50, choices=[18,34,50,101,152],help='the size of resnet')
    
    opt = parser.parse_args()

    init_torch_seeds()
    device = select_device(device=opt.device,model_size=opt.model_size,batch_size=opt.batch_size)
    wandb_run = wandb.init(
        config = opt,
        project = 'Resnet',
        name = 'Resnet'+str(opt.model_size),
    )

    train(opt=opt,device=device)
