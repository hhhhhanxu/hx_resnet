'''
Author: hanxu
Date: 2022-03-21 09:04:41
LastEditors:  
LastEditTime: 2022-03-21 19:39:07
FilePath: /resnet/test.py
Description: a test script for langqiao xingren classify

Copyright (c) 2022 by bip_hx, All Rights Reserved. 
'''
import tqdm
import torch

from torchnet import meter
from loguru import logger

def test(opt,device,model,dataloader,epoch):
    model = model.eval()
    cuda = device.type != 'cpu'
    correct=0

    confusion_matrix = meter.ConfusionMeter(3,normalized=True)

    pbar = tqdm.tqdm(enumerate(dataloader) ,total=len(dataloader))
    for index,(img,label) in pbar:
        if cuda:
            img,label = img.cuda(),label.cuda()
        output = model(img)
        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        confusion_matrix.add(output.data,label.data)
    
    cm_value = confusion_matrix.value()

    accuracy = 100.0 * (cm_value[0][0]+cm_value[1][1]+cm_value[2][2])/cm_value.sum()

    print("Epoch{} test -- accuracy:{}".format(epoch,accuracy))
    print("confusion_matrix",accuracy,"sum:",cm_value.sum())

    return accuracy,cm_value