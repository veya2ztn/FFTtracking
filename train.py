import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.autograd import Variable

from visualization import *
from image_process import*
from dataload import *
import cv2

train_path="/media/tianning/DATA/VOT/VOT2018"
one_classp='/media/tianning/DATA/VOT/VOT2018/bag'
pre_model_path='/media/tianning/DATA/Pre_Model/YOLACT/yolact_resnet50_54_800000.pth'
#xes,yes,zes,tes=get_batch_input(dataloader,imgprocesser,1,True)

from KCF import KCF
from models import DMLKCF

from backbone import *
from config import *
from model_utils import *
epoches   = 10
batches   = 10000
batch     = 20
backbone_config = resnet50_backbone
backbone_config.selected_layers=[0,1]
fpn_config      = fpn_base

cfg=Config({
    'backbone':resnet50_backbone,
    'fpn':fpn_base
})
model=DMLKCF(cfg)
_=model.cuda()
_=model.apply(rand_weight_init)
model.load_weights(pre_model_path)


#--------------------------------------
##config the dataset and pre-process
dataloader=MyDataLoader(train_path)
dataloader.one_class=True
dataloader.set_one_class(one_classp)
dataloader.isfix_frame=False
dataloader.fixed_frame=8
dataloader.delta_frame=dataloader.fixed_frame
imgprocesser=data_process(128,256)
imgprocesser.hanning_for_detected = False
#imgp.distribution = 'normal'
imgprocesser.distribution = 'ratio_form'

from dataaccelerate import DataLoader as DataLoader
from dataaccelerate import DataPrefetcher
dataloader.imgprocesser=imgprocesser
#dataloader.length = batches
data_loader=DataLoader(
    dataset=dataloader,
    num_workers=4,
    batch_size=batch,
    pin_memory=True,
    shuffle=False,
)
prefetcher = DataPrefetcher(data_loader)

# set the logging handle
import sys
from train_utils import AverageMeter,RecordLoss,adjust_learning_rate

losses  = AverageMeter()
log_hand= RecordLoss([losses],[[]],100)


cur_lr    = 0.1

cuda=True
optimizer = torch.optim.SGD(model.parameters(), lr=cur_lr, momentum=0.01,weight_decay=0.01)
dataloader.length = batches
from tqdm import tqdm
epoch_bar=tqdm(range(epoches))
#batch_bar=tqdm(range(batches))
## train head
steps=0
for epoch in epoch_bar:
    cur_lr = adjust_learning_rate(cur_lr, optimizer, epoch, gamma=0.1)
    #for iters in batch_bar:
    #    xes,yes,zes,tes=get_batch_input(dataloader,imgprocesser,batch,cuda)
    batch_bar=tqdm(total=len(dataloader))
    iters = 0
    next_batch = prefetcher.next()
    while next_batch is not None:
        iters += 1
        if iters >= batches:break
        (xes,yes,zes),tes = next_batch
        loss = model([zes,xes,yes],tes)
        if torch.isnan(loss):sys.exit(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_hand.step(steps,[loss])
        tqdm.write("epoch:{:3}/{}  batch:{:3}/{} loss:{:.4f}".format(epoch,epoches,iters,batches,loss.cpu().item()))

        steps += 1
        batch_bar.update(1)
        if steps%100==1:
            log_hand.print2file(steps,"loss.log")

        if steps%500==1:
            with torch.no_grad():response=model([zes,xes,yes])
            filename="debug/[epo{}][batch{}].jpg".format(epoch,iters)
#            cv2.imwrite(filename,debug_see(dataloader.imgprocesser,xes[-1:],yes[-1:],zes[-1:],tes[-1:],response)*255)

        if (steps) % 2000 == 1:
            file_path = os.path.join('checkpoints', 'weights-{:07d}.pth.tar'.format(steps))
            model.save_weights(file_path)
        next_batch = prefetcher.next()

file_path = os.path.join('weights', 'weights-{:07d}.pth.tar'.format(steps))
model.save_weights(file_path)
