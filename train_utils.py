import torch
import torch.nn as nn
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.win100=[]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        #self.avg = self.sum / self.count
        self.win100.append(val)
        if len(self.win100) > 100:_=self.win100.pop(0)
        sum100=sum(self.win100)
        self.avg=1.0*sum100/len(self.win100)
import torch.nn.functional as F
class MultiBoxLoss(nn.Module):
    def __init__(self):
        self.lambdaa=10
        super(MultiBoxLoss, self).__init__()

    def forward(self, predictions, targets):
        #print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
        cout, rout = predictions
        if cout.shape[0]==2:cout=cout.transpose(1,0)
        if rout.shape[0]==4:rout=rout.transpose(1,0)
        assert cout.shape[1]==2
        assert rout.shape[1]==4
        """ class """
        class_pred, class_target = cout, targets[:, 0].long()
        pos_index , neg_index    = list(np.where(class_target == 1)[0]), list(np.where(class_target == 0)[0])
        pos_num, neg_num         = len(pos_index), len(neg_index)
        class_pred, class_target = class_pred[pos_index + neg_index], class_target[pos_index + neg_index]

        closs = F.cross_entropy(class_pred, class_target, reduction='none')
        closs = torch.div(torch.sum(closs), 64)

        """ regression """
        reg_pred   = rout
        reg_target = targets[:, 1:]
        rloss = F.smooth_l1_loss(reg_pred, reg_target, reduction='none') #1445, 4
        rloss = torch.div(torch.sum(rloss, dim = 1), 4)
        rloss = torch.div(torch.sum(rloss[pos_index]), 16)

        lambdaa=self.lambdaa
        loss = closs + lambdaa*rloss
        return closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

import copy
class RecordLoss:

    def __init__(self,init_loss_records,init_loss_show,x_window):
        self.loss_records=init_loss_records
        self.init_loss_show=copy.deepcopy(init_loss_show)
        self.loss_showeds=copy.deepcopy(init_loss_show)
        self.x_window = x_window
        self.x=[]
        self.x_bounds=[0,x_window]
        self.y_bounds=[0,100]
        self.x_start = x_window
    def record_loss(self,loss_recorder):
        return loss_recorder.avg

    def update_record(self,recorder,loss):
        recorder.update(loss.cpu().item())

    def update(self,loss_list):
        for loss_recorder,loss_showed,loss in zip(self.loss_records,self.loss_showeds,loss_list):
            self.update_record(loss_recorder,loss)
            if loss_showed is not None:
                loss_showed.append(self.record_loss(loss_recorder))
    def reset(self):
        self.x=[]
        self.loss_showeds=copy.deepcopy(self.init_loss_show)

    def step(self,step,loss_list):
        x_window=self.x_window
        if step >self.x_start and step%x_window==1:
            graphs=[show for show in self.loss_showeds if show is not None]
            self.x_bounds = [0, self.x_window]
            self.y_window = max(max(graphs))-min(min(graphs))
            now_at        = self.record_loss(self.loss_records[0])
            self.y_bounds = [now_at-self.y_window, now_at+0.2*self.y_window]
            self.reset()
        self.x.append(step%x_window)
        self.update(loss_list)


    def update_graph(self,mb,step):
        if step >self.x_start:
            graphs=[show for show in self.loss_showeds if show is not None]
            graphs=[[self.x]+graphs]
            x_bounds=self.x_bounds
            y_bounds=self.y_bounds
            mb.update_graph(graphs, x_bounds, y_bounds)
            #return ll

    def print2file(self,step,file_name):
        with open(file_name,'a') as log_file:
            ll=["{:.4f}".format(self.record_loss(recorder)) for recorder in self.loss_records]
            printedstring=str(step)+' '+' '.join(ll)+'\n'
            #print(printedstring)
            _=log_file.write(printedstring)
