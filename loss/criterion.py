import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .loss import SoftmaxDSN, Softmax, OhemCrossEntropy
from .lovasz_losses import  LovaszSoftmaxDSN, LovaszSoftmax
from utils.encoding import DataParallelCriterion 

class CrossEntropy(nn.Module):
    def __init__(self,args, ignore_index=255,  reduce=True):
        super(CrossEntropy, self).__init__()
        self.dsn = args.dsn
        if args.dsn:
            self.softmax = SoftmaxDSN()
        else:
            self.softmax = Softmax()
    def forward(self, preds, target):
        return self.softmax(preds, target)

class LovaszLoss(nn.Module):
    def __init__(self, args, input_size,per_image=False, ignore=255):
        super(LovaszLoss, self).__init__()
        if args.dsn:
            self.lovasz = LovaszSoftmaxDSN(input_size)
        else:
            self.lovasz = LovaszSoftmax(input_size)
    def forward(self, preds, target):
        return self.lovasz(preds, target)


class Seg_Loss(nn.Module):
    def __init__(self,args,input_size, ignore_index=255, use_weight=True, reduce=True):
        super(Seg_Loss, self).__init__()
        self.iouloss = args.iouloss
        self.ohem = args.ohem
        if self.ohem:
            softmax =  OhemCrossEntropy(args)
        else:
            softmax =  CrossEntropy(args)
        self.softmax = DataParallelCriterion(softmax)
        if self.iouloss:
            lovasz_softmax = LovaszLoss(args, input_size)
            self.lovasz_softmax = DataParallelCriterion(lovasz_softmax)

    def forward(self, preds, target):
        loss = []
        loss.append(self.softmax(preds, target)) 
        if self.iouloss:
            loss.append(self.lovasz_softmax(preds, target))
        
        return loss


