import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from .loss import OhemCrossEntropy2d


class CriterionAll(nn.Module):
    def __init__(self, loss_type='softmax',ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        #self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        if loss_type == 'softmax':
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        if loss_type == 'ohem':
            self.criterion = OhemCrossEntropy2d(ignore_label=ignore_index)
        print (self.criterion)
   
    def parsing_loss(self, preds, target):
        h, w = target.size(1), target.size(2)


        loss = 0
        preds_parsing = preds
        scale_pred = F.interpolate(input=preds_parsing, size=(h, w), mode='bilinear', align_corners=True)
        loss += self.criterion(scale_pred, target)

        return loss

    def forward(self, preds, target):
          
        loss = self.parsing_loss(preds, target) 
        return loss
    