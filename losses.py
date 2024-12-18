# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
import torch.nn as nn


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float, len_num_keep, maskedkd):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.len_num_keep = len_num_keep            
        self.maskedkd = maskedkd            


    def forward(self, inputs, outputs, labels, attn):
        
        
        len_keep = torch.topk(attn.mean(dim=1)[:,0,1:],self.len_num_keep).indices

        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs, len_keep)

        base_loss = self.base_criterion(outputs, labels)

        if self.distillation_type == 'soft':
            T = self.tau

            distillation_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1), 
                F.softmax(teacher_outputs/T, dim=1)) * (T * T)

        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha


        return loss
