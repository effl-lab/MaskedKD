"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import habana_frameworks.torch.gpu_migration
import habana_frameworks.torch.core as htcore
from transformers.modeling_outputs import ImageClassifierOutput

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.is_autocast):
            outputs = model(samples)    
            attn = None 
            if isinstance(outputs, tuple):
                outputs, attn = outputs
            if isinstance(outputs, ImageClassifierOutput):
                outputs = outputs.logits
            loss  = criterion(samples, outputs, targets, attn)

        if (str(args.device) == 'hpu') and args.run_lazy_mode:
            htcore.mark_step()
        loss.backward()

        if (str(args.device) == 'hpu') and args.run_lazy_mode:
            htcore.mark_step()

        print("output: " , outputs.shape, outputs.device)
        print("loss: " , loss.shape, loss.device)
        loss_value = loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (str(args.device) == 'hpu'):
            optimizer.step()
            if args.run_lazy_mode:
                htcore.mark_step()
        else:
            optimizer.step()
        optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=args.is_autocast):
            output = model(images)    
            if isinstance(output, tuple):
                output, _ = output     
            if isinstance(output, ImageClassifierOutput):
                output = output.logits
                    
                loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
