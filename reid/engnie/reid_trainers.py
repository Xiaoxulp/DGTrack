import time
import sys
import os

import torch
import torchvision
import numpy as np
from torch.autograd import Variable
from scipy import misc
from tqdm import tqdm
from reid.evaluation_metrics import accuracy
from reid.utils.meters import AverageMeter
from reid.utils.data.transforms import RandomErasing


class BaseTrainer(object):
    def __init__(self, model, criterion, summary_writer, prob=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.summary_writer = summary_writer
        self.normlizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.eraser = RandomErasing(probability=prob, mean=[0., 0., 0.])

    def train(self, epoch, data_loader, optimizer, random_erasing, empty_cache=False, print_freq=20):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        end = time.time()
        pbar = tqdm(data_loader, desc=f"Epoch [{epoch}]", unit="batch", leave=True)
        # for i, inputs in enumerate(data_loader):
        for i, inputs in enumerate(pbar):
            data_time.update(time.time() - end)
            ori_inputs, targets = self._parse_data(inputs)
            in_size = inputs[0].size()
            for j in range(in_size[0]):
                ori_inputs[0][j, :, :, :] = self.normlizer(ori_inputs[0][j, :, :, :])
                if random_erasing:
                    ori_inputs[0][j, :, :, :] = self.eraser(ori_inputs[0][j, :, :, :])
            loss, all_loss, prec1 = self._forward(ori_inputs, targets)
            # losses.update(loss.data, targets.size(0))
            # precisions.update(prec1, targets.size(0))
            losses.update(loss.data, targets.size(0))
            precisions.update(prec1, targets.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if empty_cache:
                torch.cuda.empty_cache()
            batch_time.update(time.time() - end)
            end = time.time()
            pbar.set_postfix({
                'Loss': f"{losses.avg:.4f}",  
                'Cls': f"{all_loss[0].item():.3f}",  
                'Tri': f"{all_loss[1].item():.3f}",  
                'Avg': f"{precisions.avg:.2%}", 
                'prec': f"{precisions.val:.2%}", 
                'Lr': f"{optimizer.param_groups[0]['lr']:.1e}"  
            })
           
            if (i + 1) % print_freq == 0:
                if self.summary_writer is not None:
                    global_step = epoch * len(data_loader) + i
                    self.summary_writer.add_scalar('loss', loss.item(), global_step)
                    self.summary_writer.add_scalar('loss_cls', all_loss[0], global_step)
                    self.summary_writer.add_scalar('loss_tri', all_loss[1], global_step)
        pbar.close()

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class ReidTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        
        inputs = [imgs.cuda(non_blocking=True)]
        targets = pids.cuda(non_blocking=True)
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(inputs, training=True)

        loss_cls = self.criterion[0](outputs[2], targets)
        loss_tri = self.criterion[1](outputs[0], targets)

        loss = loss_cls + loss_tri
        losses = [loss_cls, loss_tri]
        prec, = accuracy(outputs[2].data, targets.data)
        prec = prec[0]
        return loss, losses, prec

