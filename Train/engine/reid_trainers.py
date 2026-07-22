import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchvision
import numpy as np
from torch.autograd import Variable
from scipy import misc
from tqdm import tqdm
from reid.evaluation_metrics import accuracy
from reid.utils.meters import AverageMeter
from reid.utils.data.transforms import RandomErasing
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BaseTrainer(object):
    def __init__(self, model, criterion, summary_writer, prob=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.summary_writer = summary_writer
        self.normlizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.strong_aug = T.Compose([
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.3),
        ])  
        self.eraser = RandomErasing(probability=prob, mean=[0., 0., 0.])
    def train(self, epoch, data_loader, optimizer, random_erasing, empty_cache=False, print_freq=10):
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        cls_losses = AverageMeter()  
        tri_losses = AverageMeter()  
        precisions = AverageMeter()
        end = time.time()
        pbar = tqdm(data_loader, desc=f"Epoch [{epoch}]", unit="batch", leave=True)
        
        for i, inputs in enumerate(pbar):
            data_time.update(time.time() - end)
            ori_inputs, targets,domain_id = self._parse_data(inputs) 
            # print(targets)
            in_size = inputs[0].size()
            for j in range(in_size[0]):
                ori_inputs[0][j, :, :, :] = self.strong_aug(ori_inputs[0][j, :, :, :])
                
                # plt.clf()
                # plt.imshow(ori_inputs[0][j, :, :, :].cpu().permute(1, 2, 0))  
                # plt.title(f'sample {j} after erasing (before normalize)')
                # plt.pause(0.3)
                # =================================
                ori_inputs[0][j, :, :, :] = self.normlizer(ori_inputs[0][j, :, :, :])
                if random_erasing:
                    ori_inputs[0][j, :, :, :] = self.eraser(ori_inputs[0][j, :, :, :])
            loss, all_loss, prec1 = self._forward(ori_inputs, targets)
            losses.update(loss.item(), targets.size(0)) 
            cls_losses.update(all_loss[0].item(), targets.size(0))  
            tri_losses.update(all_loss[1].item(), targets.size(0))  
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
                'Cls': f"{cls_losses.avg:.3f}", 
                'Tri': f"{tri_losses.avg:.3f}",  
                'Avg': f"{precisions.avg:.2%}",  
                'prec': f"{precisions.val:.2%}", 
                # 'Lr': f"{optimizer.param_groups[0]['lr']:.1e}"  
            })
            
            if (i + 1) % print_freq == 0:
                if self.summary_writer is not None:
                    global_step = epoch * len(data_loader) + i
                    self.summary_writer.add_scalar('loss_s', losses.avg, global_step)
                    self.summary_writer.add_scalar('loss_cls/cls_s', cls_losses.avg, global_step)
                    self.summary_writer.add_scalar('loss_tri/tri_s', tri_losses.avg, global_step)
                    self.summary_writer.add_scalar('prec/acc_s', precisions.avg, global_step) 
    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError

class ReidTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs,pids,domain_id = inputs
        
        inputs = [imgs.cuda(non_blocking=True)]
        targets = pids.cuda(non_blocking=True)
        domain_id = domain_id.cuda(non_blocking=True)
        return inputs, targets,domain_id

    def _forward(self, inputs, targets):
        outputs = self.model(inputs, training=True)
        loss_cls = self.criterion[0](outputs[2], targets)
        loss_tri = self.criterion[1](outputs[0], targets)
        loss = loss_cls + loss_tri
       
        prec, = accuracy(outputs[2].data, targets.data)
        prec = prec[0]
        losses = [loss_cls, loss_tri]
        return loss, losses, prec


