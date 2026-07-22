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
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-10.0 * phase * phase))
class BaseTrainer(object):
    def __init__(self, model, ema_model, criterion, summary_writer, prob=0.5, mean=[0.4914, 0.4822, 0.4465]):
        super(BaseTrainer, self).__init__()
        self.model = model  
        self.ema_model = ema_model  
        self.criterion = criterion  
        self.summary_writer = summary_writer
        self.normlizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.strong_aug = T.Compose([
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.3),
        ])  
        self.eraser = RandomErasing(probability=prob, mean=[0., 0., 0.])

    def train(self, epoch, data_loader, optimizer, random_erasing, alpha=0.0, empty_cache=False, print_freq=5):
        start_rampup = 10
        rampup_length = 40
        cons_max = 1.0
        if epoch < start_rampup:
            cons_weight = 0.0
        else:
            cons_weight = cons_max * sigmoid_rampup(epoch - start_rampup, rampup_length)
        print(cons_weight)
        self.model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        losses_s = AverageMeter()
        cls_losses_s = AverageMeter()  
        tri_losses_s = AverageMeter()  
        domain_loss_s = AverageMeter() 
        domain_accuracies = AverageMeter()  
        precisions_s = AverageMeter()
        domain_acc_s = AverageMeter()
        
        loss_kl_m = AverageMeter()

       
        cls_losses_t = AverageMeter()  
        tri_losses_t = AverageMeter() 
        precisions_t = AverageMeter()

        end = time.time()
        pbar = tqdm(data_loader, desc=f"Epoch [{epoch}]", unit="batch", leave=True)
        for i, inputs in enumerate(pbar):
            data_time.update(time.time() - end)
            ori_inputs, targets, domain_ids = self._parse_data(inputs)  
            imgs_tensor = ori_inputs[0]
           
            imgs_teacher = imgs_tensor.clone()  
            B, C, H, W = imgs_teacher.shape
            for j in range(B):
               
                # plt.clf()
                # plt.imshow(imgs_teacher[j].cpu().permute(1, 2, 0))  
                # plt.title(f'epoch{epoch}-iter{i}-sample{j}  teacher (no erase, before norm)')
                # plt.axis('off')
                # plt.pause(0.3)
                # ======================================
                imgs_teacher[j] = self.normlizer(imgs_teacher[j])
           
            imgs_student = imgs_tensor.clone()
            B, C, H, W = imgs_student.shape
            in_size = inputs[0].size()
            for j in range(B):
                imgs_student[j] = self.strong_aug(imgs_student[j])
                
                # plt.clf()
                # plt.imshow(imgs_student[j].cpu().permute(1, 2, 0))  
                # plt.title(f'sample {j} after erasing (before normalize)')
                # plt.pause(0.3)
                # =================================
                imgs_student[j] = self.normlizer(imgs_student[j])
                if random_erasing:
                    imgs_student[j] = self.eraser(imgs_student[j])
            loss_student, all_loss, prec_student, domain_acc_student, prec_teacher = self._forward(
                [imgs_student], [imgs_teacher], targets, domain_ids, alpha, cons_weight=cons_weight)
           
            losses_s.update(loss_student.item(), targets.size(0))
            cls_losses_s.update(all_loss[0].item(), targets.size(0))  
            tri_losses_s.update(all_loss[1].item(), targets.size(0))  
            domain_loss_s.update(all_loss[2].item(), targets.size(0))  
            precisions_s.update(prec_student.item(), targets.size(0))
            domain_accuracies.update(domain_acc_student.item(), targets.size(0))  
            
            cls_losses_t.update(all_loss[3].item(), targets.size(0))  
            tri_losses_t.update(all_loss[4].item(), targets.size(0))  
            precisions_t.update(prec_teacher.item(), targets.size(0))
            
            loss_kl_m.update(all_loss[5].item(), targets.size(0))

            optimizer.zero_grad()
            loss_student.backward()
            optimizer.step()
            
            self.ema_model.update(self.model, global_step=epoch * len(data_loader) + i)  
            if empty_cache:
                torch.cuda.empty_cache()
            batch_time.update(time.time() - end)
            end = time.time()
            
            pbar.set_postfix({
                'Loss_S': f"{losses_s.avg:.4f}",  
                # 'Cls_S': f"{all_loss[0].item():.3f}",  
                'Cls': f"{cls_losses_s.avg:.3f}",  
                'Tri_S': f"{tri_losses_s.avg:.3f}",  
                'Dom_S': f"{domain_loss_s.avg:.3f}",  
                'Avg_S': f"{precisions_s.avg:.2%}",  
                'prec_S': f"{precisions_s.val:.2%}",  
                'KL': f"{loss_kl_m.avg:.3f}",  
                'DomainAcc': f"{domain_accuracies.avg:.2%}",  
                'Cls': f"{cls_losses_t.avg:.3f}",  
                'Tri_T': f"{tri_losses_t.avg:.3f}",  
                'Avg_T': f"{precisions_t.avg:.2%}",  
                'prec_T': f"{precisions_t.val:.2%}",  
                # 'Lr': f"{optimizer.param_groups[0]['lr']:.1e}"  
            })
            
            if (i + 1) % print_freq == 0:
                if self.summary_writer is not None:
                    global_step = epoch * len(data_loader) + i
                    
                    self.summary_writer.add_scalar('loss_s', losses_s.avg, global_step) 
                    self.summary_writer.add_scalar('loss_cls/cls_s', cls_losses_s.avg, global_step)
                    self.summary_writer.add_scalar('loss_tri/tri_s', tri_losses_s.avg, global_step)
                    self.summary_writer.add_scalar('loss_domain/domain_s', domain_loss_s.avg, global_step)
                    self.summary_writer.add_scalar('prec/acc_s', precisions_s.avg, global_step)
                    self.summary_writer.add_scalar('dprec/acc_s',domain_accuracies.avg,global_step)
                    
                    self.summary_writer.add_scalar('Distill_kl', loss_kl_m.avg, global_step)
                    
                    self.summary_writer.add_scalar('loss_cls/cls_t', cls_losses_t.avg, global_step)
                    self.summary_writer.add_scalar('loss_tri/tri_t', tri_losses_t.avg, global_step)
                    self.summary_writer.add_scalar('prec/acc_t', precisions_t.avg, global_step)
        pbar.close()

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs_s, inputs_t, targets, domain_ids, alpha, cons_weight):
        raise NotImplementedError

class ReidTrainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs,pids,domain_id = inputs
        
        inputs = [imgs.cuda(non_blocking=True)]
        targets = pids.cuda(non_blocking=True)
        domain_id = domain_id.cuda(non_blocking=True)
        return inputs, targets,domain_id

    def _forward(self, inputs_s, inputs_t, targets, domain_ids, alpha, cons_weight):
       
        outputs_student = self.model(inputs_s, alpha=alpha, training=True)
       
        loss_cls_student = self.criterion[0](outputs_student[2], targets)
        loss_tri_student = self.criterion[1](outputs_student[0], targets)
        loss_domain_student = self.criterion[2](outputs_student[3], domain_ids)
        
        with torch.no_grad():
            outputs_teacher = self.ema_model.teacher(inputs_t, alpha=0.0, training=True) 
            
            loss_cls_teacher = self.criterion[0](outputs_teacher[2], targets)
            loss_tri_teacher = self.criterion[1](outputs_teacher[0], targets)
        
        prec_student, = accuracy(outputs_student[2].data, targets.data)
        prec_student = prec_student[0]
        
        prec_teacher, = accuracy(outputs_teacher[2].data, targets.data)
        prec_teacher = prec_teacher[0]
        
        domain_pred_student = outputs_student[3].max(1)[1]
        domain_acc_student = (domain_pred_student == domain_ids).float().mean()
        
        loss_kl = self.criterion[3](outputs_student[0], outputs_teacher[0].detach())
        loss_kl = cons_weight * loss_kl
        loss_student = loss_cls_student + loss_tri_student + loss_domain_student * 2.3 + loss_kl
        losses = [loss_cls_student, loss_tri_student, loss_domain_student, loss_cls_teacher, loss_tri_teacher,loss_kl]
        return loss_student, losses, prec_student, domain_acc_student, prec_teacher

