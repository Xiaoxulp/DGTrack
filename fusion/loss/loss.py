import torch
import torch.nn as nn

class BCEWeightLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, gt):
        x = x.reshape(-1)
        gt = gt.reshape(-1)
        pos_sum = gt.sum()
        neg_sum = len(gt) - pos_sum
        if pos_sum == 0 or neg_sum == 0:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        pa = len(gt)*len(gt)/(len(gt)-gt.sum())/2/gt.sum()
        weight_0 = pa*gt.sum()/len(gt)
        weight_1 = pa*(len(gt) - gt.sum())/len(gt)
        weight = torch.zeros(len(gt))
        for i in range(len(gt)):
            if gt[i] == 0:
                weight[i] = weight_0
            elif gt[i] == 1:
                weight[i] = weight_1
            else:
                raise RuntimeError('loss weight error')
        #weight = torch.abs(gt-0.1).detach()
        loss = nn.BCELoss(weight=weight.cuda())
        
        # loss = nn.BCELoss(reduction='sum')
        # print(gt.sum())
        out = loss(x, gt)
        return out

class SparseMatchingLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, sim_matrix, gt_matrix):
        #scale
        logits = sim_matrix / self.temperature
        
        log_den = torch.logsumexp(logits, dim=-1)
       
        pos_logits = logits.masked_fill(gt_matrix != 1, -1e9)
        log_num = torch.logsumexp(pos_logits, dim=-1)  # [B, T]
        
        loss = log_den - log_num
       
        has_pos_mask = (gt_matrix == 1).sum(dim=-1) > 0
        if has_pos_mask.sum() == 0:
            return torch.tensor(0.0).to(loss.device).requires_grad_(True)
        return loss[has_pos_mask].mean()
