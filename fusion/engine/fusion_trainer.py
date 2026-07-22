import time
import torch
import numpy as np
from tqdm import tqdm
from reid.utils.meters import AverageMeter
from fusion.Instance.track_instance import STrack
from fusion.Instance import matching
from scipy.spatial.distance import cdist
from fusion.Instance.matching import ious
import torch.nn.functional as F

def cal_acc(x, gt):
    if x.dim() == 3:
        x = x.squeeze(0)
    if gt.dim() == 3:
        gt = gt.squeeze(0)
    if x.shape[0] > x.shape[1]:
        x, gt = x.t(), gt.t()

    has_match = gt.sum(1) > 0
    if has_match.sum() == 0:
        return 0, 0
    pred = x.max(1).indices
    target = gt.max(1).indices
    correct = (pred[has_match] == target[has_match]).sum().item()
    return correct, has_match.sum().item()


def cal_match_acc(sim_matrix, gt_matrix, thresh=0.15):
    
    sim_np = sim_matrix.detach()[0].cpu().numpy()
    gt_np = gt_matrix.detach()[0].cpu().numpy()

    
    gt_dists = 1 - gt_np
    matches, u_track, u_det = matching.linear_assignment(gt_dists, thresh=thresh)

    
    pred_dists = 1 - sim_np
    pred_matches, pred_u_track, pred_u_det = matching.linear_assignment(pred_dists, thresh=thresh)

    
    gt_match_set = set((int(i), int(j)) for i, j in matches)
    pred_match_set = set((int(i), int(j)) for i, j in pred_matches)

    gt_u_track_set = set(int(i) for i in u_track)
    pred_u_track_set = set(int(i) for i in pred_u_track)

    gt_u_det_set = set(int(i) for i in u_det)
    pred_u_det_set = set(int(i) for i in pred_u_det)

    
    match_correct = len(gt_match_set & pred_match_set)
    track_correct = len(gt_u_track_set & pred_u_track_set)
    det_correct = len(gt_u_det_set & pred_u_det_set)
    correct = match_correct + track_correct + det_correct
    total = len(gt_match_set) + len(gt_u_track_set) + len(gt_u_det_set)
    return correct, total

class FusionTrainer:
    def __init__(self, model, criterion, summary_writer):
        self.model = model
        self.criterion = criterion
        self.summary_writer = summary_writer

    def train(self, epoch, data_loader, optimizer, device, print_freq=500):
        self.model.train()
       
        batch_time = AverageMeter()
        loss_meter = AverageMeter() 
        loss_iou_meter = AverageMeter() 
        loss_reid_meter = AverageMeter() 
        acc_meter = AverageMeter() 

        end = time.time()
        
        epoch_correct = 0
        epoch_total = 0
        
        fusion_tp = 0
        fusion_total = 0
        iou_tp = 0
        iou_total = 0
        reid_tp = 0
        reid_total = 0
        fusion_acc_meter = AverageMeter()
        iou_acc_meter = AverageMeter()
        reid_acc_meter = AverageMeter()

        pbar = tqdm(data_loader, desc=f"Epoch [{epoch}]", unit="batch", leave=True)
        for batch_idx, batch in enumerate(pbar):
            imgs = batch['imgs']
            targets = batch['gt_instances']
            track_imgs = imgs[0]
            det_imgs = imgs[1]
            track_targets = targets[0]
            det_targets = targets[1]
            optimizer.zero_grad()
            
            track_ids = track_targets['obj_ids'].to(device)  # [num_boxes]
            det_ids = det_targets['obj_ids'].to(device)

            det_geo = det_targets['boxes_norm'].to(device).unsqueeze(0)  # [num_boxes, 4]
            track_geo = track_targets['boxes_norm'].to(device).unsqueeze(0)

            # crop_track = track_imgs
            # crop_track_tensor = torch.stack(crop_track).to(device)
            # track_reid = self.backbone(crop_track_tensor).unsqueeze(0)
            
            # crop_det = det_imgs
            # crop_det_tensor = torch.stack(crop_det).to(device)
            # det_reid = self.backbone(crop_det_tensor).unsqueeze(0)
            track_reid = torch.stack(track_imgs).to(device).unsqueeze(0)  # [1, Nt, 2048]
            
            det_reid = torch.stack(det_imgs).to(device).unsqueeze(0)
            
            track_tlbr = track_targets['boxes_tlxy'].cpu().numpy()
            det_tlbr = det_targets['boxes_tlxy'].cpu().numpy()
            iou_np = ious(track_tlbr, det_tlbr)
            iou_matrix = torch.from_numpy(iou_np).float().unsqueeze(0).to(device)
            
            num_tracks = len(track_ids)
            num_dets = len(det_ids)
            gt_matrix = torch.zeros((num_tracks, num_dets), dtype=torch.float32, device=device)
            for t_idx, t_id in enumerate(track_ids):
                for d_idx, d_id in enumerate(det_ids):
                    if t_id == d_id:
                        gt_matrix[t_idx, d_idx] = 1.0
            dists = 1 - gt_matrix.cpu().numpy()
            gt_matrix = gt_matrix.unsqueeze(0)
            
            reid_sim = torch.matmul(
                F.normalize(track_reid, dim=-1),
                F.normalize(det_reid, dim=-1).transpose(-1, -2)
            )
            reid_sim = (reid_sim + 1) / 2
            reid_matrix = reid_sim.to(device)
            cost_matrix,track_out,det_out = self.model(det_reid=det_reid,det_geo=det_geo,track_reid=track_reid,track_geo=track_geo)
            
            loss = self.criterion(cost_matrix, gt_matrix)
            loss_iou = self.criterion(iou_matrix.detach(), gt_matrix)
            loss_reid = self.criterion(reid_matrix.detach(),gt_matrix)
            loss.backward()
            optimizer.step()
            
            # fusion
            correct, total = cal_match_acc(cost_matrix, gt_matrix)
            if total > 0:
                fusion_acc_meter.update(correct / total)
                fusion_tp += correct
                fusion_total += total

            # iou
            correct, total = cal_match_acc(iou_matrix, gt_matrix)
            if total > 0:
                iou_acc_meter.update(correct / total)
                iou_tp += correct
                iou_total += total

            # reid
            correct, total = cal_match_acc(reid_sim, gt_matrix)
            if total > 0:
                reid_acc_meter.update(correct / total)
                reid_tp += correct
                reid_total += total

            
            with torch.no_grad():
                gt_clean = gt_matrix.clone()
                gt_clean[gt_clean < 0] = 0
                correct_num, total_num = cal_acc(cost_matrix, gt_clean)
                epoch_correct += correct_num
                epoch_total += total_num
            loss_meter.update(loss.item()) 
            loss_iou_meter.update(loss_iou.item()) 
            loss_reid_meter.update(loss_reid.item()) 
            batch_acc = correct_num / max(total_num, 1)
            acc_meter.update(batch_acc)
            batch_time.update(time.time() - end)
            end = time.time()
            
            pbar.set_postfix({
                'Loss': f"{loss_meter.avg:.5f}",
                'l_i': f"{loss_iou_meter.avg:.5f}", 
                'l_r': f"{loss_reid_meter.avg:.5f}",
                'Ac': f"{acc_meter.avg:.4%}",
                # 'Match': f"({epoch_correct}/{epoch_total})", 

                "Fc": f"{fusion_acc_meter.avg:.3f}",
                # "FM": f"({fusion_tp} / {fusion_total})",
                # "Ic": f"{iou_acc_meter.avg:.3f}",
                # "IM": f"({iou_tp} / {iou_total})",
                # "Rc": f"{reid_acc_meter.avg:.3f}",
                # "RM": f"({reid_tp} / {reid_total})",
            })
            
            if self.summary_writer is not None and (batch_idx + 1) % print_freq == 0:
                global_step = epoch * len(data_loader) + batch_idx
                self.summary_writer.add_scalar('loss',loss_meter.avg, global_step)
                self.summary_writer.add_scalar('accuracy', acc_meter.avg, global_step)
                
                self.summary_writer.add_scalars('loss_tri', {
                    'fusion': loss_meter.avg,
                    'iou': loss_iou_meter.avg,
                    'reid': loss_reid_meter.avg,
                }, global_step)
                
               
                self.summary_writer.add_scalars('match', {
                    'fusion_match':fusion_acc_meter.avg,
                    'iou_match':iou_acc_meter.avg,
                    'reid_match':reid_acc_meter.avg,
                }, global_step)