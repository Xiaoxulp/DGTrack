import time
import torch
from tqdm import tqdm
from reid.utils.meters import AverageMeter
import matplotlib.pyplot as plt
import numpy as np
class TrackBookTrainer(object):
    def __init__(self, clip_model, trackbook, criterion, summary_writer, device, clip_mean, clip_std):
        self.clip_model = clip_model
        self.trackbook = trackbook
        self.criterion = criterion
        self.summary_writer = summary_writer
        self.device = device
        
        self.mean_tensor = torch.tensor(clip_mean).view(1, 3, 1, 1).to(device)
        self.std_tensor = torch.tensor(clip_std).view(1, 3, 1, 1).to(device)
    def train(self, epoch, data_loader, optimizer, print_freq=5):
        self.trackbook.train()  
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        end = time.time()
        pbar = tqdm(data_loader, desc=f"Epoch [{epoch}]", unit="bt", leave=True)
        for i, (images, pids, domains) in enumerate(pbar):
            images = images.to(self.device).float()
            pids = pids.to(self.device)
            with torch.no_grad():
                # Resize + Normalize
                images_resized = torch.nn.functional.interpolate(images, size=(224, 224), mode='bicubic',align_corners=False)
                
                # ==========================================
                # if epoch == 1:
                #     print("\n[DEBUG] Visualizing first batch (Resized 224x224)...")
                #     B = images_resized.shape[0]
                #     cols = 8
                #     rows = (B + cols - 1) // cols
                #     plt.figure(figsize=(16, 2 * rows))
                #     plt.suptitle(f"Batch Visualization (Epoch {epoch} Step {i})", fontsize=16)
                #     vis_imgs = images_resized.detach().cpu()
                #     vis_pids = pids.detach().cpu().numpy()
                #     vis_doms = domains.detach().cpu().numpy()
                #     for idx in range(B):
                #         # CHW -> HWC
                #         img_vis = vis_imgs[idx].permute(1, 2, 0).numpy()
                #         img_vis = np.clip(img_vis, 0, 1)
                #         plt.subplot(rows, cols, idx + 1)
                #         plt.imshow(img_vis)
                #         plt.title(f"ID:{vis_pids[idx]} D:{vis_doms[idx]}", fontsize=8)
                #         plt.axis('off')
                #     plt.tight_layout()
                #     plt.show()
                #     print("[DEBUG] Visualization Closed. Continue Training...")
                # ==========================================
                clip_input = (images_resized - self.mean_tensor) / self.std_tensor
                image_features = self.clip_model.encode_image(clip_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            optimizer.zero_grad()
            
            text_features = self.trackbook(pids)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            loss = self.criterion(image_features, text_features, pids)
            
            logits = torch.matmul(text_features, image_features.t())
            # plt.imshow(logits.cpu().detach().numpy()) 
            # plt.show()
            
            targets = pids.view(-1, 1).eq(pids.view(1, -1)).float()
            _, indices = torch.max(logits, dim=1)
            
            predicted_pids = pids[indices]
            correct = (predicted_pids == pids).float()
            # Batch Acc
            batch_acc = correct.mean().item() * 100.0
            acc_meter.update(batch_acc, images.size(0))
            loss.backward()
            optimizer.step()
            # Logging
            loss_meter.update(loss.item(), images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            pbar.set_postfix({
                'Loss': f"{loss_meter.avg:.4f}",
                'Acc': f"{acc_meter.avg:.2f}%",
                'Lr': f"{optimizer.param_groups[0]['lr']:.1e}"
            })
            
            if (i + 1) % print_freq == 0:
                if self.summary_writer is not None:
                    step = epoch * len(data_loader) + i
                    # self.summary_writer.add_scalar('Train/Loss', loss.item(), step)
                    self.summary_writer.add_scalar('Loss',loss_meter.avg,step)
                    # self.summary_writer.add_scalar('Train/Acc', batch_acc, step)
                    self.summary_writer.add_scalar('Acc', acc_meter.avg, step)
        return acc_meter.avg
