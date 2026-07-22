'''
@Date: 2026/1/2
@Author: <ChenRuXu>
@Description: TrackBook 专用验证器 (Grid Mosaic + Matrix + Micro-batch)
'''
import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

class TrackBookValer(object):
    def __init__(self, clip_model, trackbook, output_dir, device, clip_mean, clip_std):
        self.clip_model = clip_model
        self.trackbook = trackbook
        self.output_dir = output_dir
        self.device = device

        # 预处理 Tensor
        self.mean_tensor = torch.tensor(clip_mean).view(1, 3, 1, 1).to(device)
        self.std_tensor = torch.tensor(clip_std).view(1, 3, 1, 1).to(device)

    def val(self, epoch, dataloader):
        """
        全量遍历验证集，每个 Batch 生成一张可视化图和矩阵图
        """
        self.trackbook.eval()
        save_dir = osp.join(self.output_dir, 'vis_trackbook', f'epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)

        print(f"Visualizing TrackBook for Epoch {epoch}...")

        # 配置参数
        TEXT_HEIGHT = 50
        TITLE_HEIGHT = 40
        FONT_SCALE = 0.5
        FONT_THICKNESS = 1
        GRID_GAP = 10

        pbar = tqdm(dataloader, desc=f"Val Epoch [{epoch}]", unit="bt", leave=True)

        for batch_idx, (images, pids, domains) in enumerate(pbar):
            # images: [B, 3, 384, 128]
            images = images.to(self.device).float()
            pids = pids.to(self.device)

            # ==========================================
            # 1. 整个 Batch 直接推理 (Simple & Fast)
            # ==========================================
            with torch.no_grad():
                # Image
                imgs_resized = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)
                clip_in = (imgs_resized - self.mean_tensor) / self.std_tensor
                img_feats = self.clip_model.encode_image(clip_in)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

                # Text
                text_feats = self.trackbook(pids)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                # Similarity Matrix [B, B]
                sim_matrix = img_feats @ text_feats.t()

            # ==========================================
            # 2. 计算准确率 & 预测结果
            # ==========================================
            # 每一行(Image)找最大的列(Text)
            scores, preds = sim_matrix.max(dim=1)

            # 找到预测对应的 PID
            pred_pids = pids[preds]
            correct_count = (pred_pids == pids).sum().item()
            total_samples = images.size(0)
            batch_acc = correct_count / total_samples * 100.0

            # ==========================================
            # 3. 可视化 A: 相似度矩阵
            # ==========================================
            plt.figure(figsize=(10, 8))
            # 转 numpy
            sim_np = sim_matrix.cpu().numpy()

            plt.imshow(sim_np, cmap='viridis')
            plt.colorbar(label='Sim')
            plt.title(f"Batch {batch_idx} Matrix (Acc: {batch_acc:.1f}%)")
            plt.xlabel("Text Prompts (Index in Batch)")
            plt.ylabel("Images (Index in Batch)")
            plt.tight_layout()
            plt.savefig(osp.join(save_dir, f"batch_{batch_idx}_matrix.png"))
            plt.close()

            # ==========================================
            # 4. 可视化 B: 检索大图 (Grid Mosaic)
            # ==========================================
            num_imgs = total_samples
            cols = 8
            rows = math.ceil(num_imgs / cols)

            # 单图尺寸
            H, W = images.size(2), images.size(3)
            cell_w = W
            cell_h = H + TEXT_HEIGHT

            canvas_w = cols * cell_w + (cols + 1) * GRID_GAP
            canvas_h = rows * cell_h + (rows + 1) * GRID_GAP + TITLE_HEIGHT

            grid_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 255

            # 标题
            title_text = f"Epoch {epoch} Batch {batch_idx} | Acc: {batch_acc:.2f}%"
            cv2.putText(grid_img, title_text, (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            for i in range(num_imgs):
                # 图片
                img_tensor = images[i].cpu()
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)
                vis_orig = (img_np * 255).astype(np.uint8)

                # 文字
                gt_pid = pids[i].item()
                pr_pid = pred_pids[i].item()  # 预测出的 ID
                dom_val = domains[i].item()
                score = scores[i].item()

                is_correct = (gt_pid == pr_pid)
                text_color = (0, 200, 0) if is_correct else (255, 0, 0)  # Green / Red

                header = np.zeros((TEXT_HEIGHT, cell_w, 3), dtype=np.uint8) + 255
                cv2.putText(header, f"GT:{gt_pid} D:{dom_val}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
                cv2.putText(header, f"Pr:{pr_pid} {score:.2f}", (5, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_color, FONT_THICKNESS)

                final_cell = np.vstack((header, vis_orig))

                if not is_correct:
                    cv2.rectangle(final_cell, (0, 0), (cell_w - 1, cell_h - 1), (255, 0, 0), 2)

                r, c = i // cols, i % cols
                x_start = c * cell_w + (c + 1) * GRID_GAP
                y_start = r * cell_h + (r + 1) * GRID_GAP + TITLE_HEIGHT

                grid_img[y_start:y_start + cell_h, x_start:x_start + cell_w, :] = final_cell

            # 保存
            cv2.imwrite(osp.join(save_dir, f"batch_{batch_idx}.jpg"), cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))

        print(f"TrackBook Vis finished. Saved {batch_idx + 1} batches to {save_dir}")