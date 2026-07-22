'''
@Date: 2025/12/28
@Author: <ChenRuXu>
@Email: <EMAIL>
验证集的可视化过程。通过t-sne将结果来进行可视化
后面可以在补充grad_gam的可视化分析过程这个感觉还是单独的去写比较好
'''
import time
import sys
import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors
import os.path as osp
from matplotlib.patches import Ellipse
import cv2
import math
from vis.gradgam.utils import GradCAM, show_cam_on_image
from matplotlib.colors import ListedColormap
from matplotlib import patheffects

class BaseValer(object):
    def __init__(self, model,output_dir):
        super(BaseValer, self).__init__()
        self.model = model
        self.output_dir = output_dir
        self.normlizer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def val(self, epoch, val_loader):
        """
            执行验证流程：提取特征 -> 绘制 t-SNE
        """
        self.model.eval()

        all_feats = []
        all_pids = []
        all_doms = []
        print(f"Extracting features for Epoch {epoch} visualization...")
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch [{epoch}]", unit="VAL batch", leave=True)
            for i, inputs in enumerate(pbar):
                imgs, pids, doms = self._parse_data(inputs)
                in_size = inputs[0].size()
                for j in range(in_size[0]):
                    imgs[0][j, :, :, :] = self.normlizer(imgs[0][j, :, :, :])
                outputs = self.model(imgs, training=False)
                # 我们通常用 BN 后的特征做检索/可视化
                if isinstance(outputs, (tuple, list)):
                    feat = outputs[1]
                else:
                    feat = outputs
                # 收集数据 (转回 CPU numpy)
                all_feats.append(feat.cpu().numpy())
                all_pids.append(pids.cpu().numpy())
                all_doms.append(doms.cpu().numpy())
            # 拼接
            feats = np.concatenate(all_feats, axis=0)
            pids = np.concatenate(all_pids, axis=0)
            doms = np.concatenate(all_doms, axis=0)
        # 绘制 t-SNE
        self.plot_tsne(feats, pids, doms, epoch)

    # =========================================================
    #  功能 1: 绘制训练过程中的T-sne可视化图 (拓展功能)
    # =========================================================
    def plot_tsne(self, feats, pids, doms, epoch):
        print(f"Running t-SNE on ALL {len(feats)} samples...")

        # ----------  工具函数：画置信椭圆 ----------

        def plot_domain_ellipse(ax, X, dom_mask, color, n_std=2.0):
            pts = X[dom_mask]
            if len(pts) < 3:
                return
            cov = np.cov(pts, rowvar=False) # 计算 2×2 协方差
            # 特征值分解
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            # 椭圆参数
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * n_std * np.sqrt(eigvals)
            # 中心
            center = np.mean(pts, axis=0)
            # 画椭圆
            ell = Ellipse(xy=center, width=width, height=height,
                          angle=angle, edgecolor=color, facecolor='none',
                          linewidth=2.5, alpha=0.8)
            ax.add_patch(ell)
        # ------------------------------------------------------

        # t-SNE 降维
        tsne = TSNE(n_components=2, init='pca', learning_rate='auto',
                    random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(feats)
        X_norm = MinMaxScaler().fit_transform(X_tsne)

        # 四文件夹 单独的存储
        vis_domain_text_dir = osp.join(self.output_dir, 'vis_domain_text')  # 原：带数字
        vis_domain_clean_dir = osp.join(self.output_dir, 'vis_domain_clean')  # 原：无数字
        vis_id_grid_dir = osp.join(self.output_dir, 'vis_id_grid')  # 原：ID 无数字
        vis_domain_ellipse_dir = osp.join(self.output_dir, 'vis_domain_ellipse')  # 新：椭圆
        os.makedirs(vis_domain_text_dir, exist_ok=True)
        os.makedirs(vis_domain_clean_dir, exist_ok=True)
        os.makedirs(vis_id_grid_dir, exist_ok=True)
        os.makedirs(vis_domain_ellipse_dir, exist_ok=True)

        domain_colors = np.array(['#FF0000', '#00FF00', '#0000BF'])  # 红绿蓝
        cmap_id = plt.cm.get_cmap('gist_ncar', 256)

        # ----------------------------------------------------------
        # 图 1 原：域三色 + 网格 + 带数字
        # ----------------------------------------------------------
        plt.figure(figsize=(12, 12))
        plt.scatter(X_norm[:, 0], X_norm[:, 1],
                    c=domain_colors[doms], s=60, alpha=0.8, edgecolors='none')
        unique_pids = np.unique(pids)
        for pid in unique_pids:
            mask = pids == pid
            cx, cy = np.median(X_norm[mask], axis=0)
            plt.text(cx, cy, str(int(pid)),
                     fontsize=12, weight='bold', color='black',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))
        plt.title(f"Epoch {epoch}: Domain with Text", fontsize=16)
        plt.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
        plt.xticks([]);
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(osp.join(vis_domain_text_dir, f"epoch_{epoch}_domain_with_text.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ----------------------------------------------------------
        # 图 2 原：域三色 + 网格 + 无数字
        # ----------------------------------------------------------
        plt.figure(figsize=(12, 12))
        plt.scatter(X_norm[:, 0], X_norm[:, 1],
                    c=domain_colors[doms], s=60, alpha=0.8, edgecolors='none')
        # 图 2 加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, edgecolor=c, label=f'Domain {i}')
                           for i, c in enumerate(domain_colors)]
        plt.legend(handles=legend_elements, loc='lower right', frameon=True,
                   fancybox=True, shadow=True, fontsize=12)
        # ===========================================
        plt.title(f"Epoch {epoch}: Domain No Text", fontsize=16)
        plt.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
        plt.xticks([]);
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(osp.join(vis_domain_clean_dir, f"epoch_{epoch}_domain_no_text.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ----------------------------------------------------------
        # 图 3 新增：域三色 + 椭圆 + 网格 → 单独文件夹
        # ----------------------------------------------------------
        plt.figure(figsize=(12, 12))
        plt.scatter(X_norm[:, 0], X_norm[:, 1],
                    c=domain_colors[doms], s=60, alpha=0.8, edgecolors='none')

        # 画置信椭圆
        ax = plt.gca()
        for d, color in enumerate(domain_colors):
            plot_domain_ellipse(ax, X_norm, doms == d, color, n_std=2.0)

        plt.title(f"Epoch {epoch}: Domain Ellipse", fontsize=16)
        plt.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
        plt.xticks([]);
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(osp.join(vis_domain_ellipse_dir, f"epoch_{epoch}_domain_ellipse.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ----------------------------------------------------------
        # 图 4：ID 彩图 + 数字（每个 pid 一个固定颜色），不加图例
        # ----------------------------------------------------------
        plt.figure(figsize=(12, 12))

        unique_pids = np.unique(pids)
        num_ids = len(unique_pids)

        # 1) 给每个 pid 分配一个颜色（可支持任意数量 pid）
        cmap = plt.cm.get_cmap('gist_ncar', num_ids)  # 离散化成 num_ids 种颜色
        pid2color_idx = {pid: idx for idx, pid in enumerate(unique_pids)}
        color_idx = np.array([pid2color_idx[pid] for pid in pids], dtype=np.int32)

        plt.scatter(
            X_norm[:, 0], X_norm[:, 1],
            c=color_idx,
            cmap=cmap,
            s=60, alpha=0.8, edgecolors='none'
        )

        # 2) 像图 1 一样写数字（每个 pid 在中位点标注）
        for pid in unique_pids:
            mask = (pids == pid)
            cx, cy = np.median(X_norm[mask], axis=0)
            plt.text(
                cx, cy, str(int(pid)),
                fontsize=10, weight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7)
            )

        plt.title(f"Epoch {epoch}: ID with Text (1 color per ID)", fontsize=16)
        plt.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
        plt.xticks([]);
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(
            osp.join(vis_id_grid_dir, f"epoch_{epoch}_id_with_text_unique_colors.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    # =========================================================
    #  功能 2: Grad-CAM 可视化 (拓展功能)
    # =========================================================
    def run_cam(self, epoch, val_loader):
        '''
        运行 Grad-CAM：遍历验证集，每个 Batch 生成一张 Grid 大图
        :param epoch:
        :param val_loader:
        :return:
        '''
        self.model.eval()
        print(f"Generating Grad-CAM Grid for Epoch {epoch}...")
        save_dir = osp.join(self.output_dir, 'vis_gradcam', f'epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        # 兼容 DataParallel
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        target_layers = model_to_use.backbone.layer4
        cam = GradCAM(model=model_to_use, target_layers=target_layers, use_cuda=True)
        # 字体参数
        TEXT_HEIGHT = 50
        TITLE_HEIGHT = 40  # 顶部总标题高度
        FONT_SCALE = 0.5
        FONT_THICKNESS = 1
        GRID_GAP = 10
        MICRO_BATCH_SIZE = 16 # 设定一个安全的微批次大小

        # 遍历所有 Batch
        pbar = tqdm(val_loader, desc=f"Epoch [{epoch}]", unit="Grad_gam batch", leave=True)

        for batch_idx, inputs in enumerate(pbar):
            # 解析数据
            ori_inputs, pids, doms = self._parse_data(inputs)
            if isinstance(ori_inputs, list):
                imgs = ori_inputs[0] # imgs: [Batch, C, H, W]
            else:
                imgs = ori_inputs

            # --- 分批处理 (Micro-Batching) ---
            total_samples = imgs.size(0)
            all_grayscale_cams = []
            all_pred_pids = []

            # 循环切片
            for i in range(0, total_samples, MICRO_BATCH_SIZE):
                # 1. 取出小批次
                sub_imgs = imgs[i: i + MICRO_BATCH_SIZE]

                # 2. 归一化
                sub_imgs_norm = sub_imgs.clone()
                for j in range(len(sub_imgs_norm)):
                    sub_imgs_norm[j] = self.normlizer(sub_imgs_norm[j])

                # 3. 跑 Grad-CAM
                # 这样每次只占 16 张图的显存
                sub_cams, sub_preds = cam(input_tensor=sub_imgs_norm, target_category=None)

                # 4. 存结果 (转 numpy 存 CPU，释放 GPU)
                all_grayscale_cams.append(sub_cams)
                all_pred_pids.extend(sub_preds)

                # 5. 清理显存
                del sub_imgs_norm
                torch.cuda.empty_cache()

            # 拼接结果 (拼回一个大 Batch)
            grayscale_cams = np.concatenate(all_grayscale_cams, axis=0)
            pred_pids = all_pred_pids  # list
            # 计算 Batch 准确率
            correct_count = 0
            for i in range(len(imgs)):
                if int(pred_pids[i]) == int(pids[i].item()):
                    correct_count += 1
            batch_acc = correct_count / len(imgs) * 100

            # 准备画布
            num_imgs = imgs.size(0)
            cols = 8
            rows = math.ceil(num_imgs / cols)
            H, W = imgs.size(2), imgs.size(3)
            cell_w = W * 2
            cell_h = H + TEXT_HEIGHT

            canvas_w = cols * cell_w + (cols + 1) * GRID_GAP
            # 高度加上 TITLE_HEIGHT
            canvas_h = rows * cell_h + (rows + 1) * GRID_GAP + TITLE_HEIGHT

            grid_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) + 255

            # 写总标题
            title_text = f"Epoch {epoch} Batch {batch_idx} | Accuracy: {batch_acc:.2f}%"
            cv2.putText(grid_img, title_text, (20, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            for i in range(num_imgs):
                img_tensor = imgs[i].cpu()
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = np.clip(img_np, 0, 1)

                heatmap = grayscale_cams[i, :]
                vis_cam = show_cam_on_image(img_np, heatmap, use_rgb=True)

                vis_orig = (img_np * 255).astype(np.uint8)
                img_combined = np.hstack((vis_orig, vis_cam))

                # 文字逻辑
                gt_pid = pids[i].item()
                pr_pid = int(pred_pids[i])
                dom_val = doms[i].item()
                is_correct = (gt_pid == pr_pid)
                text_color = (0, 200, 0) if is_correct else (255, 0, 0)

                header = np.zeros((TEXT_HEIGHT, cell_w, 3), dtype=np.uint8) + 255
                cv2.putText(header, f"GT:{gt_pid} D:{dom_val}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
                cv2.putText(header, f"Pr:{pr_pid} {'OK' if is_correct else 'ERR'}", (5, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_color, FONT_THICKNESS + 1)
                final_cell = np.vstack((header, img_combined))
                if not is_correct:
                    cv2.rectangle(final_cell, (0, 0), (cell_w - 1, cell_h - 1), (255, 0, 0), 2)
                r, c = i // cols, i % cols
                x_start = c * cell_w + (c + 1) * GRID_GAP
                # y_start 要加上 TITLE_HEIGHT 偏移
                y_start = r * cell_h + (r + 1) * GRID_GAP + TITLE_HEIGHT
                grid_img[y_start:y_start + cell_h, x_start:x_start + cell_w, :] = final_cell
                # 保存
            cv2.imwrite(osp.join(save_dir, f"batch_{batch_idx}.jpg"), cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
    def _parse_data(self, inputs):
        raise NotImplementedError


class ReidValer(BaseValer):

    def _parse_data(self, inputs):
        imgs,pids,domain_id = inputs
        # 修改Variable使用 non_blocking=True 加速
        inputs = [imgs.cuda(non_blocking=True)]
        targets = pids.cuda(non_blocking=True)
        domain_id = domain_id.cuda(non_blocking=True)
        return inputs, targets,domain_id
