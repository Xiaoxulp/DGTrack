import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import os.path as osp
from matplotlib.patches import Ellipse, Patch
from fusion.Instance.matching import ious
import torch.nn.functional as F

class BaseValer(object):
    def __init__(self, model, output_dir):
        super(BaseValer, self).__init__()
        self.model = model
        self.output_dir = output_dir
        self._create_vis_dirs()

    def _create_vis_dirs(self):
        
        self.vis_dirs = {
            "domain_text": osp.join(self.output_dir, 'vis_tsne', 'domain_text'),
            "domain_clean": osp.join(self.output_dir, 'vis_tsne', 'domain_clean'),
            "id_grid": osp.join(self.output_dir, 'vis_tsne', 'id_grid'),
            "domain_ellipse": osp.join(self.output_dir, 'vis_tsne', 'domain_ellipse'),
            "sim_matrix": osp.join(self.output_dir, 'vis_sim_matrix')
        }
        for dir_path in self.vis_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def val(self, epoch, val_loader):
        self.model.eval()
        all_feats = []
        all_pids = []
        all_doms = []

        self.sim_epoch_dir = osp.join(self.vis_dirs["sim_matrix"], f"epoch_{epoch}")
        os.makedirs(self.sim_epoch_dir, exist_ok=True)

        print(f"Extracting features for Epoch {epoch} visualization...")
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch [{epoch}]", unit="VAL batch", leave=True)
            for batch_idx, batch_data in enumerate(pbar):
                imgs = batch_data['imgs']
                targets = batch_data['gt_instances']
                track_imgs = imgs[0]
                det_imgs = imgs[1]
                track_targets = targets[0]
                det_targets = targets[1]
                track_ids = track_targets['obj_ids']
                det_ids = det_targets['obj_ids']
                track_domain_id = track_targets['domain_ids']
                det_domain_id = det_targets['domain_ids']
                track_geo = track_targets['boxes_norm'].cuda().unsqueeze(0)
                det_geo = det_targets['boxes_norm'].cuda().unsqueeze(0)

                track_reid = torch.stack(track_imgs).cuda().unsqueeze(0)  # [1, Nt, 2048]
                
                det_reid = torch.stack(det_imgs).cuda().unsqueeze(0)
                
                track_tlbr = track_targets['boxes_tlxy'].cpu().numpy()
                det_tlbr = det_targets['boxes_tlxy'].cpu().numpy()
                iou_np = ious(track_tlbr, det_tlbr)
                iou_matrix = torch.from_numpy(iou_np).float().unsqueeze(0).cuda()

                # GT matrix
                num_tracks = len(track_ids)
                num_dets = len(det_ids)
                gt_matrix = torch.zeros((num_tracks, num_dets), dtype=torch.float32).cuda()
                for t_idx, t_id in enumerate(track_ids):
                    for d_idx, d_id in enumerate(det_ids):
                        if t_id == d_id:
                            gt_matrix[t_idx, d_idx] = 1.0
                gt_matrix = gt_matrix.unsqueeze(0)
              
                reid_sim = torch.matmul(
                    F.normalize(track_reid, dim=-1),
                    F.normalize(det_reid, dim=-1).transpose(-1, -2)
                )
                reid_sim = (reid_sim + 1) / 2
                reid_matrix = reid_sim.cuda()
                # Model forward
                cost_matrix, track_out, det_out = self.model(det_reid=det_reid, det_geo=det_geo, track_reid=track_reid,track_geo=track_geo)
                
                # self._plot_sim_matrix(epoch, batch_idx, cost_matrix, gt_matrix,iou_matrix)
                self._plot_sim_matrix(epoch, batch_idx, cost_matrix, gt_matrix, iou_matrix, reid_matrix)
                feats = track_out.squeeze(0)
                feat_dim = feats.shape[-1]
                feats = feats.reshape(-1, feat_dim)
                pids = track_ids
                doms = track_domain_id
                
                all_feats.append(feats.cpu().numpy())
                all_pids.append(pids.cpu().numpy())
                all_doms.append(doms.cpu().numpy())

        if len(all_feats) == 0:
            print("Warning: No valid features extracted, skip TSNE!")
            return
        feats = np.concatenate(all_feats, axis=0)
        pids = np.concatenate(all_pids, axis=0)
        doms = np.concatenate(all_doms, axis=0)
        self.plot_tsne(feats, pids, doms, epoch)

    def _plot_sim_matrix(self, epoch, batch_idx, sim_matrix, match_matrix, iou_matrix, reid_matrix):
        
        import matplotlib.gridspec as gridspec
        from scipy.stats import gaussian_kde
        from matplotlib.colors import ListedColormap
        import warnings
        warnings.filterwarnings("ignore")

        
        max_color_ratio = 0.76
        
        gamma = 2.0
       
        cmap_original = plt.cm.get_cmap('RdPu')
        x = np.linspace(0, 1, 256)
        x_gamma = x ** gamma 
        colors = cmap_original(max_color_ratio * x_gamma)
        cmap_soft_contrast = ListedColormap(colors)
        

        for sample_idx in range(sim_matrix.shape[0]):
            sim_mat = sim_matrix[sample_idx].cpu().numpy()
            match_mat = match_matrix[sample_idx].cpu().numpy()
            iou_mat = iou_matrix[sample_idx].cpu().numpy()
            reid_mat = reid_matrix[sample_idx].cpu().numpy()

            x = sim_mat
            gt = match_mat
            if x.shape[0] > x.shape[1]:
                x, gt = x.T, gt.T
            has_match = gt.sum(1) > 0
            if has_match.sum() == 0:
                acc = 0.0
            else:
                pred = x.argmax(1)
                target = gt.argmax(1)
                correct = (pred[has_match] == target[has_match]).sum()
                acc = correct / has_match.sum() * 100.0

            fig = plt.figure(figsize=(22, 24))
            gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.25, hspace=0.28)
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
            ax_dist = fig.add_subplot(gs[2, :])
            axes = [ax0, ax1, ax2, ax3]

            for ax in axes:
                ax.set_aspect('equal', adjustable='box')

            def get_optimal_fontsize(mat):
                h, w = mat.shape
                return max(7, 22 - max(h, w))

            def format_text(val, is_gt=False):
                if is_gt:
                    return f"{int(val)}"
                else:
                    return "0" if val < 0.005 else "1" if val > 0.995 else f"{val:.2f}"

            
            # Pred
            im0 = ax0.imshow(sim_mat, cmap=cmap_soft_contrast, vmin=0, vmax=1)
            fs0 = get_optimal_fontsize(sim_mat)
            for i in range(sim_mat.shape[0]):
                for j in range(sim_mat.shape[1]):
                    ax0.text(j, i, format_text(sim_mat[i, j]), ha="center", va="center", color="black", fontsize=fs0)
            ax0.set_title(f"Pred (Ours) | Acc: {acc:.1f}%", fontsize=18, pad=20)
            ax0.set_xlabel("Detection Index", fontsize=14)
            ax0.set_ylabel("Track Index", fontsize=14)
            ax0.tick_params(axis='both', labelsize=fs0)
            fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.06)

            # ReID
            im1 = ax1.imshow(reid_mat, cmap=cmap_soft_contrast, vmin=0, vmax=1)
            fs1 = get_optimal_fontsize(reid_mat)
            for i in range(reid_mat.shape[0]):
                for j in range(reid_mat.shape[1]):
                    ax1.text(j, i, format_text(reid_mat[i, j]), ha="center", va="center", color="black", fontsize=fs1)
            ax1.set_title("ReID Only", fontsize=18, pad=15)
            ax1.set_xlabel("Detection Index", fontsize=14)
            ax1.set_ylabel("Track Index", fontsize=14)
            ax1.tick_params(axis='both', labelsize=fs1)
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.06)

            # IoU
            im2 = ax2.imshow(iou_mat, cmap=cmap_soft_contrast, vmin=0, vmax=1)
            fs2 = get_optimal_fontsize(iou_mat)
            for i in range(iou_mat.shape[0]):
                for j in range(iou_mat.shape[1]):
                    ax2.text(j, i, format_text(iou_mat[i, j]), ha="center", va="center", color="black", fontsize=fs2)
            ax2.set_title("IoU", fontsize=18, pad=15)
            ax2.set_xlabel("Detection Index", fontsize=14)
            ax2.set_ylabel("Track Index", fontsize=14)
            ax2.tick_params(axis='both', labelsize=fs2)
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.06)

            # GT Match
            im3 = ax3.imshow(match_mat, cmap=cmap_soft_contrast, vmin=0, vmax=1)
            fs3 = get_optimal_fontsize(match_mat)
            for i in range(match_mat.shape[0]):
                for j in range(match_mat.shape[1]):
                    ax3.text(j, i, format_text(match_mat[i, j], is_gt=True), ha="center", va="center", color="black",
                            fontsize=fs3)
            ax3.set_title("GT Match", fontsize=18, pad=15)
            ax3.set_xlabel("Detection Index", fontsize=14)
            ax3.set_ylabel("Track Index", fontsize=14)
            ax3.tick_params(axis='both', labelsize=fs3)
            fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.06)

            
            x_scores = sim_mat.flatten()
            r_scores = reid_mat.flatten()
            y_true = match_mat.flatten()

            x_pos = x_scores[y_true == 1]
            x_neg = x_scores[y_true == 0]
            r_pos = r_scores[y_true == 1]
            r_neg = r_scores[y_true == 0]

            x_eval = np.linspace(0, 1, 100)
            if len(x_pos) > 0:
                k_xp = gaussian_kde(x_pos)
                ax_dist.plot(x_eval, k_xp(x_eval), 'darkred', linewidth=3, label='Ours-Positive')
            if len(x_neg) > 0:
                k_xn = gaussian_kde(x_neg)
                ax_dist.plot(x_eval, k_xn(x_eval), 'royalblue', linewidth=3, label='Ours-Negative')
            if len(r_pos) > 0:
                k_rp = gaussian_kde(r_pos)
                ax_dist.plot(x_eval, k_rp(x_eval), 'lightcoral', linestyle='--', linewidth=2, label='ReID-Positive')
            if len(r_neg) > 0:
                k_rn = gaussian_kde(r_neg)
                ax_dist.plot(x_eval, k_rn(x_eval), 'lightblue', linestyle='--', linewidth=2, label='ReID-Negative')

            ax_dist.set_title("Similarity Distribution Comparison", fontsize=18, pad=15)
            ax_dist.set_xlabel("Matching Similarity Score", fontsize=14)
            ax_dist.set_ylabel("Probability Density", fontsize=14)
            ax_dist.legend(fontsize=13, loc='best')
            ax_dist.grid(True, alpha=0.3, linestyle='--')
            ax_dist.set_xlim(0, 1)

            plt.suptitle(f"Batch {batch_idx} | Sample {sample_idx}",
                        fontweight='bold', fontsize=22, ha='center', y=0.97)

            save_path = osp.join(self.sim_epoch_dir, f"batch_{batch_idx}_sample_{sample_idx}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
   

    def plot_tsne(self, feats, pids, doms, epoch):
        
        def plot_domain_ellipse(ax, X, dom_mask, color, n_std=2.0):
            pts = X[dom_mask]
            if len(pts) < 3:
                return
            cov = np.cov(pts, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
            width, height = 2 * n_std * np.sqrt(eigvals)
            center = np.mean(pts, axis=0)
            ell = Ellipse(xy=center, width=width, height=height,
                          angle=angle, edgecolor=color, facecolor='none',
                          linewidth=2.5, alpha=0.8)
            ax.add_patch(ell)

        tsne = TSNE(n_components=2, init='pca', learning_rate='auto',
                    random_state=42, perplexity=min(30, len(feats) - 1))
        X_tsne = tsne.fit_transform(feats)
        X_norm = MinMaxScaler().fit_transform(X_tsne)

        domain_colors = np.array(['#FF0000', '#00FF00', '#0000BF'])

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
        plt.savefig(osp.join(self.vis_dirs["domain_text"], f"epoch_{epoch}_domain_with_text.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 12))
        plt.scatter(X_norm[:, 0], X_norm[:, 1],
                    c=domain_colors[doms], s=60, alpha=0.8, edgecolors='none')
        legend_elements = [Patch(facecolor=c, edgecolor=c, label=f'Domain {i}')
                           for i, c in enumerate(domain_colors)]
        plt.legend(handles=legend_elements, loc='lower right', frameon=True,
                   fancybox=True, shadow=True, fontsize=12)
        plt.title(f"Epoch {epoch}: Domain No Text", fontsize=16)
        plt.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
        plt.xticks([]);
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(osp.join(self.vis_dirs["domain_clean"], f"epoch_{epoch}_domain_no_text.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 12))
        plt.scatter(X_norm[:, 0], X_norm[:, 1],
                    c=domain_colors[doms], s=60, alpha=0.8, edgecolors='none')
        ax = plt.gca()
        for d, color in enumerate(domain_colors):
            plot_domain_ellipse(ax, X_norm, doms == d, color, n_std=2.0)
        plt.title(f"Epoch {epoch}: Domain Ellipse", fontsize=16)
        plt.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
        plt.xticks([]);
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(osp.join(self.vis_dirs["domain_ellipse"], f"epoch_{epoch}_domain_ellipse.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 12))
        unique_pids = np.unique(pids)
        num_ids = len(unique_pids)
        cmap = plt.cm.get_cmap('gist_ncar', num_ids)
        pid2color_idx = {pid: idx for idx, pid in enumerate(unique_pids)}
        color_idx = np.array([pid2color_idx[pid] for pid in pids], dtype=np.int32)
        plt.scatter(X_norm[:, 0], X_norm[:, 1],
                    c=color_idx, cmap=cmap, s=60, alpha=0.8, edgecolors='none')
        for pid in unique_pids:
            mask = (pids == pid)
            cx, cy = np.median(X_norm[mask], axis=0)
            plt.text(cx, cy, str(int(pid)),
                     fontsize=10, weight='bold', color='black',
                     bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))
        plt.title(f"Epoch {epoch}: ID with Text (1 color per ID)", fontsize=16)
        plt.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
        plt.xticks([]);
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(osp.join(self.vis_dirs["id_grid"], f"epoch_{epoch}_id_with_text_unique_colors.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _parse_data(self, inputs):
        raise NotImplementedError("Subclass must implement _parse_data method!")


class FusionValer(BaseValer):
    def __init__(self, model,output_dir):
        super().__init__(model, output_dir)
        self.model.eval()  