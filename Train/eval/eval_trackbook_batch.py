import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from clip import clip
from torchvision import transforms


from reid.models.TrackBook import LearnableTrackBook
from datasets.reiddataset import ReidDataset


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TrackBook Performance')

   
    parser.add_argument('--json_path', type=str,
                        default='E:/Domain/Cross_mot/UnifyTrack/UnifyTrack/data/reid_meta.json',
                        help='Path to the dataset json file')
    parser.add_argument('--ckpt_path', type=str,
                        default='E:/Domain/Cross_mot/UnifyTrack/UnifyTrack/Train/logs_trackbook/trackbook_ep400_400.pth.tar',
                        help='Path to the trained TrackBook checkpoint')
    parser.add_argument('--clip_model_path', type=str,
                        default='E:/Domain/Cross_mot/UnifyTrack/UnifyTrack/reid/weight/ViT-B-32.pt',
                        help='Path to the pre-trained CLIP model')

    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of samples to visualize (must <= total IDs)')
    parser.add_argument('--num_pids', type=int, default=405,
                        help='Total number of IDs the TrackBook was trained on')
    parser.add_argument('--n_ctx', type=int, default=6,
                        help='Number of learnable tokens (must match training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_vis', action='store_true',
                        help='If set, do not show visualization windows (for server run)')
    parser.add_argument('--save_vis', action='store_true',
                        help='If set, save visualization to disk instead of showing')
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                        help='Directory to save visualization results')

    return parser.parse_args()


def main():
    args = parse_args()
    if args.save_vis and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- TrackBook Eval (Device: {device}) ---")


    dataset = ReidDataset(args.json_path)
    print(f"Dataset loaded. Total Samples: {len(dataset)}")


    pid_pool = {}
    for idx, item in enumerate(dataset.raw_data):
        pid = int(item['class_id'])
        if pid < args.num_pids:
            if pid not in pid_pool: pid_pool[pid] = []
            pid_pool[pid].append(idx)

    available_pids = list(pid_pool.keys())
    actual_bs = min(args.batch_size, len(available_pids))
    print(f"Sampling {actual_bs} unique IDs from {len(available_pids)} available...")

    selected_pids = random.sample(available_pids, actual_bs)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD)
    ])

    vis_crops = []
    input_tensors = []
    gt_pids = []

    for pid in selected_pids:
        idx = random.choice(pid_pool[pid])
        item = dataset.raw_data[idx]

        img_path = item['img_path']
        bbox = item['bbox']

        try:
            img = Image.open(img_path).convert('RGB')
            x, y, w, h = map(int, bbox)
            w = min(w, img.width - x)
            h = min(h, img.height - y)
            crop = img.crop((x, y, x + w, y + h))

            vis_crops.append(crop)
            input_tensors.append(val_transform(crop))
            gt_pids.append(pid)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

    if len(input_tensors) == 0:
        print("Error: No valid samples found.")
        return

    input_batch = torch.stack(input_tensors).to(device)
    target_ids = torch.tensor(gt_pids).to(device)

    print("Loading Models...")
    clip_model, _ = clip.load(args.clip_model_path, device=device)
    clip_model.float()


    trackbook = LearnableTrackBook(clip_model, n_ctx=args.n_ctx, n_ids=args.num_pids)

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    print(f"Loading weights from {args.ckpt_path}...")
    checkpoint = torch.load(args.ckpt_path)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    trackbook.load_state_dict(state_dict)

    trackbook.to(device)
    trackbook.eval()

    print("Running Inference...")
    with torch.no_grad():
        img_feats = clip_model.encode_image(input_batch)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        text_feats = trackbook(target_ids)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        sim_matrix = img_feats @ text_feats.t()

    scores, preds = sim_matrix.max(dim=1)

    labels = torch.arange(len(gt_pids)).to(device)

    correct = (preds == labels).sum().item()
    acc = correct / len(gt_pids) * 100
    print(f"\n>>> Retrieval Accuracy (Top-1): {acc:.2f}% ({correct}/{len(gt_pids)})")


    if not args.no_vis:
        plt.figure(figsize=(10, 8))
        sim_np = sim_matrix.cpu().numpy()
        plt.imshow(sim_np, cmap='viridis')
        plt.colorbar(label='Similarity')
        plt.title(f"Similarity Matrix (Acc: {acc:.1f}%)")
        plt.xlabel("Text Prompts (ID)")
        plt.ylabel("Query Images (ID)")
        plt.tight_layout()

        if args.save_vis:
            plt.savefig(os.path.join(args.save_dir, "sim_matrix.png"))
            print(f"Saved matrix to {args.save_dir}")
        else:
            plt.show()

        cols = 8
        rows = (len(gt_pids) + cols - 1) // cols
        plt.figure(figsize=(16, 2.5 * rows))
        plt.suptitle("Retrieval Visualisation", fontsize=16)

        for i in range(len(gt_pids)):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(vis_crops[i])
            pred_idx = preds[i].item()
            pred_pid = gt_pids[pred_idx]
            true_pid = gt_pids[i]
            score = scores[i].item()
            color = 'green' if pred_idx == i else 'red'
            plt.title(f"GT:{true_pid}\nPr:{pred_pid}\n{score:.2f}", color=color, fontsize=10, fontweight='bold')
            plt.axis('off')
        plt.tight_layout()
        if args.save_vis:
            plt.savefig(os.path.join(args.save_dir, "retrieval_res.png"))
            print(f"Saved retrieval result to {args.save_dir}")
        else:
            plt.show()

if __name__ == "__main__":
    main()