import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
from reid.emb_computer import EmbeddingComputer
def make_parser():
    # Initialization
    parser = argparse.ArgumentParser("Extract GT ReID Features for t-SNE")
    parser.add_argument("--data_root", type=str,
                        default="E:/Domain/Cross_mot/UnifyTrack/UnifyTrack/data",help="Root path containing Domain folders")
    parser.add_argument("--weight_path", type=str,default="E:/Domain/Cross_mot/UnifyTrack/UnifyTrack/Train/logs/mot17_trackbook_dnn_self_teacher_180.pth.tar")
    parser.add_argument("--output_path", type=str,default="mot_tsne_data.npz")
    parser.add_argument("--seed", type=float, default=10000)
    return parser


def load_gt_labels(data_root):
    detections = {}
    domain_dirs = glob.glob(os.path.join(data_root, "Domain*"))
    domain_dirs.sort()
    if len(domain_dirs) == 0:
        raise OSError(f"No 'Domain*' directories found in {data_root}")
    print("Loading GT label files...")
    for domain_path in domain_dirs:
        domain_name = os.path.basename(domain_path)
        train_seq_path = os.path.join(domain_path, "labels_with_ids", "train", "*")
        seq_dirs = glob.glob(train_seq_path)
        seq_dirs.sort()
        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            video_key = f"{domain_name}/{seq_name}"
            img1_dir = os.path.join(seq_dir, "img1")
            detections[video_key] = {}
            for frame_file in sorted(os.listdir(img1_dir)):
                frame_id = int(frame_file.split('.')[0])
                frame_path = os.path.join(img1_dir, frame_file)
                try:
                    data = np.loadtxt(frame_path)
                    if data.ndim == 1:
                        data = data[np.newaxis, :]
                    detections[video_key][frame_id] = data[:, 0:7]
                except Exception:
                    detections[video_key][frame_id] = None
            print(f"-> Loaded {video_key}: {len(detections[video_key])} frames")
    return detections


def save_tsne_data(frame_data_dict, vid_name, data_root):
    all_features = []
    all_cids = []
    all_dids = []

    for frame_id, data in frame_data_dict.items():
        if data is None: continue
        if data.shape[1] > 7:
            feats = data[:, 7:]
            cids = data[:, 6].astype(int)
            dids = data[:, 0].astype(int)

            all_features.append(feats)
            all_cids.extend(cids)
            all_dids.extend(dids)
    features = np.concatenate(all_features, axis=0)
    cids = np.array(all_cids)
    dids = np.array(all_dids)

    domain_name, seq_name = vid_name.split('/')
    save_dir = os.path.join(data_root, domain_name, "images", "embeddings_feature")
    os.makedirs(save_dir, exist_ok=True)

    final_save_path = os.path.join(save_dir, f"{seq_name}.npz")

    print(f"  -> Saving {seq_name} ({features.shape[0]} samples) to {final_save_path}")
    np.savez_compressed(
        final_save_path,
        features=features,
        pids=cids,
        dids=dids
    )

if __name__ == '__main__':
    args = make_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(int(args.seed))
    torch.manual_seed(args.seed)
    print(f"Loading model from {args.weight_path}...")
    embedder = EmbeddingComputer(weight_path=args.weight_path)
    detections = load_gt_labels(args.data_root)
    for vid_name in detections.keys():
        
        domain_name, seq_name = vid_name.split('/')
        for frame_id in detections[vid_name].keys():
            # If there is no detection
            if detections[vid_name][frame_id] is None:
                continue
            # Read image
            img_path = os.path.join(args.data_root, domain_name, "images", "train", seq_name, "img1",
                                    "%06d.jpg" % frame_id)
            img = cv2.imread(img_path)
            detection = detections[vid_name][frame_id]
            # Get features
            if detection is not None:
                embedding = embedder.compute_embedding(img, detection[:, 2:6])
                detections[vid_name][frame_id] = np.concatenate([detection, embedding], axis=1)
        print(f"Finished extracting {vid_name}")
        save_tsne_data(detections[vid_name], vid_name, args.data_root)
        detections[vid_name] = None
    print("-------------end----------------")
