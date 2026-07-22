import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import pickle
import random
import argparse
import numpy as np
from reid.emb_computer import EmbeddingComputer
import matplotlib.pyplot as plt

def make_parser():
    parser = argparse.ArgumentParser("Train reid")
    parser.add_argument("--data_path", type=str,default="/home/st3/DGTrack/data/Domain3/images/train/")
    parser.add_argument("--label_path", type=str,default="/home/st3/DGTrack/data/Domain3/labels_with_ids/train/")
    parser.add_argument("--output_path", type=str,default="/home/st3/DGTrack/data/Domain3/gt_embeddings/train/gt_embeddings.pkl")
    parser.add_argument("--weight_path", type=str,default="/home/st3/DGTrack/Train/logs/UnifyTrack5T_best.pth.tar")
    parser.add_argument("--seed", type=float, default=10000)
    return parser

def load_label(label_file):
    if not os.path.isfile(label_file):
        return None
    try:
        raw = np.loadtxt(label_file, dtype=np.float32).reshape(-1, 7)
    except:
        return None
    if len(raw) == 0:
        return None
    xywh = raw[:, 2:6]
    tlbr = xywh.copy()
    tlbr[:, 2] = xywh[:, 0] + xywh[:, 2]
    tlbr[:, 3] = xywh[:, 1] + xywh[:, 3]
    return raw, tlbr

if __name__ == '__main__':
    args = make_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    embedder = EmbeddingComputer(weight_path=args.weight_path)
    detections = {}
    videos = os.listdir(args.data_path)
    for vid_name in videos:
        detections[vid_name] = {}
        img_dir = os.path.join(args.data_path, vid_name, 'img1')
        label_dir = os.path.join(args.label_path, vid_name, 'img1')
        if not os.path.isdir(img_dir):
            continue
        img_files = sorted(os.listdir(img_dir))
        for fname in img_files:
            if not fname.endswith('.jpg'):
                continue
            frame_id = int(os.path.splitext(fname)[0])
            img_path = os.path.join(img_dir, fname)
            label_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))

            label_data = load_label(label_path)
            if label_data is None:
                detections[vid_name][frame_id] = None
                continue
            raw, boxes = label_data
            if 'MOT' in args.data_path:
                img = cv2.imread(args.data_path + vid_name + '/img1/%06d.jpg' % frame_id)
            else:
                img = cv2.imread(args.data_path + vid_name + '/img1/%06d.jpg' % frame_id)
            detection = boxes.astype(np.float32)
            if detection is not None:
                embedding = embedder.compute_embedding(img, detection[:, :4])
                detections[vid_name][frame_id] = np.concatenate([raw, embedding], axis=1)
            else:
                detections[vid_name][frame_id] = None
            print(vid_name, frame_id, flush=True)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as handle:
        pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved to {args.output_path}")