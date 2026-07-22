#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from collections import defaultdict
import pickle

class MOTDataset(Dataset):
    def __init__(self, args, data_txt_path: str, seqs_folder):
        self.args = args
        self.data_txt_path = data_txt_path
        self.seqs_folder = seqs_folder
        self.video_dict = {}
        self.ema_alpha  = getattr(args, 'ema_alpha', 0.95)
        self.max_history = getattr(args, 'max_history', 60) 
        
        with open(data_txt_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [osp.join(seqs_folder, x.strip()) for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))
        self.label_files = [(x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]
        
        self.item_num = len(self.img_files) - 1
        
        self.reid_data = {}
        for pkl_path in args.reid_pkl_paths:
            print(f"  -> Loading {pkl_path}")
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            for vid in data:
                if vid not in self.reid_data:
                    self.reid_data[vid] = {}
                self.reid_data[vid].update(data[vid])
                print(f"{vid} loaded")
        print(f"ReID loaded! Total videos: {len(self.reid_data)}")
        self._register_videos()
        self._compute_domain_stats()
    def _register_videos(self):
        for label_name in self.label_files:
            video_name = '/'.join(label_name.split('/')[:-1])
            if video_name not in self.video_dict:
                print("register {}-th video: {}".format(
                    len(self.video_dict) + 1, video_name))
                self.video_dict[video_name] = len(self.video_dict)

    def _compute_domain_stats(self):
        
        print("=> Calculating Motion Dataset Domain Statistics (Memory)...")
        domain_frame_count = defaultdict(int)
        domain_track_count = defaultdict(int)
        
        for label_path in self.label_files:
            if not osp.isfile(label_path):
                continue
            try:
                raw = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)
            except:
                continue
            if len(raw) == 0:
                continue
           
            domain_ids = raw[:, 0].astype(int)
            dom = int(domain_ids[0])
           
            domain_frame_count[dom] += 1
           
            domain_track_count[dom] += len(domain_ids)
        
        self.__class__.domain_frame_count = dict(domain_frame_count)
        self.__class__.domain_track_count = dict(domain_track_count)
        
        print("  Motion Dataset Domain Statistics:")
        print("  ---------------------------------------- ")
        for d in sorted(domain_frame_count.keys()):
            n_frames = domain_frame_count[d]
            n_tracks = domain_track_count[d]
            print(f"    Domain {d}: {n_frames} Frames, {n_tracks} Tracks")
        print("  ---------------------------------------- ")

    def _pre_single_frame(self, img_path, label_path, image_idx):
        
        
        img = Image.open(img_path)
        targets = {}
        W, H = img.size
        img.close()  
        assert W > 0 and H > 0, f"invalid image {img_path} with shape {W} {H}"
        if not osp.isfile(label_path):
            raise ValueError(f'invalid label path: {label_path}')
        raw = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)
        
        domain_ids = raw[:, 0].astype(int)
        pids = raw[:, 1].astype(int)
        xywh = raw[:, 2:6].astype(int)
        class_ids = raw[:, 6].astype(int)
        
        tlxy = xywh.copy()
        tlxy[:, 2] = xywh[:, 0] + xywh[:, 2]  # x2
        tlxy[:, 3] = xywh[:, 1] + xywh[:, 3]  # y2
        
        tlxy[:, 0] = np.clip(tlxy[:, 0], 0, W)
        tlxy[:, 1] = np.clip(tlxy[:, 1], 0, H)
        tlxy[:, 2] = np.clip(tlxy[:, 2], 0, W)
        tlxy[:, 3] = np.clip(tlxy[:, 3], 0, H)
        
        cx = xywh[:, 0] + xywh[:, 2] / 2
        cy = xywh[:, 1] + xywh[:, 3] / 2
        norm_xywh = np.stack([cx / W, cy / H, xywh[:, 2] / W, xywh[:, 3] / H], axis=1)
        
        targets['image_id'] = torch.as_tensor(image_idx)  
        targets['boxes_tlxy'] = torch.as_tensor(tlxy, dtype=torch.float32)
        targets['boxes_norm'] = torch.as_tensor(norm_xywh, dtype=torch.float32)
        targets['obj_ids'] = torch.as_tensor(pids, dtype=torch.int64)
        targets['domain_ids'] = torch.as_tensor(domain_ids, dtype=torch.int64)
        targets['class_ids'] = torch.as_tensor(class_ids, dtype=torch.int64)
        return targets

    def _build_reid_features(self, vid, frame_id, targets, is_track):
        
        ids = targets["obj_ids"].numpy()
        
        if vid not in self.reid_data or frame_id not in self.reid_data[vid]:
            emb_dim = 2048
            return [torch.zeros(emb_dim).float() for _ in ids]
        
        frame_data = self.reid_data[vid][frame_id]
        
        frame_ids = frame_data[:, 1].astype(int)
        if len(ids) != len(frame_ids) or not np.array_equal(ids, frame_ids): 
            print(" ID mismatch detected!")
            print(f"Video: {vid}, Frame: {frame_id}")
            print(f"GT ids     : {ids}")
            print(f"ReID ids   : {frame_ids}")
            raise ValueError("GT ids and ReID ids are not aligned!")
        
        frame_emb = frame_data[:, 7:].astype(np.float32)
        emb_dim = frame_emb.shape[1]
        outputs = []
        
        for pid in ids:
            
            idx_arr = np.where(frame_ids == pid)[0]
            if len(idx_arr) == 0:
                outputs.append(torch.zeros(emb_dim).float())
                
                continue
           
            cur_emb = frame_emb[idx_arr[0]].copy()
            
            if not is_track:
                outputs.append(torch.from_numpy(cur_emb).float())
                continue
            
            start_f = max(1, frame_id - self.max_history)
            
            hist_embs = []
            for f in range(start_f, frame_id): 
                if f not in self.reid_data[vid]:
                   
                    continue
                prev_data = self.reid_data[vid][f]
                prev_ids = prev_data[:, 1].astype(int)
                idx_prev = np.where(prev_ids == pid)[0]
                if len(idx_prev) == 0:
                    continue
                
                hist_embs.append(prev_data[idx_prev[0], 7:].astype(np.float32).copy())
            
            if len(hist_embs) == 0:
                outputs.append(torch.from_numpy(cur_emb).float())
                continue
            
            ema = hist_embs[0]
            
            for t in range(1, len(hist_embs)):
                ema = self.ema_alpha * ema + (1 - self.ema_alpha) * hist_embs[t]
            
            ema = self.ema_alpha * ema + (1 - self.ema_alpha) * cur_emb
            
            norm = np.linalg.norm(ema)
            if norm > 0:
                ema = ema / norm
            outputs.append(torch.from_numpy(ema).float())
        return outputs

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        '''
        
        track_image_path = self.img_files[index]
        det_image_path   = self.img_files[index + 1]
        if track_image_path.split('/')[-3] != det_image_path.split('/')[-3]:
            return self.__getitem__(index - 1)  
        track_label = self.label_files[index]
        det_label   = self.label_files[index + 1]
        track_frame = int(os.path.splitext(os.path.basename(track_image_path))[0])
        det_frame   = int(os.path.splitext(os.path.basename(det_image_path))[0])
        vid = track_image_path.split('/')[-3] 
        
        track_targets = self._pre_single_frame(track_image_path, track_label, track_frame)
        det_targets   = self._pre_single_frame(det_image_path, det_label, det_frame)
        
        track_feats = self._build_reid_features(vid, track_frame, track_targets,is_track=True)
        det_feats   = self._build_reid_features(vid, det_frame,   det_targets,is_track=False)
        data = {}
        data.update({
            "imgs": [
                track_feats,
                det_feats
            ],
            "gt_instances": [
                track_targets,
                det_targets
            ]
        })
        return data
    def __len__(self):
        return self.item_num