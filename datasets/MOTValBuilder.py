#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from collections import defaultdict
import os.path as osp

class MOTValBuilder:
    def __init__(self, dataset, skip_k=1):
        
        self.dataset = dataset
        self.skip_k = skip_k
        self.index_pool = defaultdict(list)
        print("Building validation index pool (TXT version)...")
        
        video2idx = defaultdict(list)
        for idx, img_path in enumerate(dataset.img_files):
            
            video_name = '/'.join(img_path.split('/')[:-2])
            video2idx[video_name].append(idx)
        valid_frame_count = 0
        
        for video_name, indices in video2idx.items():
            if len(indices) <= self.skip_k:
                continue
            valid_indices = indices[self.skip_k:]
            for idx in valid_indices:
                label_path = dataset.label_files[idx]
                if not osp.isfile(label_path):
                    continue
                try:
                    raw = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 7)
                except:
                    continue
                if len(raw) == 0:
                    continue
                
                dom = int(raw[0, 0])
                self.index_pool[dom].append(idx)
                valid_frame_count += 1
        self.domains = sorted(list(self.index_pool.keys()))
        print(f"Pool ready. Domains: {self.domains}")
        print(f"Total valid frames (skip_k={self.skip_k}): {valid_frame_count}")
    
    def build(self, num_frames_per_domain=100, seed=42):
        
        np.random.seed(seed)
        selected_indices = []

        for dom in self.domains:
            
            frame_idxs = self.index_pool[dom]
            if not frame_idxs:
                continue

            
            if num_frames_per_domain == -1:
                sampled = frame_idxs  
            else:
               
                replace = len(frame_idxs) < num_frames_per_domain
                sampled = np.random.choice(frame_idxs, num_frames_per_domain, replace=replace)

            selected_indices.extend(sampled.tolist())

        print(f"Val Set Created: {len(selected_indices)} frames.")
        return selected_indices