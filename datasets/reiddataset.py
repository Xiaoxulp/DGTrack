import os.path as osp
import json
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

class ReidDataset(Dataset):
    def __init__(self, json_path: str):
        
        self.json_path = json_path
       
        if not osp.exists(self.json_path):
            raise ValueError(f"JSON file not found: {self.json_path}")
        print(f"Loading ReID Dataset from: {self.json_path}")
        
        with open(self.json_path, 'r') as f:
            self.raw_data = json.load(f)

        
        self.meta_data = []
        
        self.num_pids = 0
        self.domain_stats = {} 
        self.domain_pids = {} 

        
        for item in self.raw_data:
            img_path = item['img_path']
            pid = int(item['class_id'])
            domain_id = int(item['domain'])
            
            self.meta_data.append((img_path, pid, domain_id))
            
            if pid >= self.num_pids:
                self.num_pids = pid + 1
            
            self.domain_stats[domain_id] = self.domain_stats.get(domain_id, 0) + 1
            
            if domain_id not in self.domain_pids:
                self.domain_pids[domain_id] = set()
            self.domain_pids[domain_id].add(pid)

        print("=> JointReIDDataset Loaded Successfully")
        print(f"  Total Images: {len(self.meta_data)}")
        print(f"  Total IDs:    {self.num_pids}")
        print("  ---------------------------------------- ")
        print("  Domain Statistics:")
        
        sorted_domains = sorted(self.domain_stats.keys())
        for d in sorted_domains:
            n_imgs = self.domain_stats[d]
            n_ids = len(self.domain_pids[d])
            print(f"    Domain {d}: {n_ids} IDs, {n_imgs} Images")
        print("  ---------------------------------------- ")


    def __getitem__(self, index):
        
        item = self.raw_data[index]
        img_path = item['img_path']
        bbox = item['bbox']  # [x, y, w, h]
        pid = int(item['class_id'])
        domain_id = int(item['domain'])
        if not osp.exists(img_path):
            raise RuntimeError(f"Image not found: {img_path}")
        
        with Image.open(img_path) as img:
            img = Image.open(img_path)
            W, H = img.size
            

            assert W > 0 and H > 0, f"invalid image with shape {W} {H}"
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            x1 = max(0, min(x1, W))
            y1 = max(0, min(y1, H))
            x2 = min(x2, W)
            y2 = min(y2, H)

            crop = img.crop((x1, y1, x2, y2))  # PIL crop
            crop = crop.resize((128, 384), Image.BILINEAR)  # PIL resize

            
            to_tensor = ToTensor()
            crop_tensor = to_tensor(crop)
            return crop_tensor, pid, domain_id

    def __len__(self):
        return len(self.meta_data)