import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os.path as osp
import numpy as np
import json
from tqdm import tqdm
seq_root_list = [
    '/home/st3/DGTrack/data/Domain1/images/train/',
    '/home/st3/DGTrack/data/Domain2/images/train/',
    '/home/st3/DGTrack/data/Domain3/images/train/'
]
save_json_path = '/home/st3/DGTrack/data/reid_meta.json'
tid_curr = 0
tid_last = -1
pid2class = {}
class_counter = 0
all_reid_samples = []
print("Start generating ReID JSON index...")
for domain_id, seq_root in enumerate(seq_root_list):
    seqs = [s for s in os.listdir(seq_root) if os.path.isdir(osp.join(seq_root, s))]
    print(f"Processing Domain {domain_id}: {len(seqs)} sequences")
    for seq in tqdm(seqs, desc=f"Domain {domain_id}"):
        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        idx = np.lexsort(gt.T[:2, :])
        gt = gt[idx, :]
        
        img_dir = osp.join(seq_root, seq, 'img1')
        fmt_str = '{:06d}'
        if osp.exists(img_dir):
            images = [x for x in os.listdir(img_dir) if x.endswith('.jpg')]
            if len(images) > 0:
                
                if len(images[0].split('.')[0]) == 8:
                    fmt_str = '{:08d}'  
        for fid, tid, x, y, w, h, mark, cls, vis in gt:
            
            if mark == 0:
                continue
            if int(cls) != 1:
                continue
            if vis < 0.3:
                continue
            x = max(0, x)
            y = max(0, y)
            if w * h < 150:
                continue
            fid = int(fid)
            tid = int(tid)
            if tid != tid_last:
                tid_curr += 1
                tid_last = tid
            if tid_curr not in pid2class:
                pid2class[tid_curr] = class_counter
                class_counter += 1
            global_pid = pid2class[tid_curr] # class_id
            img_name = (fmt_str + '.jpg').format(fid)
            img_path = osp.join(seq_root, seq, 'img1', img_name)

            sample = {
                "img_path": img_path,  
                "bbox": [int(x), int(y), int(w), int(h)],  # [x, y, w, h]
                "class_id": int(global_pid),  
                "domain": int(domain_id)  
            }
            all_reid_samples.append(sample)

print(f"\nProcessing done.")
print(f"Total Samples: {len(all_reid_samples)}")
print(f"Total Identities (class_id): {class_counter}")


save_dir = osp.dirname(save_json_path)
if not osp.exists(save_dir):
    os.makedirs(save_dir)

with open(save_json_path, 'w') as f:
    json.dump(all_reid_samples, f)

print(f"JSON saved to: {save_json_path}")