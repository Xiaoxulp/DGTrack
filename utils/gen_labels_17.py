import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os.path as osp
import numpy as np
from tqdm import tqdm

seq_root_list = [
    '/home/st3/DGTrack/data/Domain1/images/train/',
    '/home/st3/DGTrack/data/Domain2/images/train/',
    '/home/st3/DGTrack/data/Domain3/images/train/'
]

label_root_list = [
    '/home/st3/DGTrack/data/Domain1/labels_with_ids/train/',
    '/home/st3/DGTrack/data/Domain2/labels_with_ids/train/',
    '/home/st3/DGTrack/data/Domain3/labels_with_ids/train/'
]
tid_curr = 0
tid_last = -1
pid2class = {}
class_counter = 0 
def mkdirs(path):
    if not osp.exists(path):
        os.makedirs(path)
for domain_id,(seq_root,label_root) in enumerate(zip(seq_root_list,label_root_list)):
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root) if os.path.isdir(osp.join(seq_root,s))]
    print(f" Processing Domain: {domain_id}")
    for seq in tqdm(seqs,desc=f"Domain {domain_id}"):
        tqdm.write(f"Now processing seq: {seq}")
        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        idx = np.lexsort(gt.T[:2, :])
        gt = gt[idx, :]
        seq_label_root = osp.join(label_root, seq, 'img1')
        mkdirs(seq_label_root)
       
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
            class_id = pid2class[tid_curr]
            label_fpath = osp.join(seq_label_root, (fmt_str + '.txt').format(fid))
           
            label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:d}\n'.format(
                domain_id,tid_curr, x, y, w, h, class_id
            )
            with open(label_fpath, 'a') as f:
                f.write(label_str)
print("\n============================================")
print(f"Final total identities (tid_curr) = {tid_curr}")
print(f"Final total class_id = {class_counter}")
print("============================================\n")
save_path = "/home/st3/DGTrack/datasets/data_path/total_len.txt"
with open(save_path, "w") as f:
    f.write(str(class_counter))
print(f"Saved total class_id length to {save_path}")

