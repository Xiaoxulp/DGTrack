import numpy as np
from collections import defaultdict

class ValBuilder:
    def __init__(self, dataset):
        self.dataset = dataset

        self.index_pool = defaultdict(lambda: defaultdict(list))

        print("Building validation index pool...")
        for idx, (_, pid, dom) in enumerate(dataset.meta_data):
            self.index_pool[dom][pid].append(idx)

        self.domains = sorted(list(self.index_pool.keys()))
        print(f"Pool ready. Domains: {self.domains}")

    def build(self, num_ids_per_domain=10, num_imgs_per_id=10, seed=42):
        np.random.seed(seed)
        selected_indices = []
        for dom in self.domains:
           
            all_pids = sorted(list(self.index_pool[dom].keys()))
            
            if num_ids_per_domain == -1 or num_ids_per_domain >= len(all_pids):
                selected_pids = all_pids  
            else:
                selected_pids = np.random.choice(all_pids, num_ids_per_domain, replace=False)
            
            for pid in selected_pids:
                img_idxs = self.index_pool[dom][pid]
                if num_imgs_per_id == -1:
                    
                    selected_indices.extend(img_idxs)
                else:
                    
                    if len(img_idxs) >= num_imgs_per_id:
                        sampled = np.random.choice(img_idxs, num_imgs_per_id, replace=False)
                    else:
                        sampled = np.random.choice(img_idxs, num_imgs_per_id, replace=True)
                    selected_indices.extend(sampled)
        print(f"Val Set Created: {len(selected_indices)} images.")
        return selected_indices