from __future__ import absolute_import
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import random

class RandomIdentitySampler(Sampler):
   
    def __init__(self, dataSet, num_instances=4):
       
        self.dataSet = dataSet
        self.data_source = dataSet.meta_data
        self.num_instances = num_instances
        
        self.index_dic = defaultdict(list)
       
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        
        return self.num_samples * self.num_instances

    def __iter__(self):
        
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]  
            n = len(t) 
            if len(t) >= self.num_instances:
                
                start = np.random.randint(0, n - self.num_instances + 1)
                t = t[start:start + self.num_instances]
            else:
                
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        
        return iter(ret)

        

        