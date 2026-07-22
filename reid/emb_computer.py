import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import torch
import numpy as np
# from reid.models.rga_model_dann_clip import resnet50_rga
# from reid.models.rag_model_dann import resnet50_rga
from reid.models.rga_model_clip import resnet50_rga
from reid.utils.serialization import load_checkpoint
import torchvision.transforms as T
import torch.nn.functional as F

to_tensor = T.ToTensor()
normalizer = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
class EmbeddingComputer:
    def __init__(self,weight_path, num_classes=751, features=2048,max_batch=1024): # 767
        self.model = None
        self.weight_path = weight_path
        self.features = features
        self.crop_size = (128, 384)
        self.width = 128
        self.height = 384
        self.num_classes = num_classes
        self.max_batch = max_batch

    
    def initialize_model(self):
        self.model = resnet50_rga(
            pretrained=False,
            num_feat=self.features,
            height=self.height,
            width=self.width,
            dropout=0,
            num_classes=self.num_classes,
        )
        
        checkpoint = load_checkpoint(self.weight_path)
        
        for k in list(checkpoint['state_dict'].keys()):  # state_dict
            if k.startswith('cls.'):
                checkpoint['state_dict'].pop(k)
        load_result = self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        self.load_info(load_result)
        
        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1]) 
        self.model.eval().half()

    def load_info(self,load_result):
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys
        if len(missing_keys) == 0 and len(unexpected_keys) == 0:
            print("all layers are successful")
        else:
            if len(missing_keys) > 0:
                print(f"model has {len(missing_keys)} layer not load (Missing Keys):")
                
                print(missing_keys)

    def compute_embedding(self, img, bbox):
        
        if self.model is None:
            self.initialize_model()
        
        h, w = img.shape[:2]
        bbox_clip = np.round(bbox).astype(np.int32)
        

        bbox_clip[:, 0] = bbox_clip[:, 0].clip(0, w)
        bbox_clip[:, 1] = bbox_clip[:, 1].clip(0, h)
        bbox_clip[:, 2] = bbox_clip[:, 2].clip(0, w)
        bbox_clip[:, 3] = bbox_clip[:, 3].clip(0, h)
        
        crops = []
        for box in bbox_clip:
            # Get patch, BGR -> RGB, Resize
            crop = img[box[1]:box[3], box[0]:box[2]]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
           
            crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)  #不要astype(np.float32)保持 uint8
            crop = to_tensor(crop)
            crop = normalizer(crop)
            crop = crop.unsqueeze(0).cuda()
            crops.append(crop)
            
          
        crops = torch.cat(crops, dim=0)
        
        embeddings = []
        for idx in range(0, len(crops), self.max_batch):
            
            batch_crops = crops[idx:idx + self.max_batch]
            batch_crops = batch_crops.cuda().half()
           
            batch_crops = [batch_crops,len(batch_crops)]
            
            with torch.no_grad():
                _,batch_embeddings = self.model(batch_crops,training=False)
            
            batch_embeddings = batch_embeddings.cpu()  
            
            embeddings.append(batch_embeddings)
            del batch_crops
            torch.cuda.empty_cache()  
        
        embeddings = torch.cat(embeddings, dim=0)  
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = embeddings.cpu().numpy()
        return embeddings

