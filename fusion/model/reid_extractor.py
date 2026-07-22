import torch
import torch.nn as nn
import torch.nn.functional as F
from reid.models.rga_model_dann_clip import resnet50_rga
from reid.utils.serialization import load_checkpoint

class ReidExtractor(nn.Module):
    def __init__(self, ckpt,num_classes=45, num_domains=3):
        
        super().__init__()
        print(f"Loading ReID: {ckpt}") 
        self.model = resnet50_rga(pretrained=False, num_feat=2048, height=384, width=128,num_classes=num_classes, num_domains=num_domains)
        ckpt = load_checkpoint(ckpt)
        state_dict = ckpt.get('teacher_state_dict', ckpt['teacher_state_dict'])
        new_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_dict, strict=True) 
        
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()
    def _process_reid_img(self, imgs):
        return (imgs - self.img_mean) / self.img_std

    def forward(self,img):
        x = self._process_reid_img(img)
        feat_vec, feat_bn = self.model(x, training=False)
        feat = F.normalize(feat_bn, p=2,dim=1)
        return feat






