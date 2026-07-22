import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Function
from reid.models.backbone.rga_branches import RGA_Branch

WEIGHT_PATH = "D:/Domain_mot/DGTrack/reid/weight/resnet50-19c8e357.pth"


# ==========================================
#  GRL
# ==========================================
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# ==========================================
#  (Domain Classifier)
# ==========================================
class DomainClassifier(nn.Module):
    def __init__(self, in_channels=2048, num_domains=3):
        super(DomainClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)

        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(in_channels // 2, num_domains)

        
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, alpha):
        
        x = ReverseLayerF.apply(x, alpha)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.pool(x).view(x.size(0), -1)
       
        out = self.fc(x)
        return out


# ===================
#   Initialization
# ===================
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# ==========================================
# RGA + DANN + Projector 
# ==========================================
class ResNet50_RGA_Model(nn.Module):
    def __init__(self, pretrained=True, num_feat=2048, height=256, width=128,
                 dropout=0, num_classes=0, num_domains=3, clip_dim=512,  # clip_dim
                 last_stride=1, branch_name='rgasc', scale=8, d_scale=8, model_path=WEIGHT_PATH):

        super(ResNet50_RGA_Model, self).__init__()
        self.pretrained = pretrained
        self.num_feat = num_feat
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.branch_name = branch_name

        print(f'Model Init: Feat={num_feat}, IDs={num_classes}, Domains={num_domains}, ClipDim={clip_dim}')

        if 'rgasc' in branch_name:
            spa_on = True;
            cha_on = True
        elif 'rgas' in branch_name:
            spa_on = True;
            cha_on = False
        elif 'rgac' in branch_name:
            spa_on = False;
            cha_on = True
        else:
            raise NameError

        # Backbone
        self.backbone = RGA_Branch(pretrained=pretrained, last_stride=last_stride,
                                   spa_on=spa_on, cha_on=cha_on, height=height, width=width,
                                   s_ratio=scale, c_ratio=scale, d_ratio=d_scale, model_path=model_path)

        # DANN Head
        self.domain_classifier = DomainClassifier(in_channels=self.num_feat, num_domains=self.num_domains)

        # ReID Head
        self.feat_bn = nn.BatchNorm1d(self.num_feat)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        self.cls = nn.Linear(self.num_feat, self.num_classes, bias=False)

        # CLIP Projector  2048 -> 512
        self.projector = nn.Linear(self.num_feat, clip_dim)
        #  Projector
        nn.init.normal_(self.projector.weight, std=0.01)
        nn.init.constant_(self.projector.bias, 0)

        
        self.feat_bn.apply(weights_init_kaiming)
        self.cls.apply(weights_init_classifier)

    def forward(self, inputs, alpha=0.0, training=True):
        if isinstance(inputs, (list, tuple)):
            im_input = inputs[0]
        else:
            im_input = inputs

        # Backbone
        feat_map = self.backbone(im_input)

        # DANN 
        domain_pred = None
        if training and hasattr(self, 'domain_classifier') and self.domain_classifier is not None:
            domain_pred = self.domain_classifier(feat_map, alpha)

        # ReID (Pool -> BN -> Cls)
        feat_vec = F.avg_pool2d(feat_map, feat_map.size()[2:]).view(feat_map.size(0), -1)
        feat_bn = self.feat_bn(feat_vec)

        if self.dropout > 0:
            feat_bn = self.drop(feat_bn)

        cls_score = None
        if training and self.num_classes is not None:
            cls_score = self.cls(feat_bn)

        # CLIP Projector
        
        feat_proj = None
        if training and hasattr(self, 'projector'):
            
            # feat_for_clip = feat_vec.detach()
            # feat_proj = self.projector(feat_for_clip)
            feat_proj = self.projector(feat_vec)

        if training:
           
            return (feat_vec, feat_bn, cls_score, domain_pred, feat_proj)
        else:
            return (feat_vec, feat_bn)


def resnet50_rga(*args, **kwargs):
    return ResNet50_RGA_Model(*args, **kwargs)