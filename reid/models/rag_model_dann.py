import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.autograd import Function
import torchvision
import numpy as np
from reid.models.backbone.rga_branches import RGA_Branch

__all__ = ['resnet50_rga']
# WEIGHT_PATH = os.path.join(os.path.dirname(__file__), '../..')+'/weights/pre_train/resnet50-19c8e357.pth'
WEIGHT_PATH = "D:/Domain_mot/DGTrack/reid/weight/resnet50-19c8e357.pth"



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
# (Domain Classifier)
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


class ResNet50_RGA_Model(nn.Module):
    '''
    Backbone: ResNet-50 + RGA modules.
    '''

    def __init__(self, pretrained=True, num_feat=2048, height=256, width=128,
                 dropout=0, num_classes=0, num_domains=3,last_stride=1, branch_name='rgasc', scale=8, d_scale=8,
                 model_path=WEIGHT_PATH):
        super(ResNet50_RGA_Model, self).__init__()
        self.pretrained = pretrained
        self.num_feat = num_feat
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.branch_name = branch_name
        print(f'Num of features: {self.num_feat}, Num of IDs: {self.num_classes}, Num of Domains: {self.num_domains}')

        if 'rgasc' in branch_name:
            spa_on = True
            cha_on = True
        elif 'rgas' in branch_name:
            spa_on = True
            cha_on = False
        elif 'rgac' in branch_name:
            spa_on = False
            cha_on = True
        else:
            raise NameError

        self.backbone = RGA_Branch(pretrained=pretrained, last_stride=last_stride,
                                   spa_on=spa_on, cha_on=cha_on, height=height, width=width,
                                   s_ratio=scale, c_ratio=scale, d_ratio=d_scale, model_path=model_path)
        self.domain_classifier = DomainClassifier(in_channels=self.num_feat, num_domains=self.num_domains)
        self.feat_bn = nn.BatchNorm1d(self.num_feat)
        self.feat_bn.bias.requires_grad_(False)
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)
        self.cls = nn.Linear(self.num_feat, self.num_classes, bias=False)

        self.feat_bn.apply(weights_init_kaiming)
        self.cls.apply(weights_init_classifier)

    def forward(self, inputs, alpha=0.0, training=True):
        if isinstance(inputs, (list, tuple)):
            im_input = inputs[0]
        else:
            im_input = inputs

        # 1. Backbone
        feat_map = self.backbone(im_input)

        # 2. DANN
        domain_pred = None
        if training and hasattr(self, 'domain_classifier') and self.domain_classifier is not None:
            domain_pred = self.domain_classifier(feat_map, alpha)

        # 3. ReID
        feat_ = F.avg_pool2d(feat_map, feat_map.size()[2:]).view(feat_map.size(0), -1)

        feat = self.feat_bn(feat_)

        if self.dropout > 0:
            feat = self.drop(feat)

        cls_feat = None
        if training and self.num_classes is not None:
            cls_feat = self.cls(feat)

        if training:
            return (feat_, feat, cls_feat, domain_pred)
        else:
            return (feat_, feat)

def resnet50_rga(*args, **kwargs):
    return ResNet50_RGA_Model(*args, **kwargs)

