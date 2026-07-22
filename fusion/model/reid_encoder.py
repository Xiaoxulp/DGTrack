import torch
import torch.nn as nn
import torch.nn.functional as F
from match.model.unified_components import ReIDLinProj

class ReidEncoder(nn.Module):
    def __init__(self,reid_dim=2048,hidden_dim=256,dropout=0.2):
        super().__init__()
        self.reid_proj = ReIDLinProj(reid_dim, hidden_dim, dropout)

    def forward(self, reid):
        tokens = self.reid_proj(reid)
        return tokens
