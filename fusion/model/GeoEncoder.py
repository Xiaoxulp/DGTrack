import torch
import torch.nn as nn
import torch.nn.functional as F

class GeoEncoder(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(4, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, 4]
        return self.drop(self.norm(self.linear(x))) 