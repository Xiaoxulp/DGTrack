import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
   
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cx_proj = nn.Linear(1, hidden_dim // 4)
        self.cy_proj = nn.Linear(1, hidden_dim // 4)
        self.w_proj  = nn.Linear(1, hidden_dim // 4)
        self.h_proj  = nn.Linear(1, hidden_dim // 4)
    def forward(self, x):
        cx, cy, w, h = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
        pe_cx = self.cx_proj(cx)  # [B, N, hidden_dim//4]
        pe_cy = self.cy_proj(cy)
        pe_w  = self.w_proj(w)
        pe_h  = self.h_proj(h)
        pe = torch.cat([pe_cx, pe_cy, pe_w, pe_h], dim=-1)  # [B, N, hidden_dim]
        return pe

class ReIDLinProj(nn.Module):
    
    def __init__(self, input_dim=2048, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout = dropout
        self.linear = nn.Linear(input_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)  # (B, N, hidden_dim)
        x = self.norm(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


