import torch
import torch.nn as nn
import torch.nn.functional as F

class Interactor(nn.Module):
    def __init__(self, dim=256, nhead=4, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward=dim * 4,
            dropout=dropout, batch_fist=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, track_token, det_token):
        B = track_token.shape[0]
        n_track = track_token.shape[1]
        n_det = det_token.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        src = torch.cat([det_token, track_token, cls], dim=1)
        src = self.norm(src)
        out = self.encoder(src)

        det_out = out[:, :n_det, :]
        track_out = out[:, n_det:n_det + n_track, :]
        return track_out, det_out
