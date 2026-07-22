import torch
import torch.nn as nn
import torch.nn.functional as F
from match.model.reid_encoder import ReidEncoder
from match.model.GeoEncoder import GeoEncoder
from match.model.unified_interactor import Interactor


class UnifiedModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256, output_dim=256):
        super().__init__()
        self.encoder = ReidEncoder(reid_dim=input_dim, hidden_dim=hidden_dim)
        self.geo_encoder = GeoEncoder(hidden_dim)  
        self.interactor = Interactor(dim=hidden_dim)

    def forward(self, det_reid, det_geo, track_reid,track_geo):
        det_reid_emb  = self.encoder(det_reid)
        track_reid_emb = self.encoder(track_reid)
        det_geo_emb   = self.geo_encoder(det_geo)
        track_geo_emb = self.geo_encoder(track_geo)
        
        det_token   = det_reid_emb + det_geo_emb
        track_token = track_reid_emb + track_geo_emb
        track_out, det_out = self.interactor(track_token, det_token)
        match_matrix = torch.matmul(F.normalize(track_out, dim=-1),F.normalize(det_out, dim=-1).transpose(-1, -2))
        cost_matrix = (match_matrix + 1) / 2.0
        return cost_matrix, track_out, det_out

