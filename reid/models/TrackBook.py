import torch
import torch.nn as nn
from clip import clip
import numpy as np


class LearnableTrackBook(nn.Module):
    def __init__(self, clip_model, n_ctx=6, n_ids=405):
        
        super().__init__()
        
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding  
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer 
        self.ln_final = clip_model.ln_final  
        self.text_projection = clip_model.text_projection  
        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.ctx = nn.Parameter(torch.randn(n_ids, n_ctx, ctx_dim).type(self.dtype))
        
        prompt_prefix = "A photo of a "
        device = next(clip_model.parameters()).device
        
        prefix_tokens = clip.tokenize(prompt_prefix).to(device)  # "A photo of a"
        person_tokens = clip.tokenize("person").to(device)  # "person"
        
        eos_token = clip.tokenize("").to(device)  # [SOS, EOS]
        eos_idx = eos_token[0, 1]  
        with torch.no_grad():
            
            prefix_embed = self.token_embedding(prefix_tokens).type(self.dtype)
            person_embed = self.token_embedding(person_tokens).type(self.dtype)
            eos_embed = self.token_embedding(eos_idx).unsqueeze(0).type(self.dtype)
            
           
            
            self.register_buffer("prefix_embed", prefix_embed[:, :5, :].detach()) 
           
            self.register_buffer("suffix_embed", person_embed[:, 1:2, :].detach())
            
            self.register_buffer("eos_embed", eos_embed.detach())

    def forward(self, track_ids):
        
        ctx = self.ctx[track_ids]
        B = ctx.shape[0]

        
        prefix = self.prefix_embed.expand(B, -1, -1)  # [B, 5, 512]
        suffix = self.suffix_embed.expand(B, -1, -1)  # [B, 1, 512]
        eos = self.eos_embed.expand(B, -1, -1)  # [B, 1, 512]
        
        prompts = torch.cat(
            [prefix, ctx, suffix, eos],
            dim=1
        )
        
        n_curr = prompts.shape[1]
        n_pad = 77 - n_curr
        pads = torch.zeros(B, n_pad, 512, device=prompts.device, dtype=self.dtype)
        full_prompts = torch.cat([prompts, pads], dim=1)
        
        full_prompts = full_prompts + self.positional_embedding.type(self.dtype)
        x = full_prompts.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        eos_indx = n_curr - 1
        
        text_features = x[:, eos_indx, :] @ self.text_projection
        return text_features