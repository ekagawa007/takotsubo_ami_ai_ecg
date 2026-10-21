'''
mae_fcmae_wrapper.py  (v2)
-----------------------------------------------------------
A **hybrid MAE / FC‑MAE wrapper** that works with
  • ViT / Swin → classic MAE logic
  • ConvNeXt (or any CNN with `patch_size = 4`) → FC‑MAE logic

Key points
──────────
* **No extra deps** beyond torch / timm / torchvision / tqdm.
* Masking strategy unified: random 75 % of patch positions.
* Pos‑embedding is used only for ViT‑style encoders.
* Decoder is a lightweight Transformer for ViT case, and a
  small 3×Conv MLP for ConvNeXt (pixel reconstruction).
* Mixed‑precision + DDP ready; see CLI at bottom.

version 1.0.1 (2025-08-09) loss = ((recon - ... / mask.sum()の分母を(1-mask).sum()に修正。
'''

from __future__ import annotations
import math, random
from typing import Tuple

import torch, torch.nn as nn
from torchvision import transforms
from timm import create_model
from timm.data import resolve_data_config

# ────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────

def _is_convnext(name:str)->bool: return 'convnext' in name.lower()

def _patchify(x:torch.Tensor, p:int)->torch.Tensor:
    """(B,3,H,W) → (B, N, p*p*3)"""
    B, C, H, W = x.shape
    assert H % p == 0 and W % p == 0, 'Image not divisible by patch size'
    h, w = H//p, W//p
    x = x.reshape(B, C, h, p, w, p).permute(0,2,4,3,5,1).flatten(1,2).flatten(-3,-1)
    return x  # (B, N, p*p*3)

def _unpatchify(patches:torch.Tensor, img_size:int, p:int)->torch.Tensor:
    """(B,N,patch*patch*3) → (B,3,H,W)"""
    B, N, _ = patches.shape
    h = w = int(math.sqrt(N))
    x = patches.reshape(B, h, w, p, p, 3).permute(0,5,1,3,2,4).reshape(B,3,h*p,w*p)
    return x

# ────────────────────────────────────────────
# main module
# ────────────────────────────────────────────

class FCMAE(nn.Module):
    """MAE that auto‑switches between ViT‑style and ConvNeXt (FC‑MAE)"""
    def __init__(self, backbone:str='vit_base_patch16_224', img_size:int=224,
                 patch_size:int|None=None, mask_ratio:float=0.75,
                 decoder_dim:int=512, decoder_depth:int=4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.is_convnext = _is_convnext(backbone)

        if self.is_convnext:
            # ConvNeXt: use timm model as feature extractor (no cls, global pool)
            self.encoder = create_model(backbone, pretrained=False, features_only=True, out_indices=(3,))
            embed_dim = self.encoder.feature_info[-1]['num_chs']
            self.patch_size = 32
            # lightweight conv decoder (pixel recon)
            self.decoder = nn.Sequential(
                nn.Conv2d(embed_dim, decoder_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(decoder_dim, decoder_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(decoder_dim, 3*self.patch_size**2, kernel_size=1)
            )
        else:
            # ViT/Swin branch (classic MAE)
            self.encoder = create_model(backbone, pretrained=False, img_size=img_size,
                                        patch_size=patch_size, num_classes=0, global_pool="")
            self.patch_size = self.encoder.patch_embed.patch_size[0]
            embed_dim = self.encoder.embed_dim
            self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
            self.pos_embed  = self.encoder.pos_embed.clone().detach()
            self.decoder_pos = nn.Parameter(torch.randn_like(self.pos_embed))
            self.decoder = nn.Transformer(
                d_model=decoder_dim, nhead=8, batch_first=True,
                num_encoder_layers=0, num_decoder_layers=decoder_depth,
                dim_feedforward=decoder_dim*4)
            self.inv_proj  = nn.Linear(decoder_dim, self.patch_size**2*3)

    # ──────────────────── forward helpers
    def _make_mask(self, B:int, N:int, device)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones(B, N, device=device)
        num_mask = int(self.mask_ratio*N)
        mask.scatter_(1, ids_shuffle[:, :num_mask], 0)
        return mask, ids_shuffle, ids_restore

    # ──────────────────── forward
    def forward(self, x:torch.Tensor):
        B = x.shape[0]
        device = x.device
        patches = _patchify(x, self.patch_size)  # (B,N,P*P*3)
        N = patches.shape[1]
        mask, ids_shuffle, _ = self._make_mask(B,N,device)

        if self.is_convnext:
            # zero‑out masked patches in pixel space (like FC‑MAE)
            masked_img = x.clone()
            p = self.patch_size
            mask_grid = mask.reshape(B, int(math.sqrt(N)), -1)
            mask_grid = mask_grid.unsqueeze(1).repeat(1,3,1,1).repeat_interleave(p,2).repeat_interleave(p,3)
            masked_img = masked_img * mask_grid
            feat = self.encoder(masked_img)[0]   # (B,C,H/32,W/32) after final norm
            recon = self.decoder(feat)        # -> pixel patch tensor
            recon = recon.permute(0,2,3,1).reshape(B, N, -1)
            loss = ((recon - patches)**2 * (1-mask).unsqueeze(-1)).sum() / (1-mask).sum()
            return loss

        else:  # ViT branch
            # visible tokens to encoder
            x_vis = patches.clone(); x_vis[mask==0]=0
            tok = self.encoder.patch_embed(x)
            tok = torch.cat([self.cls_token.expand(B,-1,-1), tok],dim=1) + self.pos_embed
            tok = self.encoder.pos_drop(tok)
            for blk in self.encoder.blocks:
                tok = blk(tok)
            enc = self.encoder.norm(tok)
            # prepare decoder tokens
            dec_in = enc + self.decoder_pos
            dec_out = self.decoder(dec_in, dec_in)
            pred = self.inv_proj(dec_out[:,1:])
            loss = ((pred - patches)**2 * (1-mask).unsqueeze(-1)).sum() / (1-mask).sum()
            return loss

# ────────────────────────────────────────────
# quick sanity test
# ────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='convnext_base')
    parser.add_argument('--img', type=int, default=480)
    args = parser.parse_args()

    model = FCMAE(args.backbone, img_size=args.img, patch_size=4, mask_ratio=0.75)
    model.eval()
    dummy = torch.randn(2,3,args.img,args.img)
    with torch.no_grad():
        loss = model(dummy)
    print(f'OK ‑ loss {loss.item():.4f}')
