"""2.5D multi-slot MIL model: shared pretrained 2D encoder on 3-slice windows, token
embeddings (slot + slice position), and a choice of study-level aggregation.

Aggregators (agg=):
  mean   masked mean over all window tokens         -> linear 12
  max    masked max over all window tokens          -> linear 12
  gated  Ilse et al. 2018 gated attention, 1 query  -> linear 12
  attn12 per-finding attention (12 queries), each finding scored from its own pooled vector
  tfm    2-layer transformer over tokens, then attn12
"""
import math

import timm
import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class KneeMIL(nn.Module):
    def __init__(self, backbone="convnext_tiny.fb_in22k_ft_in1k", pretrained=True, n_slots=3, n_out=12,
                 dim=384, agg="attn12", use_pos=True, use_slot=True, drop_path=0.1, grad_ckpt=False):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        if grad_ckpt:
            self.enc.set_grad_checkpointing(True)
        nf = self.enc.num_features
        self.proj = nn.Sequential(nn.Dropout(0.2), nn.Linear(nf, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(n_slots, dim) if use_slot else None
        self.pos_mlp = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim)) if use_pos else None
        self.agg, self.n_out = agg, n_out
        if agg in ("mean", "max"):
            self.head = nn.Linear(dim, n_out)
        elif agg == "gated":
            self.att_v = nn.Linear(dim, 128)
            self.att_u = nn.Linear(dim, 128)
            self.att_w = nn.Linear(128, 1)
            self.head = nn.Linear(dim, n_out)
        elif agg in ("attn12", "tfm"):
            if agg == "tfm":
                layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
                self.tfm = nn.TransformerEncoder(layer, 2)
            self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(0.1), nn.Linear(256, n_out))
            self.cls_w = nn.Parameter(torch.randn(n_out, dim) / math.sqrt(dim))
            self.cls_b = nn.Parameter(torch.zeros(n_out))
        else:
            raise ValueError(agg)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def encode(self, x01):
        """x01: float [N, 3, H, W] in [0,1] -> [N, nf]"""
        return self.enc((x01 - self.mean) / self.std)

    def forward(self, x01, mask, pos, wmask=None):
        """x01: float [B,S,K,3,H,W] in [0,1]; mask [B,S]; pos [B,S,K]; optional wmask [B,S,K] (1 = real
        window; used when slots have different window counts) -> logits [B,12]"""
        B, S, K = x01.shape[:3]
        valid = (mask > 0).unsqueeze(-1).expand(B, S, K)
        if wmask is not None:
            valid = valid & (wmask > 0)
        feats = x01.new_zeros(B, S, K, self.proj[1].out_features)
        xv = x01[valid]                                          # [N, 3, H, W], row-major over (B, S, K)
        f = self.encode(xv.contiguous(memory_format=torch.channels_last))
        feats[valid] = self.proj(f.float()).to(feats.dtype)
        tok = feats
        if self.slot_emb is not None:
            tok = tok + self.slot_emb.weight[:S].view(1, S, 1, -1)
        if self.pos_mlp is not None:
            tok = tok + self.pos_mlp(pos.unsqueeze(-1))
        tok = tok.flatten(1, 2)                                  # [B, S*K, D]
        tmask = valid.flatten(1)                                 # True = real token
        if self.agg == "mean":
            m = tmask.unsqueeze(-1).float()
            return self.head((tok * m).sum(1) / m.sum(1).clamp(min=1))
        if self.agg == "max":
            return self.head(tok.masked_fill(~tmask.unsqueeze(-1), -1e4).max(1).values)
        if self.agg == "gated":
            a = self.att_w(torch.tanh(self.att_v(tok)) * torch.sigmoid(self.att_u(tok))).squeeze(-1)
            a = torch.softmax(a.masked_fill(~tmask, -1e4), 1)
            return self.head((a.unsqueeze(-1) * tok).sum(1))
        if self.agg == "tfm":
            tok = self.tfm(tok, src_key_padding_mask=~tmask)
        a = self.att(tok).masked_fill(~tmask.unsqueeze(-1), -1e4)  # [B, T, 12]
        a = torch.softmax(a, 1)
        pooled = torch.einsum("btc,btd->bcd", a, tok)               # [B, 12, D]
        return (pooled * self.cls_w).sum(-1) + self.cls_b
