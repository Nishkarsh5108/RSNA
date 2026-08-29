"""
Region-Guided Attention (RGA).

Each anatomical region group (ligament, meniscus, OA, fluid, trauma) gets its
own set of learnable query vectors that attend over the encoded slice features
via cross-attention, then classify only the targets in that group.
"""
import torch
import torch.nn as nn


class RegionAttentionHead(nn.Module):
    """A single region-specific cross-attention head + MLP classifier."""

    def __init__(self, embed_dim, num_targets, num_queries=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries

        # Learnable region queries
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        # Cross-attention: queries attend to encoded slice features
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Classifier: pool all query outputs → predict targets
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * num_queries, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_targets),
        )

    def forward(self, x, mask=None):
        """
        x    : (B, N, D)  encoded slice features
        mask : (B, N)      1 = valid, 0 = padding
        Returns (B, num_targets) logits
        """
        B = x.size(0)

        # Expand queries for the batch
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        # key_padding_mask: True where padding (to be ignored)
        kpm = (mask == 0) if mask is not None else None

        # Cross-attention + residual + norm
        attn_out, _ = self.cross_attn(q, x, x, key_padding_mask=kpm)
        q = self.norm1(q + attn_out)

        # FFN + residual + norm
        q = self.norm2(q + self.ffn(q))

        # Pool & classify
        pooled = q.reshape(B, -1)                  # (B, Q*D)
        return self.classifier(pooled)              # (B, num_targets)


class RegionGuidedAttention(nn.Module):
    """
    Container for all region heads.
    Outputs a single (B, 12) tensor with logits ordered by TARGET_COLS.
    """

    def __init__(self, embed_dim, region_groups, target_cols,
                 num_queries=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.target_cols = target_cols

        self.heads = nn.ModuleDict()
        self.target_indices = {}           # region → list[int]

        for name, targets in region_groups.items():
            self.heads[name] = RegionAttentionHead(
                embed_dim=embed_dim,
                num_targets=len(targets),
                num_queries=num_queries,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.target_indices[name] = [target_cols.index(t) for t in targets]

    def forward(self, x, mask=None):
        B = x.size(0)
        logits = torch.zeros(B, len(self.target_cols), device=x.device, dtype=x.dtype)

        for name, head in self.heads.items():
            region_logits = head(x, mask)          # (B, K)
            for i, col_idx in enumerate(self.target_indices[name]):
                logits[:, col_idx] = region_logits[:, i]

        return logits                              # (B, 12)
