"""
Full model: DINOv2 backbone → Shared Transformer Encoder → Region-Guided Attention.
"""
import torch
import torch.nn as nn

from .dino_backbone import DINOBackbone
from .region_attention import RegionGuidedAttention


class KneeModel(nn.Module):
    """
    End-to-end study-level knee MRI classifier.

    Pipeline
    --------
    1.  DINOv2-S (frozen) extracts 384-dim CLS embeddings per slice.
    2.  Plane & sequence-type embeddings are added.
    3.  A lightweight 2-layer Transformer encoder contextualises the
        embeddings across slices / series.
    4.  Region-Guided Attention heads produce 12 pathology logits.
    """

    def __init__(self, config):
        super().__init__()
        D = config.DINO_EMBED_DIM

        # ---------- Frozen backbone ----------
        self.backbone = DINOBackbone(
            model_name=config.DINO_MODEL_NAME,
            embed_dim=D,
            chunk_size=config.DINO_CHUNK_SIZE,
        )

        # ---------- Metadata embeddings ----------
        self.plane_emb = nn.Embedding(config.NUM_PLANES, D)
        self.seq_emb   = nn.Embedding(config.NUM_SEQ_TYPES, D)

        # Learnable [CLS]-like study token (optional global summary)
        # Not used for now — region heads attend directly to slices.

        # ---------- Shared Transformer encoder ----------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=config.ENCODER_HEADS,
            dim_feedforward=D * 4,
            dropout=config.ENCODER_DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=config.ENCODER_LAYERS)
        self.encoder_norm = nn.LayerNorm(D)

        # ---------- Region-Guided Attention ----------
        self.rga = RegionGuidedAttention(
            embed_dim=D,
            region_groups=config.REGION_GROUPS,
            target_cols=config.TARGET_COLS,
            num_queries=config.REGION_NUM_QUERIES,
            num_heads=config.REGION_NUM_HEADS,
            dropout=config.REGION_DROPOUT,
        )

    # ------------------------------------------------------------------
    def forward(self, slices, plane_ids, seq_ids, masks):
        """
        Parameters
        ----------
        slices    : (B, N, H, W)  grayscale in [0, 1]
        plane_ids : (B, N)        long — 0=Sagittal, 1=Axial, 2=Coronal
        seq_ids   : (B, N)        long — 0=non-fluid, 1=fluid-sensitive
        masks     : (B, N)        float — 1=valid slice, 0=padding

        Returns
        -------
        logits    : (B, 12)       raw logits (apply sigmoid for probabilities)
        """
        # 1. DINOv2 feature extraction (no grad)
        features = self.backbone(slices)                       # (B, N, D)

        # 2. Add metadata
        features = features + self.plane_emb(plane_ids) + self.seq_emb(seq_ids)

        # 3. Shared encoder
        src_key_padding_mask = (masks == 0)                    # True → ignore
        encoded = self.encoder(features, src_key_padding_mask=src_key_padding_mask)
        encoded = self.encoder_norm(encoded)                   # (B, N, D)

        # 4. Region-guided classification
        logits = self.rga(encoded, masks)                      # (B, 12)
        return logits

    # ------------------------------------------------------------------
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
