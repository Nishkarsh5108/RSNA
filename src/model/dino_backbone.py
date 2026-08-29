"""
Frozen DINOv2-Small backbone for slice-level feature extraction.
Uses timm for clean model loading (pre-installed on Kaggle).
"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")


class DINOBackbone(nn.Module):
    """
    Wraps DINOv2-Small (ViT-S/14, 384-dim) from timm.
    Completely frozen — no gradients ever flow through this module.
    Processes slices in memory-efficient chunks.
    """

    def __init__(self, model_name="vit_small_patch14_dinov2.lvd142m", embed_dim=384, chunk_size=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size

        # Load pretrained DINOv2-Small; num_classes=0 → output is CLS token
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # ImageNet normalisation constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # ------------------------------------------------------------------
    def _normalise(self, x):
        """x: (B, 3, H, W) in [0, 1] → ImageNet-normalised."""
        return (x - self.mean) / self.std

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, N, H, W)   grayscale slices in [0, 1]

        Returns
        -------
        (B, N, embed_dim)   CLS-token embeddings per slice
        """
        B, N, H, W = x.shape

        # Flatten batch & slice dims → (B*N, 1, H, W) → 3-channel
        flat = x.reshape(B * N, 1, H, W).expand(-1, 3, -1, -1)
        flat = self._normalise(flat)

        # Process in chunks to limit peak VRAM
        parts = []
        for i in range(0, flat.size(0), self.chunk_size):
            parts.append(self.model(flat[i : i + self.chunk_size]))
        embeddings = torch.cat(parts, dim=0)  # (B*N, D)

        return embeddings.reshape(B, N, self.embed_dim)
