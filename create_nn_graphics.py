#!/usr/bin/env python3
"""
create_nn_graphics.py
Generates a modern, clean infographic comparing the neural networks used:
ConvNeXt (Local Texture), CoAtNet (Hybrid Attention), and Multi-Head Attention-MIL (Instance Fusion).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
import os

os.makedirs('assets', exist_ok=True)

# 16:7 aspect ratio
fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300)
ax.set_facecolor('#FFFFFF')
fig.patch.set_facecolor('#FFFFFF')

# Title & Banner
fig.text(0.5, 0.95, 'Neural Network Architectures in Musculoskeletal MRI', 
         ha='center', va='center', fontsize=18, fontweight='bold', color='#1E1B4B', fontfamily='sans-serif')
fig.text(0.5, 0.90, 'Mechanisms, Inductive Biases, Mathematical Formulations, and Clinical Pathology Alignment', 
         ha='center', va='center', fontsize=10.5, color='#64748B', fontfamily='sans-serif')

# Colors
C_CONV = '#2563EB'     # Royal Blue
C_CONV_BG = '#F0F7FF'
C_COAT = '#7C3AED'     # Purple
C_COAT_BG = '#F7F4FE'
C_MIL = '#059669'      # Emerald Green
C_MIL_BG = '#F0FDF4'

columns = [
    {
        "x": 0.035, "w": 0.29,
        "title": "1. ConvNeXt (s6 / s6ft)",
        "subtitle": "Modernized Pure-Convolutional Backbone",
        "color": C_CONV, "bg": C_CONV_BG,
        "badge": "Local Inductive Bias (7x7 Depthwise)",
        "items": [
            ("7x7 Depthwise Convolutions", "Expands the local receptive field (matching ViT tokens) without quadratic compute cost."),
            ("Inverted Bottleneck", "Channels expand 4x in latent space (1x1 -> 4x -> 1x1), preserving fine subtle gradients."),
            ("LayerNorm & GELU", "Replaces BatchNorm and ReLU for stable batch-size independent multi-task training."),
            ("Clinical Specialty", "Fine radial meniscus tears (0.36mm), collateral fibers, and synovial fluid margins.")
        ],
        "formula": r"$\mathbf{y} = \mathbf{W}_2 \, \text{GELU}(\mathbf{W}_1 \, \text{DWConv}_{7\times 7}(\mathbf{x})) + \mathbf{x}$"
    },
    {
        "x": 0.355, "w": 0.29,
        "title": "2. CoAtNet-384 (Dual)",
        "subtitle": "Hybrid Convolution + Self-Attention",
        "color": C_COAT, "bg": C_COAT_BG,
        "badge": "Global Receptive Field (Rel-Attention)",
        "items": [
            ("Early Stages (MBConv)", "Extracts low-level edge features, bone cortex borders, and local textures via depthwise convs."),
            ("Late Stages (Self-Attention)", "Multi-head attention dynamically links distant spatial tokens across the entire knee joint."),
            ("Relative Position Bias", "Injects physical coordinate distances directly into the attention matrix (B_{i-j})."),
            ("Clinical Specialty", "Compartment Osteoarthritis, joint space narrowing, and femur-tibia alignment.")
        ],
        "formula": r"$\mathbf{A}_{i,j} = \text{Softmax}\left(\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d}} + \mathbf{B}_{i-j}\right)$"
    },
    {
        "x": 0.675, "w": 0.29,
        "title": "3. 12-Head Attention-MIL",
        "subtitle": "Finding-Specific Instance Aggregator",
        "color": C_MIL, "bg": C_MIL_BG,
        "badge": "Finding-Specific Softmax Flashlight",
        "items": [
            ("2.5D RGB Triplet Bagging", "Decomposes 3D series into 3-slice 3.5mm slabs, capturing depth without 3D GPU crashes."),
            ("Continuous Depth Embeddings", "MLP encodes continuous slice positions (p in [0,1]) and series plane identity."),
            ("12 Dedicated Query Heads", "Each abnormality independently computes attention weights across the token bag."),
            ("Clinical Specialty", "Concentrates >85% attention mass on focal tear slices, preventing signal dilution.")
        ],
        "formula": r"$\alpha_{c,j} = \frac{\exp(\mathbf{w}_{2,c}^T \tanh(\mathbf{W}_1 \mathbf{t}_j))}{\sum_m \exp(\mathbf{w}_{2,c}^T \tanh(\mathbf{W}_1 \mathbf{t}_m))}$"
    }
]

for col in columns:
    x, w = col["x"], col["w"]
    # Outer Card
    card = patches.FancyBboxPatch((x, 0.06), w, 0.80, boxstyle="round,pad=0.015,rounding_size=0.02",
                                 facecolor=col["bg"], edgecolor=col["color"], linewidth=1.8, transform=ax.transAxes)
    ax.add_patch(card)
    
    # Header Banner
    header = patches.FancyBboxPatch((x + 0.005, 0.77), w - 0.01, 0.08, boxstyle="round,pad=0.01,rounding_size=0.015",
                                   facecolor=col["color"], edgecolor="none", transform=ax.transAxes)
    ax.add_patch(header)
    
    ax.text(x + w/2, 0.825, col["title"], ha='center', va='center', fontsize=11.5, fontweight='bold', color='#FFFFFF', transform=ax.transAxes)
    ax.text(x + w/2, 0.788, col["subtitle"], ha='center', va='center', fontsize=8.2, color='#F8FAFC', transform=ax.transAxes)
    
    # Badge
    badge = patches.FancyBboxPatch((x + 0.02, 0.71), w - 0.04, 0.04, boxstyle="round,pad=0.005,rounding_size=0.008",
                                  facecolor='#FFFFFF', edgecolor=col["color"], linewidth=1, transform=ax.transAxes)
    ax.add_patch(badge)
    ax.text(x + w/2, 0.73, col["badge"], ha='center', va='center', fontsize=8.2, fontweight='bold', color=col["color"], transform=ax.transAxes)
    
    # Bullet Items
    y_pos = 0.66
    for heading, desc in col["items"]:
        ax.text(x + 0.015, y_pos, f"• {heading}:", ha='left', va='top', fontsize=8.2, fontweight='bold', color='#1E293B',
                transform=ax.transAxes)
        wrapped_desc = textwrap.fill(desc, width=42)
        ax.text(x + 0.025, y_pos - 0.028, wrapped_desc, ha='left', va='top', fontsize=7.8, color='#475569',
                transform=ax.transAxes, linespacing=1.25)
        y_pos -= 0.115
        
    # Formula Box
    f_box = patches.FancyBboxPatch((x + 0.015, 0.09), w - 0.03, 0.075, boxstyle="round,pad=0.008,rounding_size=0.01",
                                  facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1, transform=ax.transAxes)
    ax.add_patch(f_box)
    ax.text(x + w/2, 0.127, col["formula"], ha='center', va='center', fontsize=9.2, color='#0F172A', transform=ax.transAxes)

ax.axis('off')
plt.tight_layout()

out_path = 'assets/diagram_nn_architectures.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Generated {out_path} successfully!")
