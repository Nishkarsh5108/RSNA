import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs('assets', exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# -------------------------------------------------------------
# 1. Architecture Pipeline Diagram (Compact 16:5 ratio for Slide 4)
# -------------------------------------------------------------
def make_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13.5, 4.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 4.5)

    def draw_box(x, y, w, h, title, subtitle, bg_color, border_color, title_color='#0F172A', sub_color='#334155'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 0.22, title, ha='center', va='top', fontsize=9.5, fontweight='bold', color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + (h - 0.28)/2, subtitle, ha='center', va='center', fontsize=7.5, color=sub_color, multialignment='center', zorder=3, linespacing=1.2)

    def draw_arrow(x1, y1, x2, y2, color='#2563EB', label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=1.8, headwidth=6, headlength=6, shrink=0.05), zorder=4)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1E40AF', zorder=5)

    # 1. Inputs
    draw_box(0.3, 2.35, 2.0, 1.95, "1. Volumetric MRI", 
             "• Sagittal FS / T2\n• Coronal FS / T2\n• Axial FS / T2\n(100+ DICOM Slices)", 
             '#EFF6FF', '#93C5FD')
    draw_box(0.3, 0.2, 2.0, 1.95, "2. Preprocessing", 
             "• Patient-Axis Normal\n• Laterality Mirroring\n• 1st / 99th % Window\n• 0.365 mm/px Grid", 
             '#F0FDF4', '#86EFAC')

    draw_arrow(1.3, 2.35, 1.3, 2.15)

    # 2. 2.5D MIL Windows
    draw_box(2.7, 0.65, 2.2, 3.2, "3. 2.5D Window Bags", 
             "Stratified Slicing\n━━━━━━━━━━━━━\n• K = 8 to 12 windows\n• 3 slices/window (RGB)\n• 3.5mm physical slab\n• Stratified jittering\n\nTensor: [B, 3, K, 3, H, W]", 
             '#FAF5FF', '#D8B4FE')

    draw_arrow(2.3, 1.15, 2.7, 2.25)

    # 3. Heterogeneous Encoders
    draw_box(5.3, 3.05, 2.5, 1.25, "CoAtNet-384 (Dual)", "Convolutions + Self-Attention\nVision-Pure SWA Distillation", '#FEF3C7', '#FCD34D')
    draw_box(5.3, 1.65, 2.5, 1.25, "ConvNeXt (S6 / s6ft)", "7x7 Depthwise Convolutions\nSpecialist Fine-Tuned Weights", '#FEE2E2', '#FCA5A5')
    draw_box(5.3, 0.25, 2.5, 1.25, "Heliosli R6 (384px)", "High-Capacity 384px Backbone\nSingle-Pass In-Memory Decode", '#E0E7FF', '#A5B4FC')

    draw_arrow(4.9, 2.6, 5.3, 3.65)
    draw_arrow(4.9, 2.25, 5.3, 2.25)
    draw_arrow(4.9, 1.9, 5.3, 0.85)

    # 4. Attention Pooling & Heads
    draw_box(8.2, 1.0, 2.6, 2.6, "4. Attention Pooling", 
             "12 Dedicated Attention Heads\n━━━━━━━━━━━━━━━━━\nα_{c,j} = softmax(W₂ tanh(W₁ t_j))\nFinding-specific token queries\n\n+ Slice Position Embeddings\n+ Slot Identity Embeddings", 
             '#F8FAFC', '#CBD5E1')

    draw_arrow(7.8, 3.65, 8.2, 2.9)
    draw_arrow(7.8, 2.25, 8.2, 2.3)
    draw_arrow(7.8, 0.85, 8.2, 1.7)

    # 5. Specialist Routing Matrix & Calibration
    draw_box(11.2, 1.0, 2.0, 2.6, "5. Specialist Head\nRouting Matrix", 
             "• Tissue-specific routing\n• Logit standardization\n• Temperature scaling\n\n12 Calibrated Probabilities\n(Macro AUC: 0.958)", 
             '#ECFDF5', '#6EE7B7')

    draw_arrow(10.8, 2.3, 11.2, 2.3)

    # Bottom summary strip
    strip = patches.FancyBboxPatch((0.3, 0.02), 12.9, 0.16, boxstyle="round,pad=0.01,rounding_size=0.05",
                                   facecolor='#1E293B', edgecolor='#0F172A', zorder=3)
    ax.add_patch(strip)

    plt.tight_layout()
    plt.savefig('assets/diagram_architecture.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Regenerated assets/diagram_architecture.png (Clean 16:5 aspect ratio)")

# -------------------------------------------------------------
# 2. Key Challenges & Systematic Solutions (Infographic with fixed margins)
# -------------------------------------------------------------
def make_challenges_infographic():
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.2)

    def draw_card(x, title, problem_txt, solution_txt, impact_txt, badge_color, border_color):
        # Outer Card
        card = patches.FancyBboxPatch((x, 0.2), 3.5, 4.8, boxstyle="round,pad=0.04,rounding_size=0.15",
                                     facecolor='#F8FAFC', edgecolor=border_color, linewidth=1.5, zorder=2)
        ax.add_patch(card)
        # Header Badge
        header = patches.FancyBboxPatch((x + 0.15, 4.35), 3.2, 0.52, boxstyle="round,pad=0.03,rounding_size=0.1",
                                       facecolor=badge_color, edgecolor=border_color, linewidth=1, zorder=3)
        ax.add_patch(header)
        ax.text(x + 1.75, 4.61, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#FFFFFF', zorder=4)

        # Problem section
        ax.text(x + 0.25, 4.10, "CHALLENGE / BOTTLENECK", fontsize=8, fontweight='bold', color='#DC2626', zorder=3)
        ax.text(x + 0.25, 3.25, problem_txt, fontsize=8, color='#334155', multialignment='left', zorder=3, linespacing=1.25)

        # Divider
        ax.plot([x+0.25, x+3.25], [2.55, 2.55], color='#E2E8F0', linewidth=1.2, zorder=3)

        # Solution section
        ax.text(x + 0.25, 2.35, "OUR SCIENTIFIC SOLUTION", fontsize=8, fontweight='bold', color='#059669', zorder=3)
        ax.text(x + 0.25, 1.55, solution_txt, fontsize=8, color='#334155', multialignment='left', zorder=3, linespacing=1.25)

        # Impact Badge
        impact_box = patches.FancyBboxPatch((x + 0.2, 0.42), 3.1, 0.48, boxstyle="round,pad=0.03,rounding_size=0.08",
                                           facecolor='#EFF6FF', edgecolor='#BFDBFE', linewidth=1, zorder=3)
        ax.add_patch(impact_box)
        ax.text(x + 1.75, 0.66, impact_txt, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1D4ED8', zorder=4)

    draw_card(0.2, "1. Severe Label Noise",
              "• Radiologist reports omit silent / minor\n  pathologies (high false-negative rate).\n• Synovitis clinical agreement was\n  only 0.69 AUC against image reads.\n• Report noise corrupted cross-entropy.",
              "• Formulated Pure-Vision Distillation\n  (training_labels_v6_vision_pure).\n• 4-fold Out-Of-Fold vision teachers\n  re-annotated all 4,349 studies.\n• Eliminated EHR report blind spots.",
              "Gold AUC: 0.890 ➔ 0.9402 (+0.05)",
              '#2563EB', '#93C5FD')

    draw_card(4.0, "2. Scale & Resolution",
              "• 224px squash caused 0.71 mm/px\n  anisotropic aspect-ratio distortion.\n• Fine meniscus tears (0.36mm)\n  and ACL fibers lost spatial clarity.\n• 3D volumes exceeded GPU memory.",
              "• High-Resolution 0.365 mm/px\n  physical grid with 352px field.\n• 2.5D MIL 3-slice slab sampling\n  capturing 3.5mm axial thickness.\n• Stratified jitter for full coverage.",
              "Meniscus AUC: +0.033 Gain",
              '#4F46E5', '#A5B4FC')

    draw_card(7.8, "3. 9-Hour Test Timeout",
              "• Hidden test set contains thousands of\n  uncompressed DICOM volumes.\n• Re-reading & decoding files across\n  multiple models hit the 9h cap.\n• Caused catastrophic study timeouts.",
              "• Single-Decode In-Pass Engine:\n  Decodes each DICOM volume once in RAM.\n• Feeds dual renders (384px & 320px)\n  to R6 + ConvNeXt in single worker.\n• Adaptive stride-2 execution.",
              "Latency: -42% (Zero Timeout)",
              '#0284C7', '#7DD3FC')

    plt.tight_layout()
    plt.savefig('assets/infographic_problems_solutions.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Regenerated assets/infographic_problems_solutions.png")

# -------------------------------------------------------------
# 3. Specialist Head Routing Matrix Heatmap
# -------------------------------------------------------------
def make_specialist_matrix():
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')

    findings = [
        "Effusion", "Synovitis", "Contusion", "Medial Meniscus", 
        "Lateral Meniscus", "MCL Tear", "ACL Tear", "Fracture", 
        "Medial OA", "Lateral OA", "PF OA", "Baker's Cyst"
    ]
    
    models = ["s6ft_specialist\n(Vision Distilled)", "s6ft_v1\n(MCL/Meniscus)", "CoAtNet-384\n(Dual SWA)", "Heliosli R6\n(384px Backbone)", "s6fall\n(Public Baseline)"]
    
    weights = np.array([
        [0.45, 0.10, 0.20, 0.15, 0.10], # Effusion
        [0.45, 0.05, 0.20, 0.15, 0.15], # Synovitis
        [0.40, 0.10, 0.25, 0.15, 0.10], # Contusion
        [0.40, 0.15, 0.20, 0.15, 0.10], # Medial Meniscus
        [0.15, 0.35, 0.20, 0.20, 0.10], # Lateral Meniscus
        [0.10, 0.40, 0.20, 0.20, 0.10], # MCL Tear
        [0.10, 0.15, 0.35, 0.30, 0.10], # ACL Tear
        [0.15, 0.10, 0.30, 0.35, 0.10], # Fracture
        [0.10, 0.10, 0.30, 0.40, 0.10], # Medial OA
        [0.10, 0.10, 0.30, 0.40, 0.10], # Lateral OA
        [0.15, 0.10, 0.30, 0.35, 0.10], # PF OA
        [0.25, 0.15, 0.25, 0.25, 0.10], # Baker's Cyst
    ])

    im = ax.imshow(weights, cmap='YlGnBu', aspect='auto', vmin=0.0, vmax=0.5)

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(findings)))
    ax.set_xticklabels(models, fontsize=8.5, fontweight='bold', color='#1E293B')
    ax.set_yticklabels(findings, fontsize=8, fontweight='bold', color='#1E293B')

    for i in range(len(findings)):
        for j in range(len(models)):
            val = weights[i, j]
            text_color = "white" if val > 0.25 else "#0F172A"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')

    ax.set_title("Pathology-Specific Model Weight Allocation Matrix", fontsize=10.5, fontweight='bold', color='#0F172A', pad=10)
    
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.ax.set_ylabel("Weight Fraction", rotation=-90, va="bottom", fontsize=8, fontweight='bold', color='#1E293B')
    cbar.outline.set_visible(False)

    for spine in ax.spines.values():
        spine.set_color('#E2E8F0')

    plt.tight_layout()
    plt.savefig('assets/diagram_specialist_routing.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Regenerated assets/diagram_specialist_routing.png")

# -------------------------------------------------------------
# 4. Quantitative Results & Accuracy Across 12 Abnormalities
# -------------------------------------------------------------
def make_results_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300, gridspec_kw={'width_ratios': [1.2, 1]})
    fig.patch.set_facecolor('#FFFFFF')

    # Chart 1: Per-Finding AUC Bar Chart
    findings = [
        "Medial OA", "Baker's Cyst", "MCL Tear", "ACL Tear",
        "Effusion", "Medial Meniscus", "Contusion", "Lateral Meniscus",
        "Fracture", "PF OA", "Lateral OA", "Synovitis"
    ]
    scores = [0.991, 0.989, 0.986, 0.980, 0.979, 0.972, 0.964, 0.942, 0.922, 0.885, 0.870, 0.860]
    
    colors = ['#059669' if s >= 0.97 else '#2563EB' if s >= 0.92 else '#0284C7' for s in scores]
    
    y_pos = np.arange(len(findings))
    bars = ax1.barh(y_pos, scores, color=colors, height=0.62, edgecolor='none', zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(findings, fontsize=8, fontweight='bold', color='#1E293B')
    ax1.invert_yaxis()
    ax1.set_xlim(0.75, 1.02)
    ax1.set_xlabel("ROC-AUC Score", fontsize=8.5, fontweight='bold', color='#1E293B')
    ax1.set_title("Per-Abnormality Diagnostic Accuracy (ROC-AUC)", fontsize=10, fontweight='bold', color='#0F172A', pad=8)
    ax1.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1', zorder=0)

    for bar in bars:
        w = bar.get_width()
        ax1.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=7.5, fontweight='bold', color='#0F172A')

    # Chart 2: Milestone Evolution
    milestones = [
        "Baseline\n(Report Labels)",
        "E9 High-Res\n(0.365mm/px)",
        "Strict v1\n(FastMRI-Free)",
        "Specialist FT\n(Vision Targets)",
        "Master Blend\n(final-max)"
    ]
    macro_aucs = [0.927, 0.928, 0.956, 0.958, 0.9585]
    
    ax2.plot(range(len(milestones)), macro_aucs, marker='o', markersize=8, color='#2563EB', linewidth=2.2, zorder=4)
    ax2.fill_between(range(len(milestones)), 0.915, macro_aucs, color='#DBEAFE', alpha=0.5, zorder=2)
    
    ax2.set_ylim(0.915, 0.965)
    ax2.set_ylabel("Competition Macro AUC", fontsize=8.5, fontweight='bold', color='#1E293B')
    ax2.set_title("Iterative Macro AUC Evolution", fontsize=10, fontweight='bold', color='#0F172A', pad=8)
    ax2.grid(True, linestyle='--', alpha=0.5, color='#CBD5E1', zorder=0)
    ax2.set_xticks(range(len(milestones)))
    ax2.set_xticklabels(milestones, fontsize=7.5, fontweight='bold', color='#1E293B')

    for i, val in enumerate(macro_aucs):
        offset_y = 9 if i != 3 else -14
        offset_x = 0 if i != 3 else -8
        ax2.annotate(f"{val:.3f}", (i, val), textcoords="offset points", xytext=(offset_x, offset_y), ha='center',
                     fontsize=8, fontweight='bold', color='#1E3A8A')

    # Silver/Bronze medal cut line
    ax2.axhline(0.958, color='#E11D48', linestyle=':', linewidth=1.5, zorder=3)
    ax2.text(0.1, 0.9592, "★ Silver/Bronze Medal Tier (Rank ~420 / 6,290+)", color='#BE123C', fontsize=7.5, fontweight='bold')

    for ax in [ax1, ax2]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('#CBD5E1')

    plt.tight_layout()
    plt.savefig('assets/chart_results_trajectory.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Regenerated assets/chart_results_trajectory.png")

if __name__ == '__main__':
    make_architecture_diagram()
    make_challenges_infographic()
    make_specialist_matrix()
    make_results_chart()
