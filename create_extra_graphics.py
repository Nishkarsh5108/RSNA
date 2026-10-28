import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs('assets', exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

# -------------------------------------------------------------
# 1. MRI Planes & 2.5D Slab Sampling Diagram
# -------------------------------------------------------------
def make_mri_planes_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)

    def draw_card(x, y, w, h, title, sub, bg, border, head_c):
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                                     facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(card)
        ax.text(x + w/2, y + h - 0.25, title, ha='center', va='top', fontsize=10.5, fontweight='bold', color=head_c, zorder=3)
        ax.text(x + w/2, y + h - 0.55, sub, ha='center', va='top', fontsize=8, fontweight='bold', color='#64748B', zorder=3)

    # 3 Anatomical Planes Cards
    draw_card(0.4, 1.8, 3.4, 2.6, "1. Sagittal Plane (Side View)", "ACL, PCL, Meniscus Body, Patella", '#EFF6FF', '#93C5FD', '#1E40AF')
    draw_card(4.3, 1.8, 3.4, 2.6, "2. Coronal Plane (Front View)", "MCL, LCL, Meniscus Horns, Cartilage", '#F5F3FF', '#DDD6FE', '#582C83')
    draw_card(8.2, 1.8, 3.4, 2.6, "3. Axial Plane (Top View)", "Patellofemoral Joint, Baker's Cyst", '#ECFDF5', '#A7F3D0', '#065F46')

    # Content inside Plane Cards
    ax.text(0.6, 2.9, "• Slice axis: Lateral to Medial\n• Right Knee: Slices reversed so\n  slice 0 is always lateral.\n• Normal Vector: Projection of\n  ImagePositionPatient on normal.", fontsize=8, color='#334155', linespacing=1.3, zorder=3)
    ax.text(4.5, 2.9, "• Slice axis: Anterior to Posterior\n• Right Knee: Columns mirrored (W-axis)\n  to standardize anatomical medial/lateral.\n• Best for collateral ligament tears.", fontsize=8, color='#334155', linespacing=1.3, zorder=3)
    ax.text(8.4, 2.9, "• Slice axis: Superior to Inferior\n• Critical for Trochlea & PF OA.\n• Variable slice count (20 to 144 slices;\n  high sampling density needed).", fontsize=8, color='#334155', linespacing=1.3, zorder=3)

    # Bottom Slab Formulation Card
    slab_box = patches.FancyBboxPatch((0.4, 0.2), 11.2, 1.3, boxstyle="round,pad=0.03,rounding_size=0.1",
                                     facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.5, zorder=2)
    ax.add_patch(slab_box)
    ax.text(6.0, 1.25, "THE 2.5D MIL SLAB FORMULATION: Why not 2D or 3D?", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A', zorder=3)
    ax.text(6.0, 0.72, 
            "• Pure 2D ignores depth: A tear in slice 14 is indistinguishable without context of slice 13 and 15.\n"
            "• Full 3D exhausts VRAM: A 3D CNN on 100+ slices per study causes Out-Of-Memory errors on GPUs.\n"
            "• 2.5D Triplet Window (Our Solution): Each window stacks 3 slices {c-s, c, c+s} spanning ~3.5mm into RGB channels.\n"
            "   Convolutions mix cross-slice depth in the first layer while retaining 2D pre-trained ImageNet backbones!", 
            ha='center', va='center', fontsize=8, color='#334155', linespacing=1.25, zorder=3)

    plt.tight_layout()
    plt.savefig('assets/diagram_mri_planes_windows.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Created assets/diagram_mri_planes_windows.png")

# -------------------------------------------------------------
# 2. Vision Distillation Teacher-Student Loop Diagram
# -------------------------------------------------------------
def make_vision_distillation_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.4), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)

    def draw_box(x, y, w, h, title, sub, bg, border, head_c):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                                    facecolor=bg, edgecolor=border, linewidth=1.5, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.25, title, ha='center', va='top', fontsize=9.5, fontweight='bold', color=head_c, zorder=3)
        ax.text(x + w/2, y + (h - 0.35)/2, sub, ha='center', va='center', fontsize=7.5, color='#334155', multialignment='center', zorder=3, linespacing=1.2)

    def draw_arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#2563EB', edgecolor='#2563EB', width=1.8, headwidth=6, headlength=6, shrink=0.05), zorder=4)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1E40AF', zorder=5)

    draw_box(0.4, 2.5, 3.2, 1.6, "1. Noisy EHR Text Reports", 
             "• Standard NLP-extracted labels (labels_v3)\n• Radiologists often omit incidental signs\n• Synovitis gold agreement: only 0.69 AUC\n• High false negative rate in text", 
             '#FEF2F2', '#FECACA', '#DC2626')

    draw_box(4.4, 2.5, 3.4, 1.6, "2. 4-Fold OOF Vision Teachers", 
             "• R6 & ConvNeXt trained on fold partitions\n• Predict out-of-fold probabilities on images\n• 4,349 studies scored via pure vision\n• Captures what is ACTUALLY in the scan", 
             '#EFF6FF', '#BFDBFE', '#1D4ED8')

    draw_box(8.6, 2.5, 3.0, 1.6, "3. Clean Distilled Targets", 
             "• training_labels_v6_vision_pure.csv\n• 75-100% pure vision distillation\n• Inconsistent report noise pruned\n• Preserves confirmed clinical positives", 
             '#F0FDF4', '#BBF7D0', '#059669')

    draw_arrow(3.6, 3.3, 4.4, 3.3, "OOF Inference")
    draw_arrow(7.8, 3.3, 8.6, 3.3, "Distillation Filter")

    # Bottom Student Fine-Tuning Box
    draw_box(2.2, 0.4, 7.6, 1.5, "4. Specialist Student Training (s6ft_specialist_ema)", 
             "• ConvNeXt & CoAtNet fine-tuned directly on pure vision targets\n"
             "• Multi-task soft BCE loss: L = Σ α_c [y_c log(p_c) + (1 - y_c) log(1 - p_c)]\n"
             "• Result: Standalone Gold-58 AUC jumped from 0.890 to 0.9402 (+0.05 leap!)\n"
             "• Effusion: 0.979  •  Synovitis: 0.860  •  Contusion: 0.954  •  Medial Meniscus: 0.972", 
             '#FAF5FF', '#E9D5FF', '#582C83')

    draw_arrow(10.1, 2.5, 6.0, 1.9, "Fine-Tuning Targets")

    plt.tight_layout()
    plt.savefig('assets/diagram_vision_distillation.png', dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print("Created assets/diagram_vision_distillation.png")

if __name__ == '__main__':
    make_mri_planes_diagram()
    make_vision_distillation_diagram()
