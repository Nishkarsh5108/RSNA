import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Authentic University of Delhi Purple & Modern Academic Theme
    C_BG_PAGE = RGBColor(248, 250, 252)     # Slate 50 (clean light background)
    C_WHITE = RGBColor(255, 255, 255)       # Crisp White
    C_NAVY_DARK = RGBColor(15, 23, 42)      # Slate 900 (deep readable text)
    C_DU_PURPLE = RGBColor(88, 44, 131)     # Official Delhi University Purple (#582C83)
    C_DU_PURPLE_DARK = RGBColor(65, 25, 95) # Deep Royal Purple
    C_DU_PURPLE_BG = RGBColor(245, 243, 255)# Soft Purple 50
    C_DU_PURPLE_BORDER = RGBColor(216, 180, 254) # Purple 300
    C_BLUE_ACCENT = RGBColor(37, 99, 235)   # Technical Accent Blue
    C_BLUE_LIGHT = RGBColor(239, 246, 255)  # Soft Blue 50
    C_SLATE_TEXT = RGBColor(51, 65, 85)     # Slate 700
    C_SLATE_MUTED = RGBColor(100, 116, 139) # Slate 500
    C_BORDER_LIGHT = RGBColor(226, 232, 240)# Slate 200
    C_GREEN = RGBColor(5, 150, 105)         # Emerald 600 (top metrics)
    C_GREEN_BG = RGBColor(236, 253, 245)    # Emerald 50
    C_ROSE = RGBColor(225, 29, 72)          # Rose 600 (challenges)
    C_ROSE_BG = RGBColor(255, 241, 242)     # Rose 50

    # Official Purple Delhi University Logo
    DU_LOGO_PATH = os.path.abspath('assets/du_logo_purple_vector.png')

    def set_slide_background(slide, color=C_BG_PAGE):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.color.rgb = color

    def add_header(slide, title_text, category_text, slide_num):
        header_rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(1.15))
        header_rect.fill.solid()
        header_rect.fill.fore_color.rgb = C_WHITE
        header_rect.line.color.rgb = C_BORDER_LIGHT
        header_rect.line.width = Pt(1)

        # Top purple accent line
        top_accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.04))
        top_accent.fill.solid()
        top_accent.fill.fore_color.rgb = C_DU_PURPLE
        top_accent.line.fill.background()

        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(9.5), Inches(0.28))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(8.5)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_DU_PURPLE

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.38), Inches(9.5), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(21)
        p_title.font.bold = True
        p_title.font.color.rgb = C_NAVY_DARK

        if os.path.exists(DU_LOGO_PATH):
            slide.shapes.add_picture(DU_LOGO_PATH, Inches(11.0), Inches(0.12), height=Inches(0.9))

        du_text = slide.shapes.add_textbox(Inches(11.95), Inches(0.20), Inches(1.2), Inches(0.75))
        tf_du = du_text.text_frame
        tf_du.word_wrap = True
        tf_du.margin_left = tf_du.margin_right = tf_du.margin_top = tf_du.margin_bottom = 0
        p_du1 = tf_du.paragraphs[0]
        p_du1.text = "University of Delhi"
        p_du1.font.size = Pt(8.5)
        p_du1.font.bold = True
        p_du1.font.color.rgb = C_DU_PURPLE
        p_du2 = tf_du.add_paragraph()
        p_du2.text = "Dept. of Electronics & Communication"
        p_du2.font.size = Pt(7.5)
        p_du2.font.color.rgb = C_SLATE_MUTED

        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.1), Inches(10.0), Inches(0.3))
        tf_foot = footer_box.text_frame
        tf_foot.margin_left = tf_foot.margin_top = tf_foot.margin_bottom = 0
        p_foot = tf_foot.paragraphs[0]
        p_foot.text = "Knee Abnormality Detection  •  Multi-View 2.5D MIL & Specialist Ensemble  •  University of Delhi"
        p_foot.font.size = Pt(8)
        p_foot.font.color.rgb = C_SLATE_MUTED

        num_box = slide.shapes.add_textbox(Inches(11.8), Inches(7.1), Inches(0.8), Inches(0.3))
        tf_num = num_box.text_frame
        tf_num.margin_right = tf_num.margin_top = tf_num.margin_bottom = 0
        p_num = tf_num.paragraphs[0]
        p_num.text = f"{slide_num} / 7"
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.font.size = Pt(8.5)
        p_num.font.bold = True
        p_num.font.color.rgb = C_SLATE_MUTED

    # =========================================================================
    # SLIDE 1: Title Slide (Featuring Official Purple DU Logo)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1, C_WHITE)

    top_bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(0.2))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = C_DU_PURPLE
    top_bar.line.fill.background()

    # Prominent Purple Delhi University Seal
    if os.path.exists(DU_LOGO_PATH):
        slide1.shapes.add_picture(DU_LOGO_PATH, Inches(1.0), Inches(0.8), height=Inches(1.55))

    inst_box = slide1.shapes.add_textbox(Inches(2.75), Inches(0.95), Inches(9.5), Inches(1.2))
    tf_inst = inst_box.text_frame
    tf_inst.word_wrap = True
    p1 = tf_inst.paragraphs[0]
    p1.text = "UNIVERSITY OF DELHI"
    p1.font.size = Pt(17.5)
    p1.font.bold = True
    p1.font.color.rgb = C_DU_PURPLE
    
    p2 = tf_inst.add_paragraph()
    p2.text = "DEPARTMENT OF ELECTRONICS AND COMMUNICATION  •  MAJOR RESEARCH PROJECT"
    p2.font.size = Pt(9.5)
    p2.font.bold = True
    p2.font.color.rgb = RGBColor(126, 34, 206) # Purple 700
    p2.space_before = Pt(3)

    line = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(2.6), Inches(11.333), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = C_BORDER_LIGHT
    line.line.fill.background()

    title_card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.85), Inches(11.333), Inches(1.85))
    title_card.fill.solid()
    title_card.fill.fore_color.rgb = C_DU_PURPLE_BG
    title_card.line.color.rgb = C_DU_PURPLE_BORDER
    title_card.line.width = Pt(1.5)

    title_box = slide1.shapes.add_textbox(Inches(1.3), Inches(2.95), Inches(10.7), Inches(1.6))
    tf_t = title_box.text_frame
    tf_t.word_wrap = True
    pt1 = tf_t.paragraphs[0]
    pt1.text = "Knee Abnormality Detection"
    pt1.font.size = Pt(30)
    pt1.font.bold = True
    pt1.font.color.rgb = C_NAVY_DARK

    pt2 = tf_t.add_paragraph()
    pt2.text = "Deep Multi-View 2.5D MIL, Vision Distillation & Specialist Ensembles for Musculoskeletal MRI"
    pt2.font.size = Pt(13.5)
    pt2.font.bold = True
    pt2.font.color.rgb = C_DU_PURPLE
    pt2.space_before = Pt(4)

    pt3 = tf_t.add_paragraph()
    pt3.text = "Evaluating 12 Concurrent Pathology Heads Across Multi-Plane Volumetric MRI  |  RSNA / Kaggle Machine Learning"
    pt3.font.size = Pt(10)
    pt3.font.color.rgb = C_SLATE_TEXT
    pt3.space_before = Pt(6)

    # Team Members Card (Left)
    card_team = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.9), Inches(5.5), Inches(2.0))
    card_team.fill.solid()
    card_team.fill.fore_color.rgb = C_BG_PAGE
    card_team.line.color.rgb = C_BORDER_LIGHT
    card_team.line.width = Pt(1)

    tb_team = slide1.shapes.add_textbox(Inches(1.2), Inches(5.05), Inches(5.1), Inches(1.7))
    tf_team = tb_team.text_frame
    tf_team.word_wrap = True
    p_th = tf_team.paragraphs[0]
    p_th.text = "PROJECT TEAM (STUDENTS)"
    p_th.font.size = Pt(10)
    p_th.font.bold = True
    p_th.font.color.rgb = C_DU_PURPLE

    members = [
        ("Nishkarsh Singhal", "24294917099"),
        ("Anuj Chaubey", "24294917055"),
        ("Vardhan Harsh Nigam", "24294917014")
    ]
    for name, roll in members:
        p_m = tf_team.add_paragraph()
        p_m.text = f"•  {name}   —   Roll No: {roll}"
        p_m.font.size = Pt(10.5)
        p_m.font.bold = True
        p_m.font.color.rgb = C_NAVY_DARK
        p_m.space_before = Pt(3)

    # Supervision Card (Right)
    card_sup = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.833), Inches(4.9), Inches(5.5), Inches(2.0))
    card_sup.fill.solid()
    card_sup.fill.fore_color.rgb = C_BG_PAGE
    card_sup.line.color.rgb = C_BORDER_LIGHT
    card_sup.line.width = Pt(1)

    tb_sup = slide1.shapes.add_textbox(Inches(7.033), Inches(5.05), Inches(5.1), Inches(1.7))
    tf_sup = tb_sup.text_frame
    tf_sup.word_wrap = True
    p_sh = tf_sup.paragraphs[0]
    p_sh.text = "ACADEMIC SUPERVISION"
    p_sh.font.size = Pt(10)
    p_sh.font.bold = True
    p_sh.font.color.rgb = C_DU_PURPLE

    p_s1 = tf_sup.add_paragraph()
    p_s1.text = "Supervisor:"
    p_s1.font.size = Pt(9)
    p_s1.font.color.rgb = C_SLATE_MUTED
    p_s1.space_before = Pt(4)

    p_s1n = tf_sup.add_paragraph()
    p_s1n.text = "Dr. Varun Saxena"
    p_s1n.font.size = Pt(11)
    p_s1n.font.bold = True
    p_s1n.font.color.rgb = C_NAVY_DARK

    p_s2 = tf_sup.add_paragraph()
    p_s2.text = "Co-Supervisor:"
    p_s2.font.size = Pt(9)
    p_s2.font.color.rgb = C_SLATE_MUTED
    p_s2.space_before = Pt(4)

    p_s2n = tf_sup.add_paragraph()
    p_s2n.text = "Dr. Sangeeta Yadav"
    p_s2n.font.size = Pt(11)
    p_s2n.font.bold = True
    p_s2n.font.color.rgb = C_NAVY_DARK

    # =========================================================================
    # SLIDE 2: Clinical Background & The Computational Challenge
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "Clinical Problem & The Computational Challenge", "1. Problem Motivation", 2)

    top_card2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(1.05))
    top_card2.fill.solid()
    top_card2.fill.fore_color.rgb = C_WHITE
    top_card2.line.color.rgb = C_BORDER_LIGHT
    top_card2.line.width = Pt(1)

    tb_top2 = slide2.shapes.add_textbox(Inches(1.0), Inches(1.42), Inches(11.3), Inches(0.9))
    tf_top2 = tb_top2.text_frame
    tf_top2.word_wrap = True
    p_ov1 = tf_top2.paragraphs[0]
    p_ov1.text = "CLINICAL CONTEXT: The Diagnostic Bottleneck in Musculoskeletal MRI"
    p_ov1.font.size = Pt(11)
    p_ov1.font.bold = True
    p_ov1.font.color.rgb = C_DU_PURPLE

    p_ov2 = tf_top2.add_paragraph()
    p_ov2.text = (
        "Knee pain and joint injuries affect over 250 million people worldwide. Magnetic Resonance Imaging (MRI) is the non-invasive "
        "gold standard for diagnosis. However, an accurate evaluation requires examining 100+ high-resolution 3D DICOM slices per patient across "
        "multiple anatomical orientations to detect up to 12 concurrent structural injuries—leading to severe radiologist burnout, delayed reporting, and diagnostic variability."
    )
    p_ov2.font.size = Pt(9.5)
    p_ov2.font.color.rgb = C_SLATE_TEXT
    p_ov2.space_before = Pt(3)

    col_width = Inches(3.75)
    gap = Inches(0.24)
    start_x = Inches(0.8)
    card_y = Inches(2.55)
    card_h = Inches(4.35)

    challenges = [
        (
            "Challenge 1: Volumetric Multi-Sequence",
            "Complex 3D Geometry",
            C_DU_PURPLE_BG,
            C_DU_PURPLE_BORDER,
            C_DU_PURPLE,
            [
                ("3 Anatomical Planes:", "Sagittal, Coronal, and Axial series with both Fluid-Sensitive (FS) and non-FS T2 contrast."),
                ("Slice Sorting & Alignment:", "Variable slice thickness (0.6mm - 4.0mm). Slices must be physically sorted by patient normal vectors (n = r × c)."),
                ("Laterality Normalization:", "Right and left knees have opposing lateral/medial geometry; requires automated column mirroring & slice reversal."),
                ("Large Volume Size:", "Each patient study holds ~100+ raw slices, far exceeding 3D GPU memory if loaded naively.")
            ],
            "Slab Depth: 3.5mm  •  Multi-Plane Input"
        ),
        (
            "Challenge 2: Multi-Label Pathology",
            "12 Concurrent Findings",
            RGBColor(254, 243, 199),
            RGBColor(252, 211, 77),
            RGBColor(146, 64, 14),
            [
                ("12 Target Conditions:", "ACL tear, MCL tear, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Joint Effusion, Synovitis, Baker's Cyst, Contusion, Fracture."),
                ("Micro-tears vs. Diffuse Signs:", "Subtle 0.36mm ligament fiber tears occur alongside large diffuse joint fluid accumulations."),
                ("Severe Class Imbalance:", "Positive pathology frequency ranges from <5% (fractures) to >65% (effusions)."),
                ("Pathological Co-occurrence:", "Bone contusions frequently mask underlying ACL tears, confounding simple classifiers.")
            ],
            "12 Finding Heads  •  Severe Imbalance"
        ),
        (
            "Challenge 3: Operational Constraints",
            "Strict Hidden-Test SLA",
            C_ROSE_BG,
            RGBColor(253, 164, 175),
            RGBColor(159, 18, 57),
            [
                ("9-Hour Kaggle Ceiling:", "Inference on hidden test set (thousands of uncompressed DICOMs) strictly capped at 9 hours."),
                ("The Timeout Trap:", "Multi-stage architectures that re-read DICOM files run out of time and score 0.0 on unreached studies."),
                ("Physical Scale Distortion:", "Default 224px image squash warps physical scale (0.62–0.86 mm/px), obscuring fine cartilage boundaries."),
                ("Competition Metric:", "Official ranking is Macro ROC-AUC averaged equally across all 12 abnormalities.")
            ],
            "9h Deadline  •  Sub-Second Requirement"
        )
    ]

    for i, (title, sub, bg_c, border_c, head_c, bullets, kpi_text) in enumerate(challenges):
        x = start_x + i * (col_width + gap)
        card = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, card_y, col_width, card_h)
        card.fill.solid()
        card.fill.fore_color.rgb = C_WHITE
        card.line.color.rgb = border_c
        card.line.width = Pt(1.5)

        htag = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x + Inches(0.15), card_y + Inches(0.15), col_width - Inches(0.3), Inches(0.65))
        htag.fill.solid()
        htag.fill.fore_color.rgb = bg_c
        htag.line.color.rgb = border_c
        htag.line.width = Pt(1)

        tb_h = slide2.shapes.add_textbox(x + Inches(0.25), card_y + Inches(0.18), col_width - Inches(0.5), Inches(0.55))
        tf_h = tb_h.text_frame
        tf_h.word_wrap = True
        ph1 = tf_h.paragraphs[0]
        ph1.text = title
        ph1.font.size = Pt(10.5)
        ph1.font.bold = True
        ph1.font.color.rgb = head_c
        ph2 = tf_h.add_paragraph()
        ph2.text = sub.upper()
        ph2.font.size = Pt(8)
        ph2.font.bold = True
        ph2.font.color.rgb = C_SLATE_MUTED

        tb_b = slide2.shapes.add_textbox(x + Inches(0.2), card_y + Inches(0.88), col_width - Inches(0.4), Inches(2.8))
        tf_b = tb_b.text_frame
        tf_b.word_wrap = True
        tf_b.margin_left = tf_b.margin_right = 0
        
        for idx, (b_title, b_desc) in enumerate(bullets):
            pb = tf_b.add_paragraph() if idx > 0 else tf_b.paragraphs[0]
            pb.text = f"• {b_title} "
            pb.font.size = Pt(9.2)
            pb.font.bold = True
            pb.font.color.rgb = C_NAVY_DARK
            pb.space_before = Pt(5) if idx > 0 else Pt(0)
            
            run = pb.add_run()
            run.text = b_desc
            run.font.bold = False
            run.font.size = Pt(8.5)
            run.font.color.rgb = C_SLATE_TEXT

        kpi_box = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x + Inches(0.2), card_y + card_h - Inches(0.52), col_width - Inches(0.4), Inches(0.38))
        kpi_box.fill.solid()
        kpi_box.fill.fore_color.rgb = bg_c
        kpi_box.line.color.rgb = border_c
        kpi_box.line.width = Pt(1)
        tb_kpi = slide2.shapes.add_textbox(x + Inches(0.2), card_y + card_h - Inches(0.52), col_width - Inches(0.4), Inches(0.38))
        tf_kpi = tb_kpi.text_frame
        p_kpi = tf_kpi.paragraphs[0]
        p_kpi.text = kpi_text
        p_kpi.alignment = PP_ALIGN.CENTER
        p_kpi.font.size = Pt(8.5)
        p_kpi.font.bold = True
        p_kpi.font.color.rgb = head_c

    # =========================================================================
    # SLIDE 3: Key Bottlenecks Encountered & Engineering Solutions
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "Key Bottlenecks Encountered & Engineering Solutions", "2. Diagnostic & Engineering Breakthroughs", 3)

    CHALLENGES_IMG = os.path.abspath('assets/infographic_problems_solutions.png')
    if os.path.exists(CHALLENGES_IMG):
        slide3.shapes.add_picture(CHALLENGES_IMG, Inches(0.8), Inches(1.35), width=Inches(7.6))

    rx = Inches(8.6)
    rw = Inches(3.933)
    
    card_sol1 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, rx, Inches(1.35), rw, Inches(1.75))
    card_sol1.fill.solid()
    card_sol1.fill.fore_color.rgb = C_WHITE
    card_sol1.line.color.rgb = C_DU_PURPLE_BORDER
    card_sol1.line.width = Pt(1.2)

    tb_s1 = slide3.shapes.add_textbox(rx + Inches(0.15), Inches(1.45), rw - Inches(0.3), Inches(1.55))
    tf_s1 = tb_s1.text_frame
    tf_s1.word_wrap = True
    ps1_h = tf_s1.paragraphs[0]
    ps1_h.text = "1. Empirical Proof: Vision Distillation"
    ps1_h.font.size = Pt(11)
    ps1_h.font.bold = True
    ps1_h.font.color.rgb = C_DU_PURPLE

    ps1_b = tf_s1.add_paragraph()
    ps1_b.text = (
        "• Diagnosed that text reports silently omitted incidental findings like Synovitis (0.69 agreement with imaging).\n"
        "• 4-fold Out-Of-Fold vision teachers generated pure vision targets (training_labels_v6_vision_pure).\n"
        "• Gold AUC jumped from 0.890 to 0.9402 (+0.05 leap!)."
    )
    ps1_b.font.size = Pt(8.5)
    ps1_b.font.color.rgb = C_SLATE_TEXT
    ps1_b.space_before = Pt(3)

    card_sol2 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, rx, Inches(3.22), rw, Inches(1.75))
    card_sol2.fill.solid()
    card_sol2.fill.fore_color.rgb = C_WHITE
    card_sol2.line.color.rgb = RGBColor(165, 180, 252)
    card_sol2.line.width = Pt(1.2)

    tb_s2 = slide3.shapes.add_textbox(rx + Inches(0.15), Inches(3.32), rw - Inches(0.3), Inches(1.55))
    tf_s2 = tb_s2.text_frame
    tf_s2.word_wrap = True
    ps2_h = tf_s2.paragraphs[0]
    ps2_h.text = "2. Resolution: 0.365 mm/px Physical Grid"
    ps2_h.font.size = Pt(11)
    ps2_h.font.bold = True
    ps2_h.font.color.rgb = RGBColor(79, 70, 229)

    ps2_b = tf_s2.add_paragraph()
    ps2_b.text = (
        "• Standard 224px squash caused 0.71 mm/px distortion, losing subtle 0.36mm meniscus lesions.\n"
        "• High-Res (E9) preserved exact 140mm anatomy on 384px grid with 3.5mm slab depth.\n"
        "• Meniscus ROC-AUC improved by +0.033; ACL tears clearly resolved."
    )
    ps2_b.font.size = Pt(8.5)
    ps2_b.font.color.rgb = C_SLATE_TEXT
    ps2_b.space_before = Pt(3)

    card_sol3 = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, rx, Inches(5.09), rw, Inches(1.75))
    card_sol3.fill.solid()
    card_sol3.fill.fore_color.rgb = C_WHITE
    card_sol3.line.color.rgb = RGBColor(125, 211, 252)
    card_sol3.line.width = Pt(1.2)

    tb_s3 = slide3.shapes.add_textbox(rx + Inches(0.15), Inches(5.19), rw - Inches(0.3), Inches(1.55))
    tf_s3 = tb_s3.text_frame
    tf_s3.word_wrap = True
    ps3_h = tf_s3.paragraphs[0]
    ps3_h.text = "3. Latency: Single-Decode In-Pass Cache"
    ps3_h.font.size = Pt(11)
    ps3_h.font.bold = True
    ps3_h.font.color.rgb = RGBColor(2, 132, 199)

    ps3_b = tf_s3.add_paragraph()
    ps3_b.text = (
        "• Re-decoding 100+ DICOM slices per model exceeded Kaggle's 9-hour hidden test deadline.\n"
        "• Single-decode in-pass architecture decodes once into RAM; feeds 384px & 320px in one pass.\n"
        "• Cut inference latency by 42% (1.4s/study), completely preventing test cutoffs."
    )
    ps3_b.font.size = Pt(8.5)
    ps3_b.font.color.rgb = C_SLATE_TEXT
    ps3_b.space_before = Pt(3)

    # =========================================================================
    # SLIDE 4: Technical Depth: End-to-End System Architecture
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "End-to-End System Architecture & 2.5D MIL Formulation", "3. Deep Technical Architecture", 4)

    ARCH_IMG = os.path.abspath('assets/diagram_architecture.png')
    if os.path.exists(ARCH_IMG):
        slide4.shapes.add_picture(ARCH_IMG, Inches(0.8), Inches(1.35), width=Inches(11.733))

    b_card_w = Inches(5.75)
    b_card_h = Inches(1.6)
    b_card_y = Inches(5.35)

    b_card1 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), b_card_y, b_card_w, b_card_h)
    b_card1.fill.solid()
    b_card1.fill.fore_color.rgb = C_WHITE
    b_card1.line.color.rgb = C_BORDER_LIGHT
    b_card1.line.width = Pt(1)

    tb_bc1 = slide4.shapes.add_textbox(Inches(0.95), b_card_y + Inches(0.1), b_card_w - Inches(0.3), b_card_h - Inches(0.2))
    tf_bc1 = tb_bc1.text_frame
    tf_bc1.word_wrap = True
    pbc1_h = tf_bc1.paragraphs[0]
    pbc1_h.text = "Mathematical Formulation: 2.5D Multi-Instance Learning"
    pbc1_h.font.size = Pt(10.5)
    pbc1_h.font.bold = True
    pbc1_h.font.color.rgb = C_DU_PURPLE

    pbc1_b = tf_bc1.add_paragraph()
    pbc1_b.text = (
        "• S = 3 series slots × K = 8-12 windows with 3-slice slabs (c-s, c, c+s, ~3.5mm slab depth).\n"
        "• Window Token: t_{s,k} = f_{s,k} + e_slot[s] + MLP_pos(p_{s,k}) where p is physical slice position.\n"
        "• Finding Softmax Attention: α_{c,j} = softmax(W₂ tanh(W₁ t_j + b₁) + b₂) dynamically routes attention."
    )
    pbc1_b.font.size = Pt(8.5)
    pbc1_b.font.color.rgb = C_SLATE_TEXT
    pbc1_b.space_before = Pt(3)

    b_card2 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.783), b_card_y, b_card_w, b_card_h)
    b_card2.fill.solid()
    b_card2.fill.fore_color.rgb = C_WHITE
    b_card2.line.color.rgb = C_BORDER_LIGHT
    b_card2.line.width = Pt(1)

    tb_bc2 = slide4.shapes.add_textbox(Inches(6.933), b_card_y + Inches(0.1), b_card_w - Inches(0.3), b_card_h - Inches(0.2))
    tf_bc2 = tb_bc2.text_frame
    tf_bc2.word_wrap = True
    pbc2_h = tf_bc2.paragraphs[0]
    pbc2_h.text = "Heterogeneous Encoders & Cross-Plane Evidence Fusion"
    pbc2_h.font.size = Pt(10.5)
    pbc2_h.font.bold = True
    pbc2_h.font.color.rgb = C_DU_PURPLE

    pbc2_b = tf_bc2.add_paragraph()
    pbc2_b.text = (
        "• CoAtNet-384: Combines depthwise convolutions (local features) + self-attention (global context).\n"
        "• ConvNeXt (S6/s6ft): Multi-stage 7x7 depthwise convolutions with LayerNorm and stochastic depth.\n"
        "• Logit Synthesis: z_c = Σ_j α_{c,j} ⟨t_j, w_c⟩ + b_c aggregates cross-plane evidence without saturation."
    )
    pbc2_b.font.size = Pt(8.5)
    pbc2_b.font.color.rgb = C_SLATE_TEXT
    pbc2_b.space_before = Pt(3)

    # =========================================================================
    # SLIDE 5: Technical Depth: Specialist Distillation & Routing Matrix
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "Breakthrough: Pathology-Specific Routing Matrix & Calibration", "4. Advanced Model Ensembling", 5)

    ROUTING_IMG = os.path.abspath('assets/diagram_specialist_routing.png')
    if os.path.exists(ROUTING_IMG):
        slide5.shapes.add_picture(ROUTING_IMG, Inches(0.8), Inches(1.35), width=Inches(6.5))

    rx5 = Inches(7.5)
    rw5 = Inches(5.033)

    b1 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, rx5, Inches(1.35), rw5, Inches(1.7))
    b1.fill.solid()
    b1.fill.fore_color.rgb = C_WHITE
    b1.line.color.rgb = RGBColor(253, 164, 175)
    b1.line.width = Pt(1.2)

    tb_b1 = slide5.shapes.add_textbox(rx5 + Inches(0.15), Inches(1.45), rw5 - Inches(0.3), Inches(1.5))
    tf_b1 = tb_b1.text_frame
    tf_b1.word_wrap = True
    pb1_h = tf_b1.paragraphs[0]
    pb1_h.text = "Why Uniform Model Averaging Fails"
    pb1_h.font.size = Pt(11)
    pb1_h.font.bold = True
    pb1_h.font.color.rgb = C_ROSE

    pb1_b = tf_b1.add_paragraph()
    pb1_b.text = (
        "• Naive ensembling uses fixed scalar weights (e.g. 0.5/0.5) across all findings.\n"
        "• However, different neural architectures specialize in distinct tissue types:\n"
        "  - Pure-Vision S6 excels on fluid & inflammation (Effusion: 0.979, Synovitis: 0.860).\n"
        "  - CoAtNet & R6 excel on complex structural bone/OA (Medial OA: 0.991, ACL: 0.980).\n"
        "• Averaging them uniformly dilutes the strongest specialist signals."
    )
    pb1_b.font.size = Pt(8.5)
    pb1_b.font.color.rgb = C_SLATE_TEXT
    pb1_b.space_before = Pt(2)

    b2 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, rx5, Inches(3.2), rw5, Inches(1.9))
    b2.fill.solid()
    b2.fill.fore_color.rgb = C_WHITE
    b2.line.color.rgb = C_DU_PURPLE_BORDER
    b2.line.width = Pt(1.2)

    tb_b2 = slide5.shapes.add_textbox(rx5 + Inches(0.15), Inches(3.3), rw5 - Inches(0.3), Inches(1.7))
    tf_b2 = tb_b2.text_frame
    tf_b2.word_wrap = True
    pb2_h = tf_b2.paragraphs[0]
    pb2_h.text = "The Pathology-Adaptive Routing Solution"
    pb2_h.font.size = Pt(11)
    pb2_h.font.bold = True
    pb2_h.font.color.rgb = C_DU_PURPLE

    pb2_b = tf_b2.add_paragraph()
    pb2_b.text = (
        "• Finding-Specific Allocation Matrix: Dynamically maps the 12 abnormalities to optimal weights based on Gold-58 cross-validation.\n"
        "• Fluid / Inflammation: 45% weight to s6ft_specialist_ema.\n"
        "• Ligament / Meniscal Tears: 40% weight to s6ft_v1_ema (MCL: 0.986).\n"
        "• Degenerative / Bone: 65% weight to R6 + CoAtNet-384 Vision SWA.\n"
        "• Temperature Logit Calibration: Logits are standardized before sigmoid activation: p_c = σ(Σ w_{m,c} (z_{m,c} - μ) / τ_c)."
    )
    pb2_b.font.size = Pt(8.5)
    pb2_b.font.color.rgb = C_SLATE_TEXT
    pb2_b.space_before = Pt(2)

    b3 = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, rx5, Inches(5.25), rw5, Inches(1.6))
    b3.fill.solid()
    b3.fill.fore_color.rgb = C_GREEN_BG
    b3.line.color.rgb = RGBColor(110, 231, 183)
    b3.line.width = Pt(1.5)

    tb_b3 = slide5.shapes.add_textbox(rx5 + Inches(0.15), Inches(5.35), rw5 - Inches(0.3), Inches(1.4))
    tf_b3 = tb_b3.text_frame
    tf_b3.word_wrap = True
    pb3_h = tf_b3.paragraphs[0]
    pb3_h.text = "EMPIRICAL BREAKTHROUGH: +0.0153 MACRO AUC"
    pb3_h.font.size = Pt(11)
    pb3_h.font.bold = True
    pb3_h.font.color.rgb = C_GREEN

    pb3_b = tf_b3.add_paragraph()
    pb3_b.text = (
        "• Standard Baseline Blend on Gold-58:  0.9391 Macro AUC\n"
        "• Specialist Routing Matrix:               0.9544 Macro AUC (+0.0153 LEAP!)\n"
        "• Upgrades registered across 11 out of 12 individual finding heads."
    )
    pb3_b.font.size = Pt(8.5)
    pb3_b.font.bold = True
    pb3_b.font.color.rgb = RGBColor(6, 78, 59)
    pb3_b.space_before = Pt(3)

    # =========================================================================
    # SLIDE 6: Quantitative Results & Leaderboard Trajectory
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "Experimental Results & Leaderboard Performance", "5. Empirical Validation", 6)

    RESULTS_IMG = os.path.abspath('assets/chart_results_trajectory.png')
    if os.path.exists(RESULTS_IMG):
        slide6.shapes.add_picture(RESULTS_IMG, Inches(0.8), Inches(1.35), width=Inches(7.7))

    rx6 = Inches(8.7)
    rw6 = Inches(3.833)

    card_perf = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, rx6, Inches(1.35), rw6, Inches(2.95))
    card_perf.fill.solid()
    card_perf.fill.fore_color.rgb = C_WHITE
    card_perf.line.color.rgb = C_BORDER_LIGHT
    card_perf.line.width = Pt(1.2)

    tb_p = slide6.shapes.add_textbox(rx6 + Inches(0.15), Inches(1.45), rw6 - Inches(0.3), Inches(2.75))
    tf_p = tb_p.text_frame
    tf_p.word_wrap = True
    pp_h = tf_p.paragraphs[0]
    pp_h.text = "DIAGNOSTIC ACCURACY HIGHLIGHTS"
    pp_h.font.size = Pt(10.5)
    pp_h.font.bold = True
    pp_h.font.color.rgb = C_DU_PURPLE

    results_table_data = [
        ("Medial Osteoarthritis", "0.991 AUC", "Top Tier"),
        ("Baker's Cyst", "0.989 AUC", "Top Tier"),
        ("MCL Tear", "0.986 AUC", "Top Tier"),
        ("ACL Tear", "0.980 AUC", "Top Tier"),
        ("Joint Effusion", "0.979 AUC", "Top Tier"),
        ("Medial Meniscus", "0.972 AUC", "Top Tier"),
        ("Bone Contusion", "0.964 AUC", "Strong"),
        ("Lateral Meniscus", "0.942 AUC", "Strong"),
        ("Bone Fracture", "0.922 AUC", "Calibrated")
    ]
    for name, score, tier in results_table_data:
        p_row = tf_p.add_paragraph()
        p_row.text = f"• {name}:  {score}"
        p_row.font.size = Pt(8.5)
        p_row.font.color.rgb = C_NAVY_DARK
        p_row.font.bold = True if "0.98" in score or "0.99" in score else False
        p_row.space_before = Pt(1.5)

    card_lb = slide6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, rx6, Inches(4.45), rw6, Inches(2.45))
    card_lb.fill.solid()
    card_lb.fill.fore_color.rgb = C_DU_PURPLE_BG
    card_lb.line.color.rgb = C_DU_PURPLE_BORDER
    card_lb.line.width = Pt(1.5)

    tb_lb = slide6.shapes.add_textbox(rx6 + Inches(0.15), Inches(4.55), rw6 - Inches(0.3), Inches(2.25))
    tf_lb = tb_lb.text_frame
    tf_lb.word_wrap = True
    plb_h = tf_lb.paragraphs[0]
    plb_h.text = "LEADERBOARD & COMPETITION STANDING"
    plb_h.font.size = Pt(10.5)
    plb_h.font.bold = True
    plb_h.font.color.rgb = C_DU_PURPLE

    plb_b = tf_lb.add_paragraph()
    plb_b.text = (
        "★ Final Public Score: 0.958 Macro ROC-AUC\n"
        "★ Competitive Standing: Rank ~420 out of 6,290+ international teams (Top ~6%).\n"
        "★ Medal Zone: Solidly inside the official Silver / Bronze Medal tier.\n"
        "★ Strict Rules Compliance: 100% verified private checkpoints, zero Gold-58 leak, zero execution timeouts."
    )
    plb_b.font.size = Pt(8.5)
    plb_b.font.color.rgb = C_NAVY_DARK
    plb_b.space_before = Pt(3)

    # =========================================================================
    # SLIDE 7: Key Contributions, Clinical Impact & Future Work
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "Key Contributions, Clinical Impact & Future Work", "6. Conclusions & Outlook", 7)

    grid_w = Inches(5.75)
    grid_h = Inches(2.4)
    gx1 = Inches(0.8)
    gx2 = Inches(6.783)
    gy1 = Inches(1.35)
    gy2 = Inches(3.95)

    pillars = [
        (
            gx1, gy1,
            "1. Pure-Vision Target Distillation",
            "Curing Clinical Report Omission Noise",
            C_WHITE, C_DU_PURPLE_BORDER, C_DU_PURPLE,
            [
                "Demonstrated that standard NLP-extracted clinical report labels contain severe false-negative omission bias (e.g. Synovitis 0.69).",
                "Developed an out-of-fold vision distillation protocol across 4,349 studies, bootstrapping ground-truth alignment from 0.890 to 0.9402.",
                "Provides a general methodology for medical datasets where radiologist text reports omit incidental findings."
            ]
        ),
        (
            gx2, gy1,
            "2. Single-Decode In-Pass Runtime Architecture",
            "Real-World SLA & Latency Optimization",
            C_WHITE, RGBColor(125, 211, 252), RGBColor(2, 132, 199),
            [
                "Engineered a memory-mapped single-decode pipeline eliminating redundant 3D DICOM disk I/O.",
                "Simultaneously renders 384px and 320px multi-scale views to 4 deep backbones within a single worker pass.",
                "Cut per-study latency by 42% (1.4s/study), guaranteeing complete immunity to hidden-test 9-hour cutoffs."
            ]
        ),
        (
            gx2, gy2,
            "3. Multi-View 2.5D MIL & Specialist Matrix",
            "Pathology-Adaptive Neural Weighting",
            C_WHITE, RGBColor(165, 180, 252), RGBColor(79, 70, 229),
            [
                "Overcame 3D GPU memory limits via 3-slice 2.5D slab sampling across Sagittal, Coronal, and Axial planes.",
                "Devised finding-specific softmax attention pooling, allowing models to focus on individual damaged slices.",
                "Replaced naive scalar ensembling with a calibrated specialist matrix, driving a +0.0153 Macro AUC leap."
            ]
        ),
        (
            gx1, gy2,
            "4. Clinical Translation & Future Directions",
            "Point-of-Care Radiologist Decision Support",
            C_GREEN_BG, RGBColor(110, 231, 183), C_GREEN,
            [
                "Hospital PACS Integration: Sub-second inference enables real-time background triage and worklist prioritization in emergency wards.",
                "Visual Explainability: Attention weights directly localize damaged slices for radiologist review and audit.",
                "Next Horizons: Extending 2.5D MIL to 3D Sparse Voxel Transformers and self-supervised foundation encoders."
            ]
        )
    ]

    for x, y, title, sub, bg_c, border_c, head_c, bullets in pillars:
        p_card = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, grid_w, grid_h)
        p_card.fill.solid()
        p_card.fill.fore_color.rgb = bg_c
        p_card.line.color.rgb = border_c
        p_card.line.width = Pt(1.5)

        tb_pil = slide7.shapes.add_textbox(x + Inches(0.18), y + Inches(0.12), grid_w - Inches(0.36), grid_h - Inches(0.24))
        tf_pil = tb_pil.text_frame
        tf_pil.word_wrap = True
        
        p_h = tf_pil.paragraphs[0]
        p_h.text = title
        p_h.font.size = Pt(11)
        p_h.font.bold = True
        p_h.font.color.rgb = head_c

        p_s = tf_pil.add_paragraph()
        p_s.text = sub.upper()
        p_s.font.size = Pt(8)
        p_s.font.bold = True
        p_s.font.color.rgb = C_SLATE_MUTED

        for b_idx, bullet in enumerate(bullets):
            p_b = tf_pil.add_paragraph()
            p_b.text = f"• {bullet}"
            p_b.font.size = Pt(8.5)
            p_b.font.color.rgb = C_SLATE_TEXT
            p_b.space_before = Pt(3)

    close_bar = slide7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.45), Inches(11.733), Inches(0.55))
    close_bar.fill.solid()
    close_bar.fill.fore_color.rgb = C_WHITE
    close_bar.line.color.rgb = C_BORDER_LIGHT
    close_bar.line.width = Pt(1)

    tb_close = slide7.shapes.add_textbox(Inches(1.0), Inches(6.52), Inches(11.3), Inches(0.4))
    tf_close = tb_close.text_frame
    tf_close.word_wrap = True
    p_cl = tf_close.paragraphs[0]
    p_cl.text = "Thank you!  •  Questions & Discussion  •  Department of Electronics and Communication, University of Delhi"
    p_cl.alignment = PP_ALIGN.CENTER
    p_cl.font.size = Pt(10.5)
    p_cl.font.bold = True
    p_cl.font.color.rgb = C_DU_PURPLE

    output_pptx = os.path.abspath('d:/Coding/kaggle/RSNA/Knee_Abnormality_Detection_Presentation.pptx')
    prs.save(output_pptx)
    print(f"Presentation created successfully at: {output_pptx}")

if __name__ == '__main__':
    create_presentation()
