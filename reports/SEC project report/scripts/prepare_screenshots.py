"""Cut the workstation screenshots used in the report from the captures of capture_workstation.mjs (2x device scale).
    python prepare_screenshots.py <capture-folder> [--mask]
--mask covers every MRI canvas with a grey box (used for the first draft; the final report shows the images).
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SRC = Path(sys.argv[1])
MASK = "--mask" in sys.argv
FIG = Path(__file__).resolve().parent.parent / "figures"
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)


def load(name):
    im = Image.open(SRC / f"{name}.png").convert("RGB")
    if not MASK:
        return im
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1, _ in json.load(open(SRC / f"{name}.rects.json")):
        b = [round(v * 2) for v in (x0, y0, x1, y1)]
        d.rectangle(b, fill=(58, 58, 58))
        msg = "MRI image area masked\n(competition data licence)"
        w = d.multiline_textbbox((0, 0), msg, font=font, align="center")
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        d.multiline_text((cx - (w[2] - w[0]) / 2, cy - (w[3] - w[1]) / 2), msg, font=font, fill=(200, 200, 200), align="center")
    return im


full = load("ws_finding")
full.resize((2400, 1500), Image.LANCZOS).save(FIG / "ws_overview.png", optimize=True)
full.crop((0, 0, 592, 2000)).save(FIG / "ws_left_panel.png", optimize=True)
load("ws_detail").crop((2520, 180, 3200, 2000)).save(FIG / "ws_findings_detail.png", optimize=True)
print("wrote", sorted(p.name for p in FIG.glob("ws_*.png")), "masked" if MASK else "unmasked")
