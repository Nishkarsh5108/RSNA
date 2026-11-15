"""Put the \\bibitem entries of chapters/references.tex in the order of first citation, read from the .aux file of
the last compile, and report keys that are cited but missing or listed but never cited.
    python order_references.py      (then compile twice more)
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
aux = (ROOT / "Knee_Abnormality_Detection_BTech_Report.aux").read_text(encoding="utf-8")
order = []
for group in re.findall(r"\\citation\{([^}]*)\}", aux):
    for key in group.split(","):
        key = key.strip()
        if key and key not in order:
            order.append(key)

ref = ROOT / "chapters" / "references.tex"
text = ref.read_text(encoding="utf-8")
start = text.index("\\bibitem{")
end = text.index("\\end{thebibliography}")
body = text[start:end]
items = {}
for chunk in re.split(r"(?=\\bibitem\{)", body):
    chunk = chunk.strip()
    if chunk:
        key = re.match(r"\\bibitem\{([^}]*)\}", chunk).group(1)
        items[key] = chunk

missing = [k for k in order if k not in items]
unused = [k for k in items if k not in order]
print("cited:", len(order), "listed:", len(items))
print("cited but not listed:", missing)
print("listed but not cited:", unused)
assert not missing, "every citation needs an entry"
new_body = "\n\n".join(items[k] for k in order if k in items) + "\n\n"
ref.write_text(text[:start] + new_body + text[end:], encoding="utf-8")
print("reordered; first five:", order[:5])
