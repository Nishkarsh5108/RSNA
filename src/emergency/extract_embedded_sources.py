"""Extract the embedded SOURCES dict (file name -> text) from a public notebook cell without executing it.
Handles both a plain dict literal and the zlib+base64-compressed form used by goodpjw2008 v27.
Used for plan 10c: R6, R7 and S6 trainer sources, hash-checked against each bundle manifest.
Usage: python -I src/emergency/extract_embedded_sources.py <ipynb> <cell_index> <out_dir>"""
import ast, base64, hashlib, json, os, sys, zlib
nb = json.load(open(sys.argv[1], encoding="utf-8"))
src = "".join(nb["cells"][int(sys.argv[2])]["source"])
tree = ast.parse(src)
found = None
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SOURCES" for t in node.targets):
        v = node.value
        try:
            found = ast.literal_eval(v)
        except ValueError:
            # _json.loads(_zlib.decompress(_b64.b64decode('...')))
            s = [n.value for n in ast.walk(v) if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) > 100]
            found = json.loads(zlib.decompress(base64.b64decode(s[0])))
os.makedirs(sys.argv[3], exist_ok=True)
for name, text in found.items():
    p = os.path.join(sys.argv[3], name)
    open(p, "w", encoding="utf-8", newline="").write(text)
    print(name, len(text), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
