"""Generate the new architecture diagram (SVG) in the style of the old one."""
import html

W, H = 1400, 810
out = []
A = out.append

FONT = "Helvetica, Arial, sans-serif"
CHIP = "#a9c3e3"      # blue step labels
POOL = "#7f7f7f"      # work pool (dark grey)
GREYBOX = "#d9d9d9"   # worker / server boxes
JOB = "#f8dccb"       # job (peach)
EDGE = "#4472c4"      # arrow blue


def esc(s):
    return html.escape(s)


def text(x, y, s, size=15, weight="normal", fill="#000", anchor="middle", italic=False):
    st = ' font-style="italic"' if italic else ""
    A(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" '
      f'fill="{fill}" text-anchor="{anchor}"{st}>{esc(s)}</text>')


def box(x, y, w, h, fill, lines, stroke="#404040", tfill="#000", bold_first=True, size=15, rx=3, sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
    n = len(lines)
    lh = size + 4
    y0 = y + h / 2 - (n - 1) * lh / 2 + size * 0.35
    for i, ln in enumerate(lines):
        text(x + w / 2, y0 + i * lh, ln, size=size if i == 0 else size - 2,
             weight="bold" if (i == 0 and bold_first) else "normal", fill=tfill)


def chip(x, y, s, w=None, size=14, lines=None):
    lines = lines or [s]
    w = w or max(len(ln) for ln in lines) * size * 0.56 + 16
    h = len(lines) * (size + 4) + 8
    A(f'<rect x="{x - w / 2}" y="{y - h / 2}" width="{w}" height="{h}" fill="{CHIP}"/>')
    for i, ln in enumerate(lines):
        text(x, y - h / 2 + 4 + (i + 1) * (size + 4) - 5, ln, size=size)


def panel(x, y, w, h, fill, header, hfill, hx=None):
    A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="{fill}"/>')
    hw = len(header) * 15 * 0.56 + 22
    hx = x + 18 if hx is None else hx
    A(f'<rect x="{hx}" y="{y - 14}" width="{hw}" height="28" fill="{hfill}"/>')
    text(hx + hw / 2, y + 5, header, size=16)


def arrow(d, dash=False, color=EDGE, both=False):
    da = ' stroke-dasharray="7 5"' if dash else ""
    ms = ' marker-start="url(#ahs)"' if both else ""
    A(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{da} marker-end="url(#ah)"{ms}/>')


def badge(x, y, s="NEW"):
    A(f'<rect x="{x}" y="{y}" width="44" height="20" rx="10" fill="#c00000"/>')
    text(x + 22, y + 14.5, s, size=12, weight="bold", fill="#fff")


def check(x, y):
    A(f'<path d="M{x} {y} l12 12 l24 -26" fill="none" stroke="#2e9e44" stroke-width="7" '
      f'stroke-linecap="round" stroke-linejoin="round"/>')


A(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
A('<defs>'
  f'<marker id="ah" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{EDGE}"/></marker>'
  f'<marker id="ahs" markerWidth="10" markerHeight="10" refX="2" refY="5" orient="auto"><path d="M10,0 L0,5 L10,10 z" fill="{EDGE}"/></marker>'
  '</defs>')
A(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
text(W / 2, 34, "New architecture: Job API replaces the parent worker (example: fine-tune job on NERSC)",
     size=18, weight="bold")

# ---------------- Application ----------------
A('<rect x="20" y="120" width="210" height="220" fill="#fdf3d6"/>')
A('<rect x="30" y="106" width="120" height="28" fill="#f7d45c"/>')
text(90, 126, "Application", size=16)
# 3D-ish app box
A('<polygon points="52,170 62,160 212,160 202,170" fill="#f3e6a6" stroke="#404040"/>')
A('<polygon points="202,170 212,160 212,290 202,300" fill="#e9d98a" stroke="#404040"/>')
box(52, 170, 150, 130, "#fbf0bf", ["Segmentation /", "Annotation app"], bold_first=False, size=16, rx=0)
text(127, 322, "no Prefect · no PyTorch", size=12, fill="#555", italic=True)

# ---------------- Cloud ----------------
A('<rect x="290" y="70" width="610" height="660" rx="60" fill="#edf1f8"/>')
A('<rect x="520" y="56" width="150" height="28" fill="#8faadc"/>')
text(595, 76, "Cloud / ALS", size=16)

box(330, 150, 220, 80, "#fff2cc", ["Job API (FastAPI)", "replaces parent worker", "validate · route · status"],
    stroke="#bf9000", size=16, sw=2)
badge(516, 140)
box(680, 160, 190, 60, GREYBOX, ["Prefect Server"], size=17)
box(330, 330, 220, 60, GREYBOX, ["MLflow Model Registry", "metadata · wrapped models"], size=15)
box(330, 565, 220, 90, "#e7e0f6", ["Inference Service", "MLflow model server", "(wrapped models) · GPU"],
    stroke="#7e6bc4", size=16, sw=2)
badge(516, 555)

# removed parts (ghost)
box(605, 470, 270, 110, "#f4f4f4", ["Removed vs. old design", "Parent Work Pool", "parent Worker", "Job: launch_parent_flow"],
    stroke="#9a9a9a", tfill="#8a8a8a", size=14, dash="6 4")
A('<path d="M590 455 l26 26 M616 455 l-26 26" stroke="#e03030" stroke-width="7" stroke-linecap="round"/>')

# ---------------- Server 1: ALS ----------------
panel(950, 80, 420, 190, "#eef5e8", "Server 1 · ALS", "#b6d7a8")
box(980, 140, 175, 55, POOL, ["Child Work Pool", "(docker_pool)"], tfill="#fff", size=15, stroke="#555")
box(1235, 140, 115, 55, GREYBOX, ["Worker"], bold_first=False, size=16)
box(1195, 220, 165, 36, JOB, ["Job: launch_docker"], bold_first=False, size=15, stroke="#e0b9a0")
arrow("M1235 167 L1158 167")
text(1196, 160, "polls", size=12)
arrow("M1292 195 L1292 218")
A('<path d="M906 128 l26 26 M932 128 l-26 26" stroke="#e03030" stroke-width="7" stroke-linecap="round"/>')

# ---------------- Server 2: SFAPI worker ----------------
panel(950, 305, 420, 200, "#eef5e8", "Server 2 · SFAPI worker", "#b6d7a8")
box(980, 355, 175, 55, POOL, ["Child Work Pool", "(sfapi_pool)"], tfill="#fff", size=15, stroke="#555")
box(1235, 355, 115, 55, GREYBOX, ["Worker"], bold_first=False, size=16)
box(1195, 448, 165, 36, JOB, ["Job: launch_sfapi"], bold_first=False, size=15, stroke="#e0b9a0")
text(1060, 497, "runs anywhere with HTTPS to NERSC", size=12, fill="#555", italic=True)
check(905, 350)

# ---------------- NERSC ----------------
panel(950, 545, 420, 140, "#e6eefa", "NERSC Perlmutter", "#8faadc")
box(985, 600, 350, 60, "#ffffff", ["Slurm job (fine-tuning)", "podman-hpc container · GPU"], size=15, stroke="#4472c4")

# ---------------- Arrows + step chips ----------------
# 1. App -> Job API
arrow("M202 188 L328 188")
chip(266, 172, "1. POST /jobs", size=13)
# 10. status
arrow("M202 214 L328 214", dash=True)
chip(266, 232, "8. GET status", size=13)
# 2. Job API <-> MLflow
arrow("M440 232 L440 328", both=True)
chip(512, 280, "2. resolve model", size=13)
# 3. Job API -> Prefect
arrow("M552 190 L678 190")
chip(615, 172, "3. create run", size=13)
# 4. Prefect -> sfapi_pool
arrow("M872 205 L925 205 L925 382 L978 382")
chip(925, 262, "4. Assign to", size=13)
# 5./6. polls / starts on server 2
arrow("M1235 382 L1158 382")
chip(1196, 368, "5. polls", size=12)
arrow("M1292 410 L1292 446")
chip(1316, 428, "6. starts", size=12)
# 7. launch_sfapi -> NERSC
arrow("M1250 486 L1250 598")
chip(1250, 532, "7. submit & wait (SFAPI / IRI)", size=12)
# A. interactive inference
arrow("M127 302 L127 610 L328 610")
chip(127, 450, "", lines=["A. interactive", "inference"], size=13)
chip(228, 595, "POST /invocations", size=12)
# B. inference loads model
arrow("M440 563 L440 392")
chip(505, 478, "B. load model", size=13)

# ---------------- Legend ----------------
text(20, 758, "1–8: job path (training / batch inference, Prefect-orchestrated). One job → one target → ONE child work pool", size=13, anchor="start")
text(20, 778, "A–B: interactive inference path (no Prefect)", size=13, anchor="start")
text(20, 798, "✗ = pool not selected for this job. Multi-step jobs (train → inference) stay in the same pool. Not shown: conda · podman · slurm pools", size=13, anchor="start", fill="#555")

A("</svg>")
open(__file__.replace("draw_architecture_new.py", "architecture_new.svg"), "w").write("\n".join(out))
print("ok")
