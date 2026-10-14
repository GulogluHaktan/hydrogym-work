#!/usr/bin/env python3
"""IYILESTIRME_ONERILERI.md -> PDF (uste sunmak icin).
Turkce karakterler icin DejaVu Sans (matplotlib TTF) reportlab'a gomulu.
"""
import os
import re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    ListFlowable, ListItem,
)

F = next(p for p in ("/usr/share/fonts/TTF/", "/usr/share/fonts/truetype/dejavu/")
         if os.path.isdir(p) and os.path.exists(p + "DejaVuSans.ttf"))
for name, fn in [("DejaVuSans", "DejaVuSans.ttf"),
                 ("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"),
                 ("DejaVuSans-Oblique", "DejaVuSans-Oblique.ttf"),
                 ("DejaVuSans-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
                 ("DejaVuSansMono", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, F + fn))
pdfmetrics.registerFontFamily("DejaVuSans", normal="DejaVuSans",
                              bold="DejaVuSans-Bold", italic="DejaVuSans-Oblique",
                              boldItalic="DejaVuSans-BoldOblique")

SRC = "IYILESTIRME_ONERILERI.md"
OUT = "HydroGym_Iyilestirme_Onerileri_HaktanGuloglu.pdf"

st = {
    "body": ParagraphStyle("body", fontName="DejaVuSans", fontSize=9.5,
                           leading=13, alignment=TA_JUSTIFY, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="DejaVuSans-Bold", fontSize=15,
                         leading=18, spaceBefore=16, spaceAfter=8,
                         textColor=colors.HexColor("#1F4E78")),
    "h2": ParagraphStyle("h2", fontName="DejaVuSans-Bold", fontSize=11.5,
                         leading=14, spaceBefore=11, spaceAfter=5,
                         textColor=colors.HexColor("#2c3e50")),
    "h3": ParagraphStyle("h3", fontName="DejaVuSans-Bold", fontSize=10,
                         leading=12.5, spaceBefore=7, spaceAfter=3),
    "title": ParagraphStyle("title", fontName="DejaVuSans-Bold", fontSize=17,
                            leading=21, spaceAfter=6, alignment=TA_CENTER),
    "sub": ParagraphStyle("sub", fontName="DejaVuSans-Oblique", fontSize=10,
                          textColor=colors.HexColor("#555"), spaceAfter=12,
                          alignment=TA_CENTER),
    "cell": ParagraphStyle("cell", fontName="DejaVuSans", fontSize=7.8,
                           leading=9.8, alignment=TA_LEFT),
    "cellh": ParagraphStyle("cellh", fontName="DejaVuSans-Bold", fontSize=7.8,
                            leading=9.8, alignment=TA_LEFT, textColor=colors.white),
    "li": ParagraphStyle("li", fontName="DejaVuSans", fontSize=9.5, leading=13,
                         alignment=TA_JUSTIFY, spaceAfter=3),
}

PAGE_W, _ = A4
MARGIN = 1.7 * cm
CW = PAGE_W - 2 * MARGIN


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r'<font face="DejaVuSansMono" size="8.4">\1</font>', t)
    return t


def flush_table(rows):
    if not rows:
        return None
    header = [Paragraph(inline(c), st["cellh"]) for c in rows[0]]
    body = [[Paragraph(inline(c), st["cell"]) for c in r] for r in rows[1:]]
    ncol = len(rows[0])
    hdr = " ".join(rows[0]).lower()
    if "paket" in hdr:                      # A-F work-package table
        w = [CW * x for x in [0.04, 0.30, 0.10, 0.10, 0.32, 0.14]]
    elif ncol >= 6:                         # breaking-point table (wide, many cols)
        w = [CW * x for x in ([0.12] + [(0.88 / (ncol - 1))] * (ncol - 1))]
    else:
        w = [CW / ncol] * ncol
    t = Table([header] + body, colWidths=w, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef3f8")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def parse():
    lines = open(SRC, encoding="utf-8").read().splitlines()
    story = []
    i = 0
    first_h1 = True
    tbl = []
    bullets = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            story.append(ListFlowable(
                [ListItem(Paragraph(inline(b), st["li"]), leftIndent=12)
                 for b in bullets],
                bulletType="bullet", start="•", leftIndent=14))
            story.append(Spacer(1, 4))
            bullets = []

    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("|") and "|" in ln[1:]:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                i += 1
                continue
            tbl.append(cells)
            i += 1
            continue
        else:
            if tbl:
                flush_bullets()
                t = flush_table(tbl)
                if t:
                    story.append(t)
                    story.append(Spacer(1, 8))
                tbl = []
        if not ln.strip():
            flush_bullets()
            i += 1
            continue
        if ln.startswith("# "):
            flush_bullets()
            if first_h1:
                story.append(Paragraph(inline(ln[2:]), st["title"]))
                first_h1 = False
            else:
                story.append(Paragraph(inline(ln[2:]), st["h1"]))
        elif ln.startswith("## "):
            flush_bullets()
            story.append(Paragraph(inline(ln[3:]), st["h2"]))
        elif ln.startswith("### "):
            flush_bullets()
            story.append(Paragraph(inline(ln[4:]), st["h3"]))
        elif ln.strip() == "---":
            flush_bullets()
            story.append(Spacer(1, 3))
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color=colors.HexColor("#bbb")))
            story.append(Spacer(1, 3))
        elif re.match(r"^\s*[-*] ", ln):
            bullets.append(re.sub(r"^\s*[-*] ", "", ln))
        elif re.match(r"^\s*\*.+\*\s*$", ln) and i < 4:
            story.append(Paragraph(inline(ln.strip()), st["sub"]))
        else:
            flush_bullets()
            story.append(Paragraph(inline(ln), st["body"]))
        i += 1
    flush_bullets()
    if tbl:
        t = flush_table(tbl)
        if t:
            story.append(t)
    return story


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                        topMargin=1.4 * cm, bottomMargin=1.5 * cm,
                        title="HydroGym - Iyilestirme Onerileri - Haktan Guloglu")
doc.build(parse())
print("wrote", OUT)
