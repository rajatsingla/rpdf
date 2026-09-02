"""Resize a PDF page to an exact trim size by stretching (no cropping),
and embed any non-embedded base-14 fonts.

Usage: python fix_page_size_and_fonts.py IN.pdf OUT.pdf [W_IN] [H_IN] [FIT]
                                         [PAD_TOP_IN] [PAD_BOTTOM_IN]

Resizing: a source rectangle is mapped onto the new page with a `cm` matrix
wrapped around the existing content streams. The scale is anisotropic (x and
y computed independently), so nothing is cropped or letterboxed -- the
artwork stretches to fill.

FIT picks the source rectangle:
  bleed (default)  MediaBox n CropBox n BleedBox -- i.e. the inked artwork,
                   dropping any blank slop left in the page box by an
                   earlier trim step. Gives an edge-to-edge cover.
  box              MediaBox n CropBox verbatim, blank margins and all.

PAD_TOP_IN / PAD_BOTTOM_IN reserve blank bands at the top and bottom of the
new page. The artwork is fitted into what is left, so padding trades
vertical stretch for whitespace rather than adding to the page height. The
artwork is clipped to its own rectangle so crop marks and bleed spill from
outside the source box cannot leak into the bands.

Font embedding: handled by embed_fonts.embed_missing_fonts -- any simple
font with no FontDescriptor (i.e. an unembedded base-14 reference) gets
MuPDF's bundled URW clone of that face embedded as a Type1C / FontFile3.
The URW clones are metrically identical to the Adobe originals, so glyph
advances -- and therefore the existing text positions -- are unchanged.
"""
import sys
import fitz

from embed_fonts import embed_missing_fonts

SRC = sys.argv[1]
DST = sys.argv[2]
W_IN = float(sys.argv[3]) if len(sys.argv) > 3 else 14.625
H_IN = float(sys.argv[4]) if len(sys.argv) > 4 else 10.75
FIT = sys.argv[5] if len(sys.argv) > 5 else "bleed"
assert FIT in ("bleed", "box"), FIT
PAD_T = float(sys.argv[6]) * 72.0 if len(sys.argv) > 6 else 0.0
PAD_B = float(sys.argv[7]) * 72.0 if len(sys.argv) > 7 else PAD_T

TARGET_W = W_IN * 72.0
TARGET_H = H_IN * 72.0
# the band of the new page the artwork actually occupies (PDF coords, y up)
ART_W = TARGET_W
ART_H = TARGET_H - PAD_T - PAD_B
assert ART_H > 0, "padding exceeds page height"

doc = fitz.open(SRC)


# --------------------------------------------------------------- resizing
def parse_box(s):
    return [float(v) for v in s.strip().lstrip("[").rstrip("]").split()]


for page in doc:
    pxref = page.xref

    # The visible area is CropBox intersected with MediaBox; fall back to
    # MediaBox when there is no CropBox.
    mb = parse_box(doc.xref_get_key(pxref, "MediaBox")[1])
    got, cb_raw = doc.xref_get_key(pxref, "CropBox")
    cb = parse_box(cb_raw) if got == "array" else mb
    boxes = [mb, cb]
    if FIT == "bleed":
        got, bl_raw = doc.xref_get_key(pxref, "BleedBox")
        if got == "array":
            boxes.append(parse_box(bl_raw))
    x0 = max(min(b[0], b[2]) for b in boxes)
    y0 = max(min(b[1], b[3]) for b in boxes)
    x1 = min(max(b[0], b[2]) for b in boxes)
    y1 = min(max(b[1], b[3]) for b in boxes)
    src_w, src_h = x1 - x0, y1 - y0

    sx = ART_W / src_w
    sy = ART_H / src_h
    tx, ty = -x0 * sx, PAD_B - y0 * sy

    print(f"page {page.number}: source ({FIT}) "
          f"[{x0:.3f} {y0:.3f} {x1:.3f} {y1:.3f}] = {src_w:.3f} x {src_h:.3f} pt "
          f"({src_w/72:.4f} x {src_h/72:.4f} in)")
    print(f"   -> page {TARGET_W:g} x {TARGET_H:g} pt ({W_IN} x {H_IN} in), "
          f"artwork {ART_W:g} x {ART_H:g} pt "
          f"({ART_W/72:.4f} x {ART_H/72:.4f} in)")
    if PAD_T or PAD_B:
        print(f"   white bands: {PAD_T:g} pt top ({PAD_T/72:g} in), "
              f"{PAD_B:g} pt bottom ({PAD_B/72:g} in)")
    print(f"   scale x{sx:.6f} / y{sy:.6f}  (aspect skew {sy/sx - 1:+.4%})")

    pre = doc.get_new_xref()
    doc.update_object(pre, "<<>>")
    # clip to the artwork band so crop marks / bleed spill lying outside the
    # source rectangle cannot render into the padding
    prefix = (f"q\n0 {PAD_B:.6f} {ART_W:.6f} {ART_H:.6f} re W n\n"
              f"{sx:.11f} 0 0 {sy:.11f} {tx:.6f} {ty:.6f} cm\n")
    doc.update_stream(pre, prefix.encode(), new=True)
    post = doc.get_new_xref()
    doc.update_object(post, "<<>>")
    doc.update_stream(post, b"\nQ\n", new=True)

    kind, contents = doc.xref_get_key(pxref, "Contents")
    inner = contents.strip()[1:-1] if kind == "array" else contents
    doc.xref_set_key(pxref, "Contents", f"[ {pre} 0 R {inner} {post} 0 R ]")

    box = f"[ 0 0 {TARGET_W:g} {TARGET_H:g} ]"
    for key in ("MediaBox", "CropBox", "TrimBox", "BleedBox", "ArtBox"):
        doc.xref_set_key(pxref, key, box)
    # stale InDesign private data describing the pre-resize geometry
    if doc.xref_get_key(pxref, "PieceInfo")[0] != "null":
        doc.xref_set_key(pxref, "PieceInfo", "null")


# --------------------------------------------------------- font embedding
embedded, skipped = embed_missing_fonts(doc)
for name in embedded:
    print(f"embedded {name}")
for name, reason in skipped:
    print(f"skipped {name}: {reason}")
if not embedded and not skipped:
    print("all fonts already embedded")

doc.save(DST, garbage=3, deflate=True, clean=True)
print("wrote", DST)
