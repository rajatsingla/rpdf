"""Embed the non-embedded base-14 Times-Roman in l.pdf.

Uses MuPDF's bundled URW Nimbus Roman (metrically identical to Times-Roman,
freely licensed for embedding) as a Type1C/CFF FontFile3, so glyph advances
and therefore the existing content-stream text positions are unchanged.
"""
import sys
import fitz

SRC = sys.argv[1] if len(sys.argv) > 1 else "l.pdf"
DST = sys.argv[2] if len(sys.argv) > 2 else "l_fixed.pdf"

doc = fitz.open(SRC)

# ---- WinAnsiEncoding code -> unicode -------------------------------------
def winansi(code):
    if code == 0xA0:            # NBSP renders as space
        return " "
    if code == 0xAD:            # soft hyphen renders as hyphen
        return "-"
    try:
        return bytes([code]).decode("cp1252")
    except UnicodeDecodeError:
        return None

targets = []
for xref in range(1, doc.xref_length()):
    try:
        obj = doc.xref_object(xref, compressed=True).replace(" ", "")
    except Exception:
        continue
    if "/Type/Font" not in obj:
        continue
    if "/FontDescriptor" in obj or "/DescendantFonts" in obj:
        continue            # already has (or delegates) a descriptor
    targets.append(xref)

if not targets:
    print("nothing to do: no unembedded simple fonts found")
    raise SystemExit(0)

for xref in targets:
    base = doc.xref_get_key(xref, "BaseFont")[1].lstrip("/")
    font = fitz.Font(base)                      # MuPDF base-14 substitute
    buf = bytes(font.buffer)
    assert buf[:2] == b"\x01\x00", f"{base}: expected bare CFF, got {buf[:4]!r}"

    widths = []
    for code in range(32, 256):
        ch = winansi(code)
        widths.append(round(font.glyph_advance(ord(ch)) * 1000) if ch else 0)

    # FontFile3 stream (bare CFF -> /Subtype /Type1C)
    ff = doc.get_new_xref()
    doc.update_object(ff, "<< /Subtype /Type1C >>")
    doc.update_stream(ff, buf, compress=True)

    bb = font.bbox
    x0, y0, x1, y1 = bb.x0, bb.y0, bb.x1, bb.y1
    flags = 32 | (1 if font.flags["mono"] else 0) | (2 if font.flags["serif"] else 0)

    fd = doc.get_new_xref()
    doc.update_object(fd, f"""<<
  /Type /FontDescriptor
  /FontName /{base}
  /Flags {flags}
  /FontBBox [{round(x0*1000)} {round(y0*1000)} {round(x1*1000)} {round(y1*1000)}]
  /ItalicAngle 0
  /Ascent {round(font.ascender*1000)}
  /Descent {round(font.descender*1000)}
  /CapHeight 662
  /StemV 84
  /FontFile3 {ff} 0 R
>>""")

    doc.xref_set_key(xref, "FirstChar", "32")
    doc.xref_set_key(xref, "LastChar", "255")
    doc.xref_set_key(xref, "Widths", "[" + " ".join(map(str, widths)) + "]")
    doc.xref_set_key(xref, "FontDescriptor", f"{fd} 0 R")
    print(f"embedded {base} (xref {xref}) via FontFile3 {ff}, descriptor {fd}")

doc.save(DST, garbage=3, deflate=True, clean=True)
print("wrote", DST)
