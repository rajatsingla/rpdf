# Copyright (C) 2026 Rajat Singla <rajat@stck.me>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# embed_fonts.py
# pip install pymupdf
#
# Embed font programs for any simple font in a document that references one
# without carrying it (the classic "base-14 by name only" font dict, and the
# Arial/Times New Roman aliases producers write in its place).
#
# Printers reject unembedded fonts because the RIP has to substitute something,
# and what it picks - if it picks anything - is out of the file's control. The
# substitute embedded here is MuPDF's bundled URW clone of the base-14 face. The
# URW clones are metrically identical to the Adobe originals, so glyph advances,
# and therefore the text positions already baked into the content streams, are
# unchanged.
#
# Fonts that cannot be handled safely are left alone rather than guessed at:
# see _resolve_builtin and embed_missing_fonts for what is skipped and why.

import logging

import fitz  # PyMuPDF

log = logging.getLogger(__name__)

# The base-14 faces MuPDF ships a clone of, by family and style.
_HELVETICA = ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique",
              "Helvetica-BoldOblique")
_TIMES = ("Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic")
_COURIER = ("Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique")

# Family detection runs in order, on the normalised BaseFont name, so the first
# matching token wins. "Helv"/"Tiro"/"Cour" are PyMuPDF's short codes, which turn
# up in files produced by PyMuPDF itself.
_FAMILIES = (
    (("courier", "cour", "liberationmono", "nimbusmono"), _COURIER),
    (("helvetica", "helv", "arial", "liberationsans", "nimbussans"), _HELVETICA),
    (("times", "tiro", "liberationserif", "nimbusroman"), _TIMES),
)

_BOLD = ("bold", "black", "heavy", "semibold", "demibold")
_ITALIC = ("italic", "oblique")

# Symbolic base-14 faces. Their built-in encodings are their own, so WinAnsi
# widths cannot be synthesised for them (see embed_missing_fonts).
_SYMBOLIC = {"symbol": "Symbol", "zapfdingbats": "ZapfDingbats",
             "dingbats": "ZapfDingbats", "zadb": "ZapfDingbats"}

# Encodings whose code -> glyph mapping a width array can be built from, and the
# Python codec that reproduces it.
_ENCODING_CODECS = {
    "WinAnsiEncoding": "cp1252",
    "MacRomanEncoding": "mac_roman",
}


def _normalise(base_font: str) -> str:
    """
    Reduce a BaseFont name to a bare lowercase family+style string.

    Strips the six-letter subset prefix ("ABCDEF+Arial"), the PostScript suffixes
    producers append ("ArialMT", "TimesNewRomanPSMT"), and every separator, so
    "Arial,BoldItalic", "Arial-BoldItalicMT" and "ArialBoldItalic" all normalise
    to the same thing.
    """
    name = base_font.lstrip("/")
    if len(name) > 7 and name[6] == "+":
        name = name[7:]

    name = "".join(c for c in name.lower() if c.isalnum())
    for suffix in ("psmt", "psm", "ps", "mt"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    return name


def _resolve_builtin(base_font: str) -> str | None:
    """
    Map a BaseFont name onto the base-14 face to embed for it, or None when there
    is no safe answer.

    None means "leave this font alone": embedding a face whose metrics are not
    the ones the file was set with would move every glyph after the first, which
    is worse than the substitution the RIP would have done anyway. Only names
    that are a base-14 face, or a documented metric clone of one (Arial for
    Helvetica, Times New Roman for Times, Courier New for Courier, and the URW /
    Liberation clones of those), resolve.
    """
    name = _normalise(base_font)
    if not name:
        return None

    if name in _SYMBOLIC:
        return _SYMBOLIC[name]

    for tokens, faces in _FAMILIES:
        if not any(token in name for token in tokens):
            continue
        # "narrow"/"condensed" are not metric clones of the regular face.
        if "narrow" in name or "condensed" in name:
            return None
        bold = any(token in name for token in _BOLD)
        italic = any(token in name for token in _ITALIC)
        return faces[(1 if bold else 0) + (2 if italic else 0)]

    return None


def _codec(doc: fitz.Document, xref: int) -> str | None:
    """
    The Python codec matching the font's /Encoding, or None when widths cannot be
    synthesised for it.

    A font with no /Encoding uses its built-in one; for the text base-14 faces
    that is StandardEncoding, which differs from WinAnsi above code 127. Rather
    than guess, such a font is given /WinAnsiEncoding to match the widths written
    for it - see embed_missing_fonts, which is the only caller and only does this
    when the font carries no /Widths of its own, i.e. when nothing in the file
    depends on the old mapping.

    /Differences remaps individual codes, so a width array built from the base
    encoding would be wrong for exactly the codes the file bothered to remap.
    """
    kind, enc = doc.xref_get_key(xref, "Encoding")

    if kind == "null":
        return "cp1252"

    if kind == "name":
        return _ENCODING_CODECS.get(enc.lstrip("/"))

    if kind == "xref":
        enc_xref = int(enc.split()[0])
        if doc.xref_get_key(enc_xref, "Differences")[0] != "null":
            return None
        base_kind, base = doc.xref_get_key(enc_xref, "BaseEncoding")
        if base_kind == "null":
            return "cp1252"
        return _ENCODING_CODECS.get(base.lstrip("/"))

    return None


def _char(code: int, codec: str) -> str | None:
    """The character a byte maps to under ``codec``, or None if it maps to none."""
    if codec == "cp1252":
        if code == 0xA0:            # NBSP renders as space
            return " "
        if code == 0xAD:            # soft hyphen renders as hyphen
            return "-"
    try:
        return bytes([code]).decode(codec)
    except UnicodeDecodeError:
        return None


def _widths(font: fitz.Font, codec: str) -> list[int]:
    """Glyph advances for codes 32..255 under ``codec``, in 1/1000 em."""
    out = []
    for code in range(32, 256):
        ch = _char(code, codec)
        out.append(round(font.glyph_advance(ord(ch)) * 1000) if ch else 0)
    return out


def _unembedded_font_xrefs(doc: fitz.Document) -> list[int]:
    """
    Every simple font object in ``doc`` that carries no font program.

    A /FontDescriptor is what holds the program, so its absence is the marker.
    /DescendantFonts (a Type0 parent, whose descendant holds the descriptor) and
    /Type3 (whose glyphs are content streams in the file already) are not simple
    unembedded fonts and are left to their own devices.
    """
    out = []
    for xref in range(1, doc.xref_length()):
        try:
            obj = doc.xref_object(xref, compressed=True).replace(" ", "")
        except Exception:
            continue
        if "/Type/Font" not in obj:
            continue
        if "/FontDescriptor" in obj or "/DescendantFonts" in obj:
            continue
        if "/Subtype/Type3" in obj:
            continue
        out.append(xref)
    return out


def embed_missing_fonts(doc: fitz.Document) -> tuple[list[str], list[tuple[str, str]]]:
    """
    Embed a font program for every simple font in ``doc`` that references one by
    name only. Mutates ``doc``.

    Returns ``(embedded, skipped)``: the BaseFont names embedded, and the ones
    left alone as ``(name, reason)`` pairs. Skipping is not an error - it is the
    safe outcome for a font whose metrics cannot be reproduced (see
    _resolve_builtin) or whose code -> glyph mapping cannot be read (see _codec).

    An existing /Widths array is never overwritten: it, not the embedded program,
    is what a viewer spaces the text with, so the file's own numbers are kept.
    Widths are synthesised only for a font that has none, which a standard-14
    font may legitimately do - but once a program is embedded the font is no
    longer standard-14, and the array becomes required.
    """
    embedded: list[str] = []
    skipped: list[tuple[str, str]] = []

    for xref in _unembedded_font_xrefs(doc):
        kind, base_font = doc.xref_get_key(xref, "BaseFont")
        if kind != "name":
            skipped.append((f"xref {xref}", "no BaseFont"))
            continue

        base_font = base_font.lstrip("/")
        builtin = _resolve_builtin(base_font)
        if builtin is None:
            skipped.append((base_font, "no metric-compatible base-14 face"))
            continue

        has_widths = doc.xref_get_key(xref, "Widths")[0] != "null"
        codec = None
        if not has_widths:
            if builtin in ("Symbol", "ZapfDingbats"):
                # Built-in symbolic encoding: a WinAnsi width array would be
                # nonsense, and a font with neither is unusable.
                skipped.append((base_font, "symbolic font without Widths"))
                continue
            codec = _codec(doc, xref)
            if codec is None:
                skipped.append((base_font, "cannot synthesise Widths for encoding"))
                continue

        try:
            font = fitz.Font(builtin)
            buf = bytes(font.buffer)
        except Exception as exc:
            skipped.append((base_font, f"no builtin {builtin}: {exc}"))
            continue

        if buf[:2] != b"\x01\x00":
            # Not a bare CFF, so /Subtype /Type1C would be a lie about the stream.
            skipped.append((base_font, f"{builtin} is not bare CFF"))
            continue

        widths = None if has_widths else _widths(font, codec)

        program = doc.get_new_xref()
        doc.update_object(program, "<< /Subtype /Type1C >>")
        doc.update_stream(program, buf, new=True, compress=True)

        bb = font.bbox
        flags = (
            (1 if font.flags["mono"] else 0)
            | (2 if font.flags["serif"] else 0)
            | (4 if builtin in ("Symbol", "ZapfDingbats") else 32)
        )

        descriptor = doc.get_new_xref()
        doc.update_object(descriptor, f"""<<
  /Type /FontDescriptor
  /FontName /{base_font}
  /Flags {flags}
  /FontBBox [{round(bb.x0 * 1000)} {round(bb.y0 * 1000)} {round(bb.x1 * 1000)} {round(bb.y1 * 1000)}]
  /ItalicAngle {-12 if font.flags["italic"] else 0}
  /Ascent {round(font.ascender * 1000)}
  /Descent {round(font.descender * 1000)}
  /CapHeight 662
  /StemV 84
  /FontFile3 {program} 0 R
>>""")
        doc.xref_set_key(xref, "FontDescriptor", f"{descriptor} 0 R")

        if widths is not None:
            doc.xref_set_key(xref, "FirstChar", "32")
            doc.xref_set_key(xref, "LastChar", "255")
            doc.xref_set_key(xref, "Widths", "[" + " ".join(map(str, widths)) + "]")
            if codec == "cp1252" and doc.xref_get_key(xref, "Encoding")[0] == "null":
                # Pin the mapping the widths were built from, since the embedded
                # program's built-in encoding is not WinAnsi.
                doc.xref_set_key(xref, "Encoding", "/WinAnsiEncoding")

        embedded.append(base_font)
        log.debug("embedded %s as %s (font xref %d)", base_font, builtin, xref)

    if skipped:
        log.info("fonts left unembedded: %s",
                 ", ".join(f"{name} ({reason})" for name, reason in skipped))

    return embedded, skipped


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python embed_fonts.py <input.pdf> <output.pdf>")
        raise SystemExit(1)

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")

    in_path, out_path = sys.argv[1:3]
    doc = fitz.open(in_path)
    embedded, skipped = embed_missing_fonts(doc)
    doc.save(out_path, garbage=3, deflate=True, clean=True)

    print(f"embedded: {', '.join(embedded) if embedded else 'nothing'}")
    for name, reason in skipped:
        print(f"skipped:  {name} - {reason}")
    print(f"Saved: {out_path}")
