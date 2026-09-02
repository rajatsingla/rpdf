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

# fix_interior_file.py
# pip install pymupdf numpy
#
# One entry point that fixes an interior (book block) PDF given as bytes:
#   1. Match the page size to the supported book trim sizes, allowing up to
#      +0.25 in of added trim/bleed.
#   2. Only when it does not match: remove crop marks if present (no flaps, no
#      whitespace trim), cut any bleed the crop left behind so the page lands on
#      the trim line, and match again on the cut size. A file that already
#      measures as a supported size is never cut - see fix_interior_file.
#   3. Always run the resize pipeline to normalise the file:
#        - match found  -> resize to the file's own dimensions (normalise only,
#                          which also fixes stray rotated/horizontal pages).
#        - no match     -> resize to the closest supported trim size.
#   4. Always embed a font program for any font referenced by name only.
#
# All detection/editing logic is reused from the existing scripts in this folder.

import fitz  # PyMuPDF

from remove_crop_marks import (
    existing_box_clip,
    detect_crop_mark_clip,
    rects_different,
)
from resize import resize_doc
from fix_cover import _apply_clip
from embed_fonts import embed_missing_fonts

POINTS_PER_INCH = 72

# A file may carry up to this much added trim/bleed over the true trim size.
TRIM_TOLERANCE_IN = 0.25

# Allow the file to be this much *smaller* than a trim size and still match it,
# to absorb measurement/rounding error (e.g. a 5.9999 in page that is really 6 in).
SIZE_MATCH_ERROR_IN = 0.01

# How many pages to inspect when looking for crop marks. The opening pages of an
# interior often carry none (blank half-title, full-bleed art, a cover page kept
# in the block), so detection cannot rely on page 1 alone. These pages vote and
# the majority wins; the first page with marks does NOT get to decide, because a
# single page of body text can score like a set of marks (a narrow column of dark
# pixels inside the detector's edge search band) and would otherwise crop the
# whole file to its text block.
DETECT_MAX_PAGES = 3

# Two clips count as the same vote when every edge is within this many points.
# Real crop marks are drawn identically on every page, so detection across pages
# lands within a pixel of itself (0.36 pt at the 200 dpi detect render), while a
# false positive read off body text sits tens of points away and never merges.
VOTE_TOLERANCE_PT = 2.0

# Supported book trim sizes (international superset of domesticBookSizes),
# mirrored from tango/src/utils/book/index.ts. Includes both portrait and
# landscape variants, so (width, height) is compared directly.
SUPPORTED_SIZES = [
    {"name": "Pocket Book", "width_in": 4.25, "height_in": 6.87},
    {"name": "Novella", "width_in": 5.0, "height_in": 8.0},
    {"name": "Digest", "width_in": 5.5, "height_in": 8.5},
    {"name": "A5", "width_in": 5.83, "height_in": 8.27},
    {"name": "US Trade", "width_in": 6.0, "height_in": 9.0},
    {"name": "Royal", "width_in": 6.14, "height_in": 9.21},
    {"name": "Executive", "width_in": 7.0, "height_in": 10.0},
    {"name": "Crown Quarto", "width_in": 7.44, "height_in": 9.68},
    {"name": "Small Square", "width_in": 7.5, "height_in": 7.5},
    {"name": "A4", "width_in": 8.27, "height_in": 11.69},
    {"name": "Square", "width_in": 8.5, "height_in": 8.5},
    {"name": "US Letter", "width_in": 8.5, "height_in": 11.0},
    {"name": "Small Landscape", "width_in": 9.0, "height_in": 7.0},
    {"name": "US Letter Landscape", "width_in": 11.0, "height_in": 8.5},
    {"name": "A4 Landscape", "width_in": 11.69, "height_in": 8.27},
]

# Domestic-only trim sizes (domesticBookSizes from tango/src/utils/book/index.ts).
DOMESTIC_SIZES = [
    {"name": "Pocket Book", "width_in": 4.25, "height_in": 6.87},
    {"name": "Novella", "width_in": 5.0, "height_in": 8.0},
    {"name": "Digest", "width_in": 5.5, "height_in": 8.5},
    {"name": "US Trade", "width_in": 6.0, "height_in": 9.0},
    {"name": "Square", "width_in": 8.5, "height_in": 8.5},
    {"name": "US Letter", "width_in": 8.5, "height_in": 11.0},
]


def _majority(votes: list[fitz.Rect]) -> fitz.Rect | None:
    """
    The rect that a majority of ``votes`` agree on, to within VOTE_TOLERANCE_PT
    on every edge, or None when no rect reaches a majority of the votes cast.

    "Majority of the votes cast" and not of DETECT_MAX_PAGES, so a two-page or
    one-page file still resolves: 3 votes need 2, 2 need 2, 1 needs 1.
    """
    if not votes:
        return None

    needed = len(votes) // 2 + 1

    for clip in votes:
        agree = sum(
            1 for other in votes
            if not rects_different(clip, other, VOTE_TOLERANCE_PT)
        )
        if agree >= needed:
            return clip

    return None


def _crop_marks_clip(doc: fitz.Document) -> fitz.Rect:
    """
    Decide the crop-mark clip for an interior, in priority order:
      1. TrimBox if present (the true cut size -> book size matches exactly)
      2. else BleedBox if present
      3. else visual crop-mark detection
    Falls back to the full page rect when no crop marks are found.
    No whitespace trim and no flap handling for interiors.

    TrimBox is preferred over BleedBox so interiors are cut to the trim line:
    the kept page size is the real book size (e.g. 6x9), which then matches a
    supported trim size exactly instead of a larger size due to retained bleed.

    Interiors are geometrically uniform, so a page that disagrees with the others
    is a page that misread. The first DETECT_MAX_PAGES pages therefore vote and
    the majority wins, rather than the first page with a clip deciding for the
    whole file. The winning clip is reused for every page.
    """
    pages = [doc[index] for index in range(min(DETECT_MAX_PAGES, doc.page_count))]

    # Declared boxes are authoritative and cost nothing to read, so they vote
    # first and settle the clip without rendering anything. Only pages that
    # declare a box vote here: a page that declares none stays silent rather than
    # voting against, because a missing box is absent metadata, not evidence that
    # the page is untrimmed.
    box_votes = []
    for page in pages:
        clip = existing_box_clip(page, page.trimbox)
        if clip is None:
            clip = existing_box_clip(page, page.bleedbox)
        if clip is not None:
            box_votes.append(clip)

    winner = _majority(box_votes)
    if winner is not None:
        return winner

    # Nothing declared: read the marks off the rendered pages instead. Here a
    # page that finds no marks DOES vote - for its own rect, i.e. for not
    # cropping - because detection looks at the whole page, so "no marks here" is
    # real evidence and not a gap in the metadata. That is what stops one page of
    # body text from cropping the file down to its text block.
    detect_votes = []
    for page in pages:
        clip, info = detect_crop_mark_clip(page)
        detect_votes.append(clip if info.get("detected") else page.rect)

    winner = _majority(detect_votes)
    if winner is not None:
        return winner

    return doc[0].rect


def _area(size: dict) -> float:
    return size["width_in"] * size["height_in"]


def _match_size(
    width_in: float, height_in: float, is_domestic: bool = False
) -> tuple[str, dict]:
    """
    Match (width_in, height_in) against the supported sizes (the domestic-only
    list when is_domestic, otherwise the full international list).

    Every file is modelled as a base trim size plus 0..TRIM_TOLERANCE_IN of
    added bleed, with a small SIZE_MATCH_ERROR_IN slack below for measurement
    error. A size matches when:
        size.w - 0.01 <= width  <= size.w + 0.25
        size.h - 0.01 <= height <= size.h + 0.25

    Returns ("match", size) when at least one size matches; on overlap the
    SMALLEST base size wins, since the file is read as that base plus bleed
    (e.g. 6.25x9.25 is US Trade 6x9 + full bleed, not Royal). Otherwise
    ("resize", closest_size) by Euclidean distance, ties towards smaller area.
    """
    lo = TRIM_TOLERANCE_IN  # upper slack (added bleed)
    err = SIZE_MATCH_ERROR_IN  # lower slack (measurement error)
    sizes = DOMESTIC_SIZES if is_domestic else SUPPORTED_SIZES

    matches = [
        s for s in sizes
        if s["width_in"] - err <= width_in <= s["width_in"] + lo
        and s["height_in"] - err <= height_in <= s["height_in"] + lo
    ]

    if matches:
        return "match", min(matches, key=_area)

    def distance(s: dict) -> tuple[float, float]:
        dw = width_in - s["width_in"]
        dh = height_in - s["height_in"]
        return (dw * dw + dh * dh, _area(s))

    return "resize", min(sizes, key=distance)


def _trim_retained_bleed(
    clip: fitz.Rect, page_rect: fitz.Rect, is_domestic: bool = False
) -> fitz.Rect:
    """
    Cut the bleed off a clip that came out as a supported trim size plus bleed.

    Crop marks come in sets: trim marks on the cut line, bleed marks outboard of
    them. Detection cannot reliably tell the two apart - two mark sets drawn to
    the same length score the same, and the strongest peak wins, which favours the
    outer set - and a file declaring a BleedBox but no TrimBox hands over the bleed
    box outright. Either way the clip is the book size plus bleed.

    That anything was cropped at all is the evidence needed: the page carried area
    outside the trim, so whatever the clip holds over the trim size it matched is
    bleed, and bleed exists to be cut off. Shaving it centred lands the page on the
    exact book size without scaling the artwork.

    Returns ``clip`` unchanged when nothing was cropped - the page is then its own
    trim size, so any excess over a standard size is real content - or when the
    clip is not a supported trim size plus bleed.
    """
    if not rects_different(clip, page_rect):
        return clip

    kind, size = _match_size(
        clip.width / POINTS_PER_INCH,
        clip.height / POINTS_PER_INCH,
        is_domestic,
    )

    if kind != "match":
        return clip

    # Clamped at zero because a match may be up to SIZE_MATCH_ERROR_IN smaller than
    # its size, and never more than half of TRIM_TOLERANCE_IN because that is as
    # much as _match_size tolerates.
    dx = max((clip.width - size["width_in"] * POINTS_PER_INCH) / 2, 0.0)
    dy = max((clip.height - size["height_in"] * POINTS_PER_INCH) / 2, 0.0)

    return fitz.Rect(clip.x0 + dx, clip.y0 + dy, clip.x1 - dx, clip.y1 - dy)


def _target_size_in(
    width_in: float, height_in: float, kind: str, size: dict
) -> tuple[float, float]:
    """
    Pick the dimensions to resize to for a measured page and its match result.

    A match that is only measurement noise away from its standard size is snapped
    to the exact standard dims, so the output is perfectly sized. A match that
    carries real added bleed keeps its own dims, so the content is not
    scaled/distorted and the resize pass only normalises. No match resizes to the
    closest supported size.
    """
    if kind != "match":
        return size["width_in"], size["height_in"]

    within_error = (
        abs(width_in - size["width_in"]) <= SIZE_MATCH_ERROR_IN
        and abs(height_in - size["height_in"]) <= SIZE_MATCH_ERROR_IN
    )
    if within_error:
        return size["width_in"], size["height_in"]

    return width_in, height_in


def fix_interior_file(
    pdf_bytes: bytes,
    output_path: str | None = None,
    is_domestic: bool = False,
) -> bytes:
    """
    Fix an interior PDF: match to a supported size, cutting the file down only if
    it does not already measure as one, resize/normalise, and embed any font the
    file references without carrying.

    Args:
        pdf_bytes:   The interior PDF as bytes.
        output_path: Optional path to also write the final PDF to.
        is_domestic: Match against the domestic trim sizes only.

    Returns:
        The final PDF as bytes.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    width_in = doc[0].rect.width / POINTS_PER_INCH
    height_in = doc[0].rect.height / POINTS_PER_INCH
    kind, size = _match_size(width_in, height_in, is_domestic)

    # A file that already measures as a supported trim size - exactly, or as that
    # size plus up to TRIM_TOLERANCE_IN of bleed - is accepted as it stands, so
    # there is nothing to gain by cutting it: crop-mark removal, the retained-bleed
    # shave and whitespace trimming can only take real content off a page that
    # already passes. It still goes through the resize pass below, which normalises
    # the block (a stray page left rotated/horizontal is the common case) and is
    # also where fonts get checked, so the file is not passed through untouched.
    if kind != "match":
        # Remove crop marks (BleedBox / TrimBox / visual marks), then cut any bleed
        # the crop kept, so the page ends up on the trim line. Interiors are
        # geometrically uniform, so detect the clip once and reuse it for every
        # page. This avoids a full-page render per page, which is the dominant cost
        # for large multi-page files.
        cut = fitz.open()
        cut.set_metadata(doc.metadata)
        clip = _trim_retained_bleed(_crop_marks_clip(doc), doc[0].rect, is_domestic)
        for page_index in range(doc.page_count):
            _apply_clip(doc, page_index, cut, clip)
        doc.close()
        doc = cut

        # Match again: the cut may well have landed the page on a supported size.
        width_in = doc[0].rect.width / POINTS_PER_INCH
        height_in = doc[0].rect.height / POINTS_PER_INCH
        kind, size = _match_size(width_in, height_in, is_domestic)

    target_w_in, target_h_in = _target_size_in(width_in, height_in, kind, size)
    resize_doc(doc, target_w_in, target_h_in)

    # Print requires the fonts to travel with the file, and a page-geometry pass
    # is the only place the block gets rewritten, so check on every run - the
    # file that needed no cutting is exactly the one that would otherwise go out
    # with its fonts still referenced by name only.
    embed_missing_fonts(doc)

    data = doc.tobytes(garbage=4, deflate=True)
    doc.close()

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(data)

    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python fix_interior_file.py <input.pdf> <output.pdf>")
        raise SystemExit(1)

    in_path, out_path = sys.argv[1:3]
    with open(in_path, "rb") as f:
        result = fix_interior_file(f.read(), out_path)

    print(f"Saved: {out_path} ({len(result)} bytes)")
