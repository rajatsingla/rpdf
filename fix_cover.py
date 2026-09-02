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

# fix_cover.py
# pip install pymupdf numpy
#
# One entry point that fixes a cover PDF (given as bytes) and resizes it to the
# desired final dimensions. All detection logic is reused from the existing
# scripts in this folder:
#   - remove_crop_marks.py : BleedBox/TrimBox crop + visual crop-mark detection
#   - trim_white_space.py  : whitespace fallback crop
#   - trim_flaps.py        : flap detection/removal
#   - resize.py            : final scaling
#   - embed_fonts.py       : font embedding
#
# A size check runs before every trim step and stops the trimming the moment the
# page measures the size the caller asked for. Unlike an interior there is no
# bleed tolerance here: the caller states the final size outright, so a match is
# an exact match (bar measurement noise). See _is_final_size.

import fitz  # PyMuPDF

from remove_crop_marks import (
    existing_box_clip,
    detect_crop_mark_clip,
    is_valid_clip,
    rects_different,
)
from trim_white_space import (
    detect_nonwhite_bbox,
    expand_rect,
    has_real_crop,
    MIN_CROP_PT,
    PADDING_PT,
)
from trim_flaps import detect_flap_clip
from resize import resize_doc
from embed_fonts import embed_missing_fonts

POINTS_PER_INCH = 72

# Slack on the final-size check, for measurement/rounding error only (a 11.9999 in
# page that is really 12 in). A cover carries no bleed allowance: the caller names
# the finished dimensions, so anything beyond them is bleed or marks to be cut -
# unlike an interior, whose size is inferred from a list of standard trim sizes
# and so is allowed to arrive as "standard size plus up to 0.25 in of bleed".
SIZE_MATCH_ERROR_IN = 0.01


def _is_final_size(
    rect: fitz.Rect, final_width_in: float, final_height_in: float
) -> bool:
    """
    True when ``rect`` already measures the requested final size.

    This is the gate on every trim step. A page that is already the finished size
    has nothing outside its trim left to remove, so any further crop - a bleed
    box, a mark detection, a whitespace shave, a flap seam - can only eat into the
    design itself. The resize pass still runs afterwards, so the file is still
    normalised and its fonts still checked.
    """
    return (
        abs(rect.width / POINTS_PER_INCH - final_width_in) <= SIZE_MATCH_ERROR_IN
        and abs(rect.height / POINTS_PER_INCH - final_height_in) <= SIZE_MATCH_ERROR_IN
    )


def _apply_clip(src: fitz.Document, page_index: int, out: fitz.Document, clip: fitz.Rect) -> None:
    """
    Draw ``src`` page ``page_index`` into a new page of ``out``, cropped to ``clip``.

    Destination size == clip size, so artwork is cropped but never scaled or
    rasterized. All standard page boxes are reset to the new page. This is the
    same block used by the trim/flap scripts' main() loops.
    """
    new_page = out.new_page(width=clip.width, height=clip.height)

    new_page.show_pdf_page(
        new_page.rect,
        src,
        page_index,
        clip=clip,
    )

    new_page.set_cropbox(new_page.rect)
    new_page.set_trimbox(new_page.rect)
    new_page.set_bleedbox(new_page.rect)
    new_page.set_artbox(new_page.rect)


def _declares_trimmed(page: fitz.Page) -> bool:
    """
    True when the page declares no area outside its own trim, i.e. every standard
    box is the page rect. Absent boxes default to the MediaBox under the PDF spec,
    so this also covers a file that declares nothing at all.

    Such a file is stating that it is already the finished page, so there is
    nothing outside the trim to shave and the whitespace fallback must not run.
    A cover's outer white is its margin, and cropping it deletes margin the
    design was laid out with.
    """
    page_rect = page.rect

    return not any(
        rects_different(box, page_rect)
        for box in (page.cropbox, page.trimbox, page.bleedbox, page.artbox)
    )


def _both_sided_only(clip: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect:
    """
    Keep only the whitespace that shows up on BOTH sides of an axis.

    Excess white around a press file surrounds the artwork, so it appears left
    and right, or top and bottom. White down one side alone is the cover's own
    margin - the space the design leaves above its title - and shaving it both
    deletes that margin and pulls the artwork off centre. So an axis is left
    whole unless both of its edges have white to give.
    """
    left = clip.x0 - page_rect.x0
    right = page_rect.x1 - clip.x1
    top = clip.y0 - page_rect.y0
    bottom = page_rect.y1 - clip.y1

    if min(left, right) <= MIN_CROP_PT:
        left = right = 0.0

    if min(top, bottom) <= MIN_CROP_PT:
        top = bottom = 0.0

    return fitz.Rect(
        page_rect.x0 + left,
        page_rect.y0 + top,
        page_rect.x1 - right,
        page_rect.y1 - bottom,
    )


def _trim_clip(
    page: fitz.Page, final_width_in: float, final_height_in: float
) -> fitz.Rect:
    """
    Decide the trim clip for a page using the requested priority:
      0. no clip at all if the page already measures the final size
      1. BleedBox if present
      2. else TrimBox if present
      3. else visual crop-mark detection
      4. else whitespace crop, but only on a page that does not declare itself
         already trimmed, and only on axes with white to spare at both ends
    Falls back to the full page rect if nothing qualifies.

    The size check runs before each step, so the ladder stops at the first thing
    that measures the requested final size and nothing further is taken off.
    """
    page_rect = page.rect

    # 0. Already the finished size - nothing to trim.
    if _is_final_size(page_rect, final_width_in, final_height_in):
        return page_rect

    bleed = existing_box_clip(page, page.bleedbox)
    trim = existing_box_clip(page, page.trimbox)

    # A declared box that lands exactly on the requested size is the cut line,
    # whichever box it is. A file declaring BleedBox 12.5x9.25 and TrimBox 12x9
    # for a 12x9 cover is stating where the knife goes; cropping to the bleed by
    # rank alone would keep that bleed and then scale it into the finished cover,
    # shrinking the artwork and pulling the spine off centre.
    for candidate in (bleed, trim):
        if candidate is not None and _is_final_size(
            candidate, final_width_in, final_height_in
        ):
            return candidate

    # 1. BleedBox / 2. TrimBox, by declared priority.
    for candidate in (bleed, trim):
        if candidate is not None:
            return candidate

    # 3. Visual crop marks
    clip, info = detect_crop_mark_clip(page)
    if info.get("detected"):
        return clip

    # 4. Whitespace fallback
    detected = None if _declares_trimmed(page) else detect_nonwhite_bbox(page)
    if detected is not None:
        detected = expand_rect(detected, PADDING_PT, page_rect)
        detected = _both_sided_only(detected, page_rect)
        # Require both a real crop and a sane size. Without the size guard a
        # single stray dark pixel (or the any-non-white fallback inside
        # detect_nonwhite_bbox) could crop the cover down to a speck.
        if has_real_crop(page_rect, detected) and is_valid_clip(detected, page_rect):
            return detected

    return page_rect


def _clipped(doc: fitz.Document, clip: fitz.Rect) -> fitz.Document:
    """
    A new single-page document holding page 0 of ``doc`` cropped to ``clip``.
    Closes ``doc``.
    """
    out = fitz.open()
    out.set_metadata(doc.metadata)
    _apply_clip(doc, 0, out, clip)
    doc.close()
    return out


def fix_cover(
    cover_bytes: bytes,
    final_width_in: float,
    final_height_in: float,
    output_path: str | None = None,
) -> bytes:
    """
    Fix a cover PDF and resize it to the final dimensions.

    Pipeline:
      0.   Keep only the first page (a cover is a single page).
      1-4. Crop to BleedBox / TrimBox / detected crop marks / whitespace, unless
           the page already measures the final size.
      5.   Detect and remove flaps, unless the page already measures the final
           size.
      6.   Resize to ``final_width_in`` x ``final_height_in`` (inches).
      7.   Embed any font the file references without carrying.

    The size check before steps 1-4 and step 5 is the whole trimming policy: once
    the page measures what the caller asked for, cutting can only remove design,
    so it stops. Steps 6 and 7 always run, so a file that needed no cutting is
    still normalised and still leaves with its fonts embedded.

    Args:
        cover_bytes:     The cover PDF as bytes.
        final_width_in:  Desired final width in inches.
        final_height_in: Desired final height in inches.
        output_path:     Optional path to also write the final PDF to.

    Returns:
        The final PDF as bytes.
    """
    doc = fitz.open(stream=cover_bytes, filetype="pdf")

    # Stage 0: a cover is one page. Dropping the rest up front keeps every stage
    # below from spending work - a render per detector - on pages that were going
    # to be discarded anyway, and keeps fonts used only by those pages from being
    # embedded into the file that ships.
    if doc.page_count > 1:
        doc.delete_pages(from_page=1, to_page=doc.page_count - 1)

    # Stage A: trim (bleed / trim / crop marks / whitespace).
    if not _is_final_size(doc[0].rect, final_width_in, final_height_in):
        clip = _trim_clip(doc[0], final_width_in, final_height_in)
        if rects_different(clip, doc[0].rect):
            doc = _clipped(doc, clip)

        # Stage B: remove flaps - but only if the trim did not already land the
        # page on the final size, in which case there are no flaps left to find
        # and a seam detected inside the artwork would cut the design.
        if not _is_final_size(doc[0].rect, final_width_in, final_height_in):
            clip, _info = detect_flap_clip(doc[0])
            if rects_different(clip, doc[0].rect):
                doc = _clipped(doc, clip)

    # Stage C: resize to final dimensions.
    resize_doc(doc, final_width_in, final_height_in)

    # Stage D: the press needs the fonts in the file, whatever route it took to
    # get here.
    embed_missing_fonts(doc)

    data = doc.tobytes(garbage=4, deflate=True)
    doc.close()

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(data)

    return data


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("usage: python fix_cover.py <input.pdf> <width_in> <height_in> <output.pdf>")
        raise SystemExit(1)

    in_path, w_in, h_in, out_path = sys.argv[1:5]
    with open(in_path, "rb") as f:
        result = fix_cover(f.read(), float(w_in), float(h_in), out_path)

    print(f"Saved: {out_path} ({len(result)} bytes)")
