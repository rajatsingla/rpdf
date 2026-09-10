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

import logging

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
from resize import needs_flattening, resize_doc
from embed_fonts import embed_missing_fonts

log = logging.getLogger(__name__)

POINTS_PER_INCH = 72

# Slack on the final-size check, for measurement/rounding error only (a 11.9999 in
# page that is really 12 in). A cover carries no bleed allowance: the caller names
# the finished dimensions, so anything beyond them is bleed or marks to be cut -
# unlike an interior, whose size is inferred from a list of standard trim sizes
# and so is allowed to arrive as "standard size plus up to 0.25 in of bleed".
SIZE_MATCH_ERROR_IN = 0.01

# Most a page may measure over the requested final size, per axis, and still be
# treated as marks/bleed/slug to be cropped off rather than artwork to be scaled
# down. A press sheet carries bleed plus a mark and slug margin - commonly
# 0.25-0.5 in a side - so an inch on an axis is ordinary and two is generous.
# Past that the file is not an oversized press sheet, it is a different size, and
# scaling it (with resize_doc's warning) is the honest answer.
MAX_CROPPABLE_SURPLUS_IN = 2.0


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


def carry_over_links(
    src: fitz.Document, out: fitz.Document, clips: list[fitz.Rect]
) -> None:
    """
    Re-attach ``src``'s links and bookmarks to the rebuilt pages in ``out``.

    Rebuilding a page with show_pdf_page copies its content, not the annotations
    layered over it, and a fresh document has no outline - so an interior would
    otherwise leave with its cross-references and table of contents stripped.
    ``clips`` holds the crop taken from each page, in order, so a link rectangle
    can be moved into the rebuilt page's coordinates; a link lying wholly outside
    the kept area goes away with the artwork it sat on.

    Runs after every page exists: a GoTo link is resolved against the destination
    document as it is inserted, so one pointing at a page not yet added fails.
    """
    for page_index, clip in enumerate(clips):
        offset = (-clip.x0, -clip.y0, -clip.x0, -clip.y0)
        out_page = out[page_index]

        for link in src[page_index].get_links():
            moved = fitz.Rect(link["from"]) + offset
            if (moved & out_page.rect).is_empty:
                continue

            link = {**link, "from": moved}

            # An internal link also names a point on the page it jumps to, in
            # that page's coordinates, so it shifts by that page's crop and not
            # by this one's.
            target = link.get("page", -1)
            if link.get("to") is not None and 0 <= target < len(clips):
                target_clip = clips[target]
                link["to"] = fitz.Point(
                    link["to"].x - target_clip.x0,
                    link["to"].y - target_clip.y0,
                )

            out_page.insert_link(link)

    toc = src.get_toc()
    if toc:
        out.set_toc(toc)


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
    Keep only the whitespace that shows up on BOTH sides of an axis, and take the
    same amount off each end of it.

    Excess white around a press file surrounds the artwork, so it appears left
    and right, or top and bottom. White down one side alone is the cover's own
    margin - the space the design leaves above its title - and shaving it both
    deletes that margin and pulls the artwork off centre. So an axis is left
    whole unless both of its edges have white to give.

    Even then only the smaller of the two amounts comes off both ends. An uneven
    shave moves the centre of the artwork, and stage C then scales that
    off-centre crop up to the finished size, so a cover's spine ends up away from
    the middle of the wrap - the file measures right and the panels do not.
    """
    left = clip.x0 - page_rect.x0
    right = page_rect.x1 - clip.x1
    top = clip.y0 - page_rect.y0
    bottom = page_rect.y1 - clip.y1

    if min(left, right) <= MIN_CROP_PT:
        left = right = 0.0
    else:
        left = right = min(left, right)

    if min(top, bottom) <= MIN_CROP_PT:
        top = bottom = 0.0
    else:
        top = bottom = min(top, bottom)

    return fitz.Rect(
        page_rect.x0 + left,
        page_rect.y0 + top,
        page_rect.x1 - right,
        page_rect.y1 - bottom,
    )


def _fmt(rect: fitz.Rect) -> str:
    """``rect`` as inches, for the log."""
    return (
        f"{rect.width / POINTS_PER_INCH:.4g}x{rect.height / POINTS_PER_INCH:.4g} in "
        f"@({rect.x0:.1f},{rect.y0:.1f})pt"
    )


def _chose(clip: fitz.Rect, rung: str) -> fitz.Rect:
    """Log which rung of the trim ladder fired, then return its clip."""
    log.info("cover trim: %s -> %s", rung, _fmt(clip))
    return clip


def _log_input(page: fitz.Page, final_width_in: float, final_height_in: float) -> None:
    """
    Record the geometry the file arrived with.

    Everything that can put a cover's artwork somewhere other than where it
    belongs is visible here: a MediaBox origin away from (0, 0), a CropBox
    smaller than the MediaBox, a declared trim that disagrees with what was
    asked for, a /Rotate. Without it a complaint about a shifted spine cannot be
    traced back to the file that caused it.
    """
    log.info(
        "cover in: page %s media %s crop %s trim %s bleed %s rotate %s "
        "-> requested %.4gx%.4g in",
        _fmt(page.rect), _fmt(page.mediabox), _fmt(page.cropbox),
        _fmt(page.trimbox), _fmt(page.bleedbox), page.rotation,
        final_width_in, final_height_in,
    )


def _crop_to_final(
    page: fitz.Page, final_width_in: float, final_height_in: float
) -> fitz.Rect | None:
    """
    Last resort: a window of exactly the requested size, centred on the ink.

    Nothing was detected, so there is no declared or drawn cut line to crop to -
    but the caller has named the finished size, and a page measuring more than
    that is carrying bleed, crop marks and slug that all have to come off.
    Cutting a window of the requested size takes them off and leaves every panel
    at its own size. The alternative, which is what returning None means, is to
    let stage C scale the whole sheet: that squeezes the marks and the white
    margin into the finished cover along with the design, shrinking every panel
    and, where the two axes disagree, stretching the artwork as well. On a
    12.4409x9.4488 in sheet asked for an 11.65x8.76 in cover that is a 0.35 in
    loss on each panel and a 1% aspect error.

    Centred on the ink rather than on the sheet: a mark and slug margin is often
    wider on one side, and centring on the sheet would carry that asymmetry
    straight into the spine position.

    Returns None when cropping cannot do the job - a page smaller than the
    request on either axis, or so much larger that the surplus cannot be marks.
    """
    page_rect = page.rect

    want_w = final_width_in * POINTS_PER_INCH
    want_h = final_height_in * POINTS_PER_INCH

    surplus_w = page_rect.width - want_w
    surplus_h = page_rect.height - want_h
    limit = MAX_CROPPABLE_SURPLUS_IN * POINTS_PER_INCH

    # Not even on an axis: a window bigger than the page cannot be cut out of it,
    # and clamping one to fit would hand back something other than the size asked
    # for. Rung 0 has already let through anything within SIZE_MATCH_ERROR_IN, so
    # what is left here is genuinely undersized and belongs to stage C.
    if surplus_w < 0 or surplus_h < 0:
        log.info(
            "cover: page is smaller than the requested size on an axis "
            "(%.3g x %.3g in surplus), cannot crop to it",
            surplus_w / POINTS_PER_INCH, surplus_h / POINTS_PER_INCH,
        )
        return None

    if surplus_w > limit or surplus_h > limit:
        log.warning(
            "cover: page exceeds the requested size by %.3g x %.3g in, more than "
            "MAX_CROPPABLE_SURPLUS_IN=%.3g - treating as a different size, not as "
            "marks to crop",
            surplus_w / POINTS_PER_INCH, surplus_h / POINTS_PER_INCH,
            MAX_CROPPABLE_SURPLUS_IN,
        )
        return None

    # Centre of the printed area, falling back to the centre of the sheet on a
    # page that reads as blank.
    ink = detect_nonwhite_bbox(page)
    box = ink if ink is not None else page_rect
    cx = (box.x0 + box.x1) / 2
    cy = (box.y0 + box.y1) / 2

    x0 = cx - want_w / 2
    y0 = cy - want_h / 2

    # Keep the window on the page; centring on ink that sits near an edge can
    # otherwise push it off.
    x0 = min(max(x0, page_rect.x0), page_rect.x1 - want_w)
    y0 = min(max(y0, page_rect.y0), page_rect.y1 - want_h)

    return fitz.Rect(x0, y0, x0 + want_w, y0 + want_h)


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
      5. else a window of exactly the requested size cut out of an oversized
         page, centred on the ink
    Falls back to the full page rect if nothing qualifies.

    The size check runs before each step, so the ladder stops at the first thing
    that measures the requested final size and nothing further is taken off.
    """
    page_rect = page.rect

    # 0. Already the finished size - nothing to trim.
    if _is_final_size(page_rect, final_width_in, final_height_in):
        return _chose(page_rect, "page is already the final size")

    bleed = existing_box_clip(page, page.bleedbox)
    trim = existing_box_clip(page, page.trimbox)

    # A declared box that lands exactly on the requested size is the cut line,
    # whichever box it is. A file declaring BleedBox 12.5x9.25 and TrimBox 12x9
    # for a 12x9 cover is stating where the knife goes; cropping to the bleed by
    # rank alone would keep that bleed and then scale it into the finished cover,
    # shrinking the artwork and pulling the spine off centre.
    for name, candidate in (("bleedbox", bleed), ("trimbox", trim)):
        if candidate is not None and _is_final_size(
            candidate, final_width_in, final_height_in
        ):
            return _chose(candidate, f"{name} is the final size")

    # 1. BleedBox / 2. TrimBox, by declared priority.
    for name, candidate in (("bleedbox", bleed), ("trimbox", trim)):
        if candidate is not None:
            return _chose(candidate, f"declared {name}")

    # 3. Visual crop marks
    clip, info = detect_crop_mark_clip(page)
    if info.get("detected"):
        return _chose(clip, "detected crop marks")

    # 4. Whitespace fallback
    detected = None if _declares_trimmed(page) else detect_nonwhite_bbox(page)
    if detected is not None:
        detected = expand_rect(detected, PADDING_PT, page_rect)
        detected = _both_sided_only(detected, page_rect)
        # Require both a real crop and a sane size. Without the size guard a
        # single stray dark pixel (or the any-non-white fallback inside
        # detect_nonwhite_bbox) could crop the cover down to a speck.
        if has_real_crop(page_rect, detected) and is_valid_clip(detected, page_rect):
            return _chose(detected, "whitespace")

    # 5. Nothing found. Cut the requested size out of the sheet rather than leave
    # the marks on and let stage C scale them into the cover.
    to_final = _crop_to_final(page, final_width_in, final_height_in)
    if to_final is not None:
        return _chose(to_final, "cropped to the requested size, centred on the ink")

    return _chose(page_rect, "nothing to trim")


def _clipped(doc: fitz.Document, clip: fitz.Rect) -> fitz.Document:
    """
    A new single-page document holding page 0 of ``doc`` cropped to ``clip``.
    Closes ``doc``.
    """
    out = fitz.open()
    out.set_metadata(doc.metadata)
    _apply_clip(doc, 0, out, clip)
    carry_over_links(doc, out, [clip])
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
      1-5. Crop to BleedBox / TrimBox / detected crop marks / whitespace / a
           window of the requested size, unless the page already measures the
           final size.
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

    _log_input(doc[0], final_width_in, final_height_in)

    # Stage A: trim (bleed / trim / crop marks / whitespace).
    if not _is_final_size(doc[0].rect, final_width_in, final_height_in):
        page_rect = doc[0].rect
        clip = _trim_clip(doc[0], final_width_in, final_height_in)
        if rects_different(clip, page_rect):
            # Removed amounts, per edge. An uneven pair on either axis is what
            # moves a cover's spine off centre once stage C scales the crop back
            # up, so it is the first thing to look at on a complaint.
            log.info(
                "cover trim: removed l=%.2f r=%.2f t=%.2f b=%.2f pt",
                clip.x0 - page_rect.x0, page_rect.x1 - clip.x1,
                clip.y0 - page_rect.y0, page_rect.y1 - clip.y1,
            )
            doc = _clipped(doc, clip)

        # Stage B: remove flaps - but only if the trim did not already land the
        # page on the final size, in which case there are no flaps left to find
        # and a seam detected inside the artwork would cut the design.
        if not _is_final_size(doc[0].rect, final_width_in, final_height_in):
            clip, _info = detect_flap_clip(doc[0])
            if rects_different(clip, doc[0].rect):
                doc = _clipped(doc, clip)

    # Stage B2: flatten the page boxes before scaling. Stage C is only a correct
    # transform on a page whose MediaBox is its visible page and starts at
    # (0, 0); see resize.needs_flattening. A page rebuilt by _clipped above is
    # already flat, so this costs nothing on the paths that trimmed.
    if needs_flattening(doc[0]):
        log.info("cover: flattening page boxes before resize")
        doc = _clipped(doc, doc[0].rect)

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
