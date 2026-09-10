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

# trim_pdf_by_crop_marks.py
# pip install pymupdf numpy

from pathlib import Path
import fitz  # PyMuPDF
import numpy as np


INPUT_PDF = Path("bleeds.pdf")
OUTPUT_PDF = Path("bleeds_no_crop_marks.pdf")

POINTS_PER_INCH = 72

# Render is used only for detecting crop marks.
# Output remains vector PDF content.
DETECT_DPI = 200

# Use existing PDF TrimBox if available.
# This is usually the most reliable way to remove crop marks.
USE_EXISTING_TRIMBOX = True

# Crop marks are usually black/gray.
# Increase this if your crop marks are light gray.
DARK_THRESHOLD = 175

# Helps avoid detecting colorful artwork as crop marks.
# Crop marks are usually neutral gray/black.
MAX_RGB_SPREAD = 65

# Search near page edges only.
# Crop marks should be near the outer edges.
EDGE_SEARCH_PT = 72  # 1 inch

# Minimum dark-pixel length required for a crop mark candidate.
MIN_MARK_LENGTH_PT = 7

# Maximum allowed thickness of a crop-mark peak.
# A real crop mark is a thin line, so its dark-pixel group projects to a narrow
# peak (a few points wide). A solid dark fill (e.g. a full-bleed dark cover)
# saturates the whole edge band, producing a group as wide as the search band.
# Rejecting wide groups prevents misreading full-bleed artwork as crop marks.
MAX_MARK_WIDTH_PT = 6.0

# Do not crop unless we remove at least this much from an edge.
MIN_CROP_PT = 1.0

# Optional tiny inward shave to avoid hairline crop mark residue at edge.
# Keep 0.0 if you want exact trim-line crop.
EDGE_SHAVE_PT = 0.25

# Crop marks should usually produce similar left/right and top/bottom margins.
REQUIRE_OPPOSITE_MARGIN_SIMILARITY = True
MAX_OPPOSITE_MARGIN_DIFF_PT = 8.0


def pt_to_in(value: float) -> float:
    return value / POINTS_PER_INCH


def rects_different(a: fitz.Rect, b: fitz.Rect, tolerance: float = 0.5) -> bool:
    return (
        abs(a.x0 - b.x0) > tolerance or
        abs(a.y0 - b.y0) > tolerance or
        abs(a.x1 - b.x1) > tolerance or
        abs(a.y1 - b.y1) > tolerance
    )


def is_valid_clip(clip: fitz.Rect, page_rect: fitz.Rect) -> bool:
    if clip.is_empty:
        return False

    if clip.width <= 0 or clip.height <= 0:
        return False

    # Avoid accidental huge crops.
    if clip.width < page_rect.width * 0.5:
        return False

    if clip.height < page_rect.height * 0.5:
        return False

    return True


def box_to_page_rect(page: fitz.Page, box: fitz.Rect) -> fitz.Rect:
    """
    Convert a declared page box into page coordinates.

    PyMuPDF reports TrimBox/BleedBox/ArtBox/CropBox in the file's own coordinate
    space, but ``page.rect``, ``get_pixmap`` and ``show_pdf_page(clip=...)`` all
    work in page coordinates, whose origin is the top-left of the CropBox. The
    two only coincide when the CropBox origin happens to be (0, 0). On a press
    export whose origin sits at the trim corner with the bleed running negative -
    MediaBox [-36 -36 856.8 648] and the like - using a raw box as a clip crops a
    correctly sized window from the wrong place: the file measures right and the
    artwork is bodily shifted, so a cover's spine lands off centre and one panel
    gains exactly what the other loses.

    Shifting by the CropBox origin and then applying the page rotation maps the
    box onto the same space as ``page.rect``.
    """
    cropbox = page.cropbox

    shift = fitz.Matrix(1, 0, 0, 1, -cropbox.x0, -cropbox.y0)

    return fitz.Rect(box) * shift * page.rotation_matrix


def existing_box_clip(page: fitz.Page, box: fitz.Rect | None) -> fitz.Rect | None:
    """
    Use a PDF page box (TrimBox or BleedBox) when available.

    Many print-ready PDFs already contain:
    - MediaBox:  full page including marks
    - BleedBox:  final cut size plus bleed (print-ready)
    - TrimBox:   final cut size

    If the box is smaller than the visible page, cropping to it removes crop marks.
    """
    page_rect = page.rect

    if not box or box.is_empty:
        return None

    # Declared boxes are not in page coordinates; convert before using as a clip.
    box = box_to_page_rect(page, box)

    # Keep clip inside current page rectangle.
    box = box & page_rect

    if not is_valid_clip(box, page_rect):
        return None

    if not rects_different(box, page_rect):
        return None

    return box


def existing_trimbox_clip(page: fitz.Page) -> fitz.Rect | None:
    """Backwards-compatible wrapper: crop to the PDF TrimBox when available."""
    return existing_box_clip(page, page.trimbox)


def strongest_group(profile: np.ndarray, start: int, end: int, min_score: float,
                    max_width: float | None = None):
    """
    Find strongest peak group in profile[start:end].

    A crop mark is a thin line, so its peak group should be narrow. When
    ``max_width`` is given (in profile pixels) and the group is wider than that,
    the peak is treated as a solid dark region (not a crop mark) and ``None`` is
    returned. This guards against full-bleed dark artwork saturating the edge
    band, whose centroid would otherwise land at the band midpoint and fake a
    crop mark.

    Returns:
        center_index, peak_score  (or None when no thin mark is found)
    """
    start = max(0, start)
    end = min(len(profile), end)

    if end <= start:
        return None

    segment = profile[start:end]
    peak_local = int(np.argmax(segment))
    peak_score = float(segment[peak_local])

    if peak_score < min_score:
        return None

    # Group nearby high-value pixels around the peak.
    threshold = max(min_score, peak_score * 0.35)

    left = peak_local
    while left > 0 and segment[left - 1] >= threshold:
        left -= 1

    right = peak_local
    while right + 1 < len(segment) and segment[right + 1] >= threshold:
        right += 1

    # A real crop mark is a thin line; a wide group is a solid dark fill.
    if max_width is not None and (right - left + 1) > max_width:
        return None

    indexes = np.arange(left, right + 1)
    weights = segment[left:right + 1].astype(np.float64)

    if weights.sum() <= 0:
        center = start + peak_local
    else:
        center = start + int(round(float(np.average(indexes, weights=weights))))

    return center, peak_score


def detect_crop_mark_clip(page: fitz.Page) -> tuple[fitz.Rect, dict]:
    """
    Detect crop marks from rendered pixels and return the trim clip.

    Detection logic:
    - Render the page.
    - Detect dark neutral pixels.
    - Search only near page edges.
    - Use vertical crop marks to infer left/right trim edges.
    - Use horizontal crop marks to infer top/bottom trim edges.
    - Mirror a single side that carries no marks of its own.
    """
    page_rect = page.rect
    zoom = DETECT_DPI / POINTS_PER_INCH

    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        colorspace=fitz.csRGB,
        alpha=False,
    )

    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.width, pix.n)

    rgb = arr[:, :, :3].astype(np.int16)

    rgb_max = rgb.max(axis=2)
    rgb_min = rgb.min(axis=2)

    dark_neutral = (
        (rgb_max <= DARK_THRESHOLD) &
        ((rgb_max - rgb_min) <= MAX_RGB_SPREAD)
    )

    h, w = dark_neutral.shape

    edge_px = int(EDGE_SEARCH_PT * zoom)
    edge_px = max(20, min(edge_px, int(min(w, h) * 0.25)))

    min_score = max(3, MIN_MARK_LENGTH_PT * zoom)
    max_width = MAX_MARK_WIDTH_PT * zoom

    # Vertical crop-mark candidates:
    # count dark pixels in top and bottom edge bands for every x column.
    vertical_profile = (
        dark_neutral[:edge_px, :].sum(axis=0) +
        dark_neutral[h - edge_px:, :].sum(axis=0)
    )

    # Horizontal crop-mark candidates:
    # count dark pixels in left and right edge bands for every y row.
    horizontal_profile = (
        dark_neutral[:, :edge_px].sum(axis=1) +
        dark_neutral[:, w - edge_px:].sum(axis=1)
    )

    left = strongest_group(vertical_profile, 0, edge_px, min_score, max_width)
    right = strongest_group(vertical_profile, w - edge_px, w, min_score, max_width)

    top = strongest_group(horizontal_profile, 0, edge_px, min_score, max_width)
    bottom = strongest_group(horizontal_profile, h - edge_px, h, min_score, max_width)

    info = {
        "method": "crop-mark-detection",
        "detected": False,
        "reason": "",
        "scores": {
            "left": left[1] if left else None,
            "right": right[1] if right else None,
            "top": top[1] if top else None,
            "bottom": bottom[1] if bottom else None,
            "minimum_required": min_score,
        },
    }

    # Inset (in points) of each detected mark from its own page edge, measured off
    # the page rectangle rather than the pixmap, which is rounded up to whole pixels.
    insets = {}

    if left:
        insets["left"] = left[0] / zoom - page_rect.x0
    if right:
        insets["right"] = page_rect.x1 - right[0] / zoom
    if top:
        insets["top"] = top[0] / zoom - page_rect.y0
    if bottom:
        insets["bottom"] = page_rect.y1 - bottom[0] / zoom

    # Marks are often printed on the leading corners only, so one side can carry
    # none: mirror it from the side opposite, as the trim box is centred on the
    # media box in any normal imposition. Only ever one side, though - body text
    # inside the search band scores like a mark, and inferring two sides from that
    # would crop a page that has no marks at all.
    if len(insets) == 3:
        for side, opposite in (("left", "right"), ("right", "left"),
                               ("top", "bottom"), ("bottom", "top")):
            if side not in insets:
                insets[side] = insets[opposite]

    if len(insets) < 4:
        info["reason"] = "could not find crop marks on at least three sides"
        return page_rect, info

    removed_left = insets["left"]
    removed_right = insets["right"]
    removed_top = insets["top"]
    removed_bottom = insets["bottom"]

    x0 = page_rect.x0 + removed_left
    x1 = page_rect.x1 - removed_right
    y0 = page_rect.y0 + removed_top
    y1 = page_rect.y1 - removed_bottom

    if max(removed_left, removed_right, removed_top, removed_bottom) < MIN_CROP_PT:
        info["reason"] = "detected crop is too small"
        return page_rect, info

    if REQUIRE_OPPOSITE_MARGIN_SIMILARITY:
        if abs(removed_left - removed_right) > MAX_OPPOSITE_MARGIN_DIFF_PT:
            info["reason"] = "left/right crop-mark margins are not similar"
            return page_rect, info

        if abs(removed_top - removed_bottom) > MAX_OPPOSITE_MARGIN_DIFF_PT:
            info["reason"] = "top/bottom crop-mark margins are not similar"
            return page_rect, info

    clip = fitz.Rect(
        x0 + EDGE_SHAVE_PT,
        y0 + EDGE_SHAVE_PT,
        x1 - EDGE_SHAVE_PT,
        y1 - EDGE_SHAVE_PT,
    )

    clip = clip & page_rect

    if not is_valid_clip(clip, page_rect):
        info["reason"] = "detected trim rectangle is invalid"
        return page_rect, info

    info.update({
        "detected": True,
        "reason": "crop marks detected",
        "removed_left_pt": removed_left,
        "removed_right_pt": removed_right,
        "removed_top_pt": removed_top,
        "removed_bottom_pt": removed_bottom,
    })

    return clip, info


def get_trim_clip(page: fitz.Page) -> tuple[fitz.Rect, dict]:
    page_rect = page.rect

    if USE_EXISTING_TRIMBOX:
        trim = existing_trimbox_clip(page)
        if trim is not None:
            return trim, {
                "method": "existing-trimbox",
                "detected": True,
                "reason": "used existing PDF TrimBox",
            }

    return detect_crop_mark_clip(page)


def main():
    src = fitz.open(INPUT_PDF)
    out = fitz.open()

    out.set_metadata(src.metadata)

    for page_index, page in enumerate(src):
        page_rect = page.rect
        clip, info = get_trim_clip(page)

        new_page = out.new_page(width=clip.width, height=clip.height)

        # Destination size equals clip size, so there is no resizing.
        # Rendering was used only for detection.
        new_page.show_pdf_page(
            new_page.rect,
            src,
            page_index,
            clip=clip,
        )

        # Make all standard page boxes match the new trimmed page.
        new_page.set_cropbox(new_page.rect)
        new_page.set_trimbox(new_page.rect)
        new_page.set_bleedbox(new_page.rect)
        new_page.set_artbox(new_page.rect)

        print(f"Page {page_index + 1}:")
        print(
            f"  old size: {page_rect.width:.2f} x {page_rect.height:.2f} pt "
            f"({pt_to_in(page_rect.width):.3f} x {pt_to_in(page_rect.height):.3f} in)"
        )
        print(
            f"  new size: {clip.width:.2f} x {clip.height:.2f} pt "
            f"({pt_to_in(clip.width):.3f} x {pt_to_in(clip.height):.3f} in)"
        )
        print(f"  method:   {info.get('method')}")
        print(f"  trimmed:  {info.get('detected')}")
        print(f"  reason:   {info.get('reason')}")

        print(
            f"  removed:  left={clip.x0 - page_rect.x0:.2f} pt, "
            f"top={clip.y0 - page_rect.y0:.2f} pt, "
            f"right={page_rect.x1 - clip.x1:.2f} pt, "
            f"bottom={page_rect.y1 - clip.y1:.2f} pt"
        )

        if info.get("scores"):
            scores = ", ".join(
                f"{side}={'-' if value is None else f'{value:.1f}'}"
                for side, value in info["scores"].items()
            )
            print(f"  scores:   {scores}")

    out.save(OUTPUT_PDF, garbage=4, deflate=True)
    out.close()
    src.close()

    print(f"\nSaved: {OUTPUT_PDF.resolve()}")


if __name__ == "__main__":
    main()