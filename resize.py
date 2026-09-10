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

# resize_pdf_final.py
# pip install pymupdf

import logging
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger(__name__)

INPUT_PDF = Path("yo3.pdf")
OUTPUT_PDF = Path("yoyo3.pdf")

TARGET_WIDTH_IN = 5.5
TARGET_HEIGHT_IN = 8.5
POINTS_PER_INCH = 72


def needs_flattening(page: fitz.Page) -> bool:
    """
    True when ``page``'s boxes are not already flat, i.e. it is not an unrotated
    page whose MediaBox is its visible page and starts at (0, 0).

    ``resize_doc`` scales the content about the PDF origin and rewrites every box
    to [0 0 w h]. That is only the right transform on a flat page. A MediaBox at
    a non-zero origin, a CropBox smaller than its MediaBox, or a /Rotate all
    leave the artwork scaled by the wrong factor or bodily displaced, and since
    the boxes are written out at the target size regardless, the result measures
    exactly right while the design sits off centre - on a cover, one panel gains
    what the other loses and the spine leaves the middle of the wrap.

    Callers must rebuild such a page through ``show_pdf_page`` first, which bakes
    origin, crop and rotation into the content and makes the scale below exact.
    """
    if page.rotation:
        return True

    media = page.mediabox

    return (
        abs(media.x0) > 0.01
        or abs(media.y0) > 0.01
        or abs(media.width - page.rect.width) > 0.5
        or abs(media.height - page.rect.height) > 0.5
    )


def resize_doc(doc: fitz.Document, target_w_in: float, target_h_in: float) -> fitz.Document:
    """
    Scale every page of ``doc`` to the target size in inches.

    The page contents are wrapped in a single reusable scaling transform, so
    artwork is scaled (not rasterized) and all standard page boxes are updated.
    Assumes uniform page sizes; scaling is computed from the first page.
    Mutates and returns ``doc``.
    """
    target_w = target_w_in * POINTS_PER_INCH
    target_h = target_h_in * POINTS_PER_INCH

    # Pages are uniform, so calculate scaling from the first page.
    #
    # Measured off the page rect, i.e. the visible (CropBox) page - the same
    # rectangle every caller measures and every detector renders. The MediaBox
    # can be larger (marks, slug) or sit at a non-zero origin, and scaling by it
    # would shrink the artwork by that surplus while the boxes below are still
    # written out at the target size, leaving the design short on one side and
    # overhanging on the other. Callers must hand over a page whose boxes are
    # already flat - see needs_flattening above.
    old_w = doc[0].rect.width
    old_h = doc[0].rect.height

    sx = target_w / old_w
    sy = target_h / old_h

    # x and y are scaled independently, so a source whose aspect ratio differs
    # from the target gets stretched. On a cover that is nearly always the wrong
    # operation - the panels are fixed trim sizes and only the spine grows with
    # the bulk - so say so rather than distorting the artwork quietly.
    if abs(sx - sy) > 0.002:
        log.warning(
            "resize: non-uniform scale sx=%.5f sy=%.5f - artwork will be "
            "stretched (%.4gx%.4g in -> %.4gx%.4g in)",
            sx, sy,
            old_w / POINTS_PER_INCH, old_h / POINTS_PER_INCH,
            target_w_in, target_h_in,
        )

    # Add one reusable scaling wrapper around existing page contents.
    prefix_xref = doc.get_new_xref()
    doc.update_object(prefix_xref, "<<>>")
    doc.update_stream(
        prefix_xref,
        f"q\n{sx:.12g} 0 0 {sy:.12g} 0 0 cm\n".encode("ascii")
    )

    suffix_xref = doc.get_new_xref()
    doc.update_object(suffix_xref, "<<>>")
    doc.update_stream(suffix_xref, b"\nQ\n")

    new_box = f"[0 0 {target_w:.12g} {target_h:.12g}]"

    for page in doc:
        original_contents = page.get_contents()

        # Wrap original content in the scaling transform.
        if original_contents:
            contents_refs = (
                [f"{prefix_xref} 0 R"]
                + [f"{xref} 0 R" for xref in original_contents]
                + [f"{suffix_xref} 0 R"]
            )
            doc.xref_set_key(page.xref, "Contents", "[" + " ".join(contents_refs) + "]")

        # Set all standard page boxes to the target size.
        for box_name in ("MediaBox", "CropBox", "TrimBox", "BleedBox", "ArtBox"):
            doc.xref_set_key(page.xref, box_name, new_box)

    return doc


def main():
    doc = fitz.open(INPUT_PDF)

    resize_doc(doc, TARGET_WIDTH_IN, TARGET_HEIGHT_IN)

    doc.save(OUTPUT_PDF)
    doc.close()

    print(f"Saved: {OUTPUT_PDF.resolve()}")


if __name__ == "__main__":
    main()
