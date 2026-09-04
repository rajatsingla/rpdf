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

# main.py
# Lightweight FastAPI service exposing the cover/interior PDF fixers.
# Bytes in, bytes out: the PDF is sent as the raw request body and the fixed
# PDF is returned as the raw response body. No files are written to disk.
#
# Run:
#   uvicorn main:app --host 0.0.0.0 --port 8000

import logging
import os
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from fix_cover import fix_cover
from fix_interior_file import fix_interior_file

app = FastAPI(title="PDF Fix Service")

PDF_MEDIA_TYPE = "application/pdf"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Minimal file logging: one rotating file at logs/app.log (5 MB x 3).
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(os.path.dirname(__file__), "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"), maxBytes=5 * 1024 * 1024, backupCount=3
)
_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
)
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger("pdf_fix")

# AGPL-3.0 §13: anyone interacting with this service over a network must be
# offered its complete corresponding source. This service is a combined work
# with PyMuPDF (AGPL-3.0) and is itself licensed under AGPL-3.0-or-later.
# Override with SOURCE_URL if you deploy a modified version elsewhere.
SOURCE_URL = os.environ.get("SOURCE_URL", "https://github.com/rajatsingla/pdf_fix")


@app.middleware("http")
async def add_source_offer_header(request: Request, call_next):
    # Advertise the source offer to every network user, on every response.
    response = await call_next(request)
    response.headers["Link"] = f'<{SOURCE_URL}>; rel="source"'
    return response


@app.get("/rpdf/source")
def source() -> dict:
    """AGPL-3.0 source offer (see the LICENSE file at the repository root)."""
    return {"license": "AGPL-3.0-or-later", "source": SOURCE_URL}


# Allow the browser to call this API directly (no Node proxy). Override with
# ALLOW_ORIGINS=https://foo.com,https://bar.com ; default "*" for any origin.
_origins = ["https://stck.dev", "https://stck.me"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/rpdf/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/rpdf")
def index() -> FileResponse:
    # Serve the UI from the same origin as the API (no CORS/mixed-content issues).
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def _read_pdf_body(request: Request) -> bytes:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body")
    return body


@app.post("/rpdf/covers")
async def fix_cover_endpoint(
    request: Request,
    width_in: float = Query(..., gt=0, description="Final cover width in inches"),
    height_in: float = Query(..., gt=0, description="Final cover height in inches"),
) -> Response:
    body = await _read_pdf_body(request)
    log.info("cover: in=%d bytes size=%sx%s in", len(body), width_in, height_in)
    try:
        data = await run_in_threadpool(fix_cover, body, width_in, height_in)
    except Exception as exc:  # malformed/unsupported PDF -> 400, not 500
        log.exception("cover: failed")
        raise HTTPException(status_code=400, detail=f"failed to process PDF: {exc}")
    log.info("cover: out=%d bytes", len(data))
    return Response(content=data, media_type=PDF_MEDIA_TYPE)


@app.post("/rpdf/interiors")
async def fix_interior_endpoint(
    request: Request,
    is_domestic: bool = Query(
        False, description="Match against domestic trim sizes only"
    ),
) -> Response:
    body = await _read_pdf_body(request)
    log.info("interior: in=%d bytes is_domestic=%s", len(body), is_domestic)
    try:
        data = await run_in_threadpool(fix_interior_file, body, None, is_domestic)
    except Exception as exc:
        log.exception("interior: failed")
        raise HTTPException(status_code=400, detail=f"failed to process PDF: {exc}")
    log.info("interior: out=%d bytes", len(data))
    return Response(content=data, media_type=PDF_MEDIA_TYPE)
