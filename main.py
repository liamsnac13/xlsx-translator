import os
import re
import io
import shutil
import tempfile
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from openpyxl import load_workbook
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Grec moderne + polytonique
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")


def is_greek_text(v) -> bool:
    return isinstance(v, str) and bool(GREEK_RE.search(v))


def is_formula_cell(cell) -> bool:
    # openpyxl: data_type "f" = formula
    if getattr(cell, "data_type", None) == "f":
        return True
    v = cell.value
    return isinstance(v, str) and v.lstrip().startswith("=")


def cleanup_files(*paths: str) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate")
async def translate(
    file: UploadFile = File(...),
    model: str = Query(default="gpt-4.1-mini"),
    batch_size: int = Query(default=60, ge=10, le=200),
):
    # ---- Validate filename/ext ----
    original_name = file.filename or "input.xlsx"
    fname = original_name.lower()

    if not (fname.endswith(".xlsx") or fname.endswith(".xlsm")):
        raise HTTPException(status_code=400, detail="Upload a .xlsx or .xlsm file")

    is_xlsm = fname.endswith(".xlsm")
    out_ext = "xlsm" if is_xlsm else "xlsx"
    base = original_name.rsplit(".", 1)[0]
    out_name = f"{base}_EN.{out_ext}"

    # ---- Save upload to disk (avoid raw bytes in RAM) ----
    # Important on Railway: stop duplicating big buffers in memory.
    in_suffix = ".xlsm" if is_xlsm else ".xlsx"
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=in_suffix)
    tmp_in_path = tmp_in.name
    tmp_in.close()

    try:
        with open(tmp_in_path, "wb") as f:
            # stream copy upload -> disk
            file.file.seek(0)
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        cleanup_files(tmp_in_path)
        raise HTTPException(status_code=400, detail=f"Failed to store upload: {e}")

    # ---- Load workbook (still uses RAM, but less peak overall) ----
    try:
        wb = load_workbook(
            filename=tmp_in_path,
            data_only=False,
            keep_vba=is_xlsm,  # keep macros if xlsm
        )
    except Exception as e:
        cleanup_files(tmp_in_path)
        raise HTTPException(status_code=400, detail=f"Cannot open workbook: {e}")

    # ---- Translation loop (batch on the fly, no huge targets list) ----
    cache = {}  # repeated strings -> translated

    def translate_lines(lines: List[str]) -> List[str]:
        user_text = "\n".join(lines)

        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Translate Greek financial spreadsheet labels into clear English. "
                        "Do NOT change numbers, dates, tickers, abbreviations, or punctuation. "
                        "Keep terminology consistent (Revenue, EBITDA, Working Capital). "
                        "Return ONLY the translated lines, EXACTLY one line per input line, "
                        "same order, no numbering."
                    ),
                },
                {"role": "user", "content": user_text},
            ],
        )

        out_lines = resp.output_text.splitlines()
        if len(out_lines) != len(lines):
            raise HTTPException(
                status_code=500,
                detail=f"Translation line mismatch: expected {len(lines)}, got {len(out_lines)}",
            )
        return out_lines

    # Collect pending cells for a batch: (cell, original_text)
    pending: List[Tuple[object, str]] = []
    pending_texts: List[str] = []

    try:
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    v = cell.value
                    if v is None:
                        continue
                    if is_formula_cell(cell):
                        continue
                    if not is_greek_text(v):
                        continue

                    # cache hit => write immediately
                    if v in cache:
                        cell.value = cache[v]
                        continue

                    pending.append((cell, v))
                    pending_texts.append(v)

                    if len(pending_texts) >= batch_size:
                        translated = translate_lines(pending_texts)
                        for (c, orig), tr in zip(pending, translated):
                            cache[orig] = tr
                            c.value = tr
                        pending.clear()
                        pending_texts.clear()

        # flush remaining
        if pending_texts:
            translated = translate_lines(pending_texts)
            for (c, orig), tr in zip(pending, translated):
                cache[orig] = tr
                c.value = tr

    except HTTPException:
        # re-raise cleanly
        cleanup_files(tmp_in_path)
        raise
    except Exception as e:
        cleanup_files(tmp_in_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    # ---- Save output to disk (avoid BytesIO RAM spike) ----
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=f".{out_ext}")
    tmp_out_path = tmp_out.name
    tmp_out.close()

    try:
        wb.save(tmp_out_path)
    except Exception as e:
        cleanup_files(tmp_in_path, tmp_out_path)
        raise HTTPException(status_code=500, detail=f"Failed to save output: {e}")
    finally:
        try:
            wb.close()
        except Exception:
            pass

    # cleanup input immediately, keep output until response is sent
    cleanup_files(tmp_in_path)

    return FileResponse(
        path=tmp_out_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=out_name,
        background=BackgroundTask(cleanup_files, tmp_out_path),
    )
