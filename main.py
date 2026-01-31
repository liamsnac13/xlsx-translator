import io
import os
import re

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from openpyxl import load_workbook
from openai import OpenAI

app = FastAPI()

# OpenAI client (clé via variable d'env OPENAI_API_KEY)
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


def chunked(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate")
async def translate(file: UploadFile = File(...), model: str = "gpt-4.1-mini"):
    # --- Validate extension (xlsx/xlsm) ---
    fname = (file.filename or "").lower()
    if not (fname.endswith(".xlsx") or fname.endswith(".xlsm")):
        raise HTTPException(status_code=400, detail="Upload a .xlsx or .xlsm file")

    is_xlsm = fname.endswith(".xlsm")

    # --- Read input ---
    raw = await file.read()

    # Keep VBA only for xlsm
    try:
        wb = load_workbook(
            io.BytesIO(raw),
            data_only=False,          # keep formulas as formulas
            keep_vba=is_xlsm          # preserve macros if xlsm
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open workbook: {e}")

    # --- Collect cells to translate ---
    # targets: (sheet_name, cell_address, original_text)
    targets = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                if v is None:
                    continue
                if is_formula_cell(cell):
                    continue
                if is_greek_text(v):
                    targets.append((ws.title, cell.coordinate, v))

    # --- Output filename (keep ext) ---
    base = file.filename.rsplit(".", 1)[0] if file.filename else "translated"
    out_ext = "xlsm" if is_xlsm else "xlsx"
    out_name = f"{base}_EN.{out_ext}"

    # If nothing to translate, return as-is (but with _EN + correct ext)
    if not targets:
        out = io.BytesIO()
        wb.save(out)
        out.seek(0)
        return StreamingResponse(
            out,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{out_name}"'}
        )

    # --- Cache repeated strings ---
    cache = {}

    # 80 lignes par batch = OK en général. Tu peux baisser si timeout.
    BATCH_SIZE = 80

    for batch in chunked(targets, BATCH_SIZE):
        to_translate = []
        idx_map = []  # (wsname, addr, orig)

        for wsname, addr, orig in batch:
            if orig in cache:
                wb[wsname][addr].value = cache[orig]
            else:
                to_translate.append(orig)
                idx_map.append((wsname, addr, orig))

        if not to_translate:
            continue

        user_text = "\n".join(to_translate)

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

        # Strict mapping: 1 input line -> 1 output line
        if len(out_lines) != len(to_translate):
            raise HTTPException(
                status_code=500,
                detail=f"Translation line mismatch: expected {len(to_translate)}, got {len(out_lines)}"
            )

        for (wsname, addr, orig), translated in zip(idx_map, out_lines):
            cache[orig] = translated
            wb[wsname][addr].value = translated

    # --- Return translated workbook ---
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)

    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'}
    )
