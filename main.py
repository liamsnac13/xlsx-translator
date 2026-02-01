import os
import re
import json
import shutil
import tempfile
from typing import Dict, List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from openpyxl import load_workbook
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Greek (modern + polytonic)
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")


def is_greek_text(v) -> bool:
    return isinstance(v, str) and bool(GREEK_RE.search(v))


def is_formula_cell(cell) -> bool:
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


def translate_batch_json(lines: List[str], model: str) -> List[str]:
    """
    Robust translation:
    - Send JSON array of strings
    - Require JSON array output with same length
    This avoids "line mismatch" due to newline weirdness.
    """
    payload = json.dumps(lines, ensure_ascii=False)

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You translate Greek spreadsheet labels into clear English.\n"
                    "Rules:\n"
                    "- Keep numbers, dates, tickers, abbreviations, punctuation unchanged.\n"
                    "- Keep terminology consistent (Revenue, EBITDA, Working Capital).\n"
                    "- Input is a JSON array of strings.\n"
                    "- Output MUST be a JSON array of strings of the EXACT same length.\n"
                    "- Do NOT add commentary. Output JSON only."
                ),
            },
            {"role": "user", "content": payload},
        ],
    )

    text = (resp.output_text or "").strip()
    try:
        out = json.loads(text)
    except Exception:
        raise HTTPException(status_code=500, detail=f"Model did not return JSON. Got: {text[:200]}")

    if not isinstance(out, list) or not all(isinstance(x, str) for x in out):
        raise HTTPException(status_code=500, detail="Model output JSON is not a string array")

    if len(out) != len(lines):
        raise HTTPException(
            status_code=500,
            detail=f"Translation count mismatch: expected {len(lines)}, got {len(out)}",
        )

    return out


@app.post("/translate")
async def translate(
    file: UploadFile = File(...),
    model: str = Query(default="gpt-4.1-mini"),
    batch_size: int = Query(default=40, ge=10, le=120),
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

    # ---- Save upload to disk ----
    in_suffix = ".xlsm" if is_xlsm else ".xlsx"
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=in_suffix)
    tmp_in_path = tmp_in.name
    tmp_in.close()

    try:
        with open(tmp_in_path, "wb") as f:
            file.file.seek(0)
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        cleanup_files(tmp_in_path)
        raise HTTPException(status_code=400, detail=f"Failed to store upload: {e}")

    # ---- PASS 1: scan in read_only mode (lower RAM) ----
    # Collect only the coordinates + text (not whole cell objects)
    to_translate_coords: List[Tuple[str, str, str]] = []  # (sheet_name, cell_coord, original_text)
    try:
        wb_ro = load_workbook(
            filename=tmp_in_path,
            read_only=True,
            data_only=False,
            keep_vba=is_xlsm,
        )
        for ws in wb_ro.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    v = cell.value
                    if v is None:
                        continue
                    if is_formula_cell(cell):
                        continue
                    if is_greek_text(v):
                        to_translate_coords.append((ws.title, cell.coordinate, v))
    except Exception as e:
        cleanup_files(tmp_in_path)
        raise HTTPException(status_code=400, detail=f"Cannot read workbook: {e}")
    finally:
        try:
            wb_ro.close()
        except Exception:
            pass

    # Nothing to do -> return original as output (renamed)
    if not to_translate_coords:
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=f".{out_ext}")
        tmp_out_path = tmp_out.name
        tmp_out.close()
        try:
            # just copy input to output
            shutil.copyfile(tmp_in_path, tmp_out_path)
        except Exception as e:
            cleanup_files(tmp_in_path, tmp_out_path)
            raise HTTPException(status_code=500, detail=f"Failed to produce output: {e}")
        finally:
            cleanup_files(tmp_in_path)

        return FileResponse(
            path=tmp_out_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=out_name,
            background=BackgroundTask(cleanup_files, tmp_out_path),
        )

    # ---- Translate in batches (with caching) ----
    cache: Dict[str, str] = {}
    translations: Dict[Tuple[str, str], str] = {}  # (sheet, coord) -> translated

    batch_texts: List[str] = []
    batch_meta: List[Tuple[str, str, str]] = []

    for wsname, coord, text in to_translate_coords:
        if text in cache:
            translations[(wsname, coord)] = cache[text]
            continue
        batch_texts.append(text)
        batch_meta.append((wsname, coord, text))

        if len(batch_texts) >= batch_size:
            out_lines = translate_batch_json(batch_texts, model=model)
            for (s, c, orig), tr in zip(batch_meta, out_lines):
                cache[orig] = tr
                translations[(s, c)] = tr
            batch_texts.clear()
            batch_meta.clear()

    if batch_texts:
        out_lines = translate_batch_json(batch_texts, model=model)
        for (s, c, orig), tr in zip(batch_meta, out_lines):
            cache[orig] = tr
            translations[(s, c)] = tr

    # ---- PASS 2: open in normal mode and write translations ----
    try:
        wb = load_workbook(
            filename=tmp_in_path,
            data_only=False,
            keep_vba=is_xlsm,
        )
        for (wsname, coord), tr in translations.items():
            try:
                wb[wsname][coord].value = tr
            except Exception:
                # skip cell if something odd happens, but don't crash whole job
                continue
    except Exception as e:
        cleanup_files(tmp_in_path)
        raise HTTPException(status_code=500, detail=f"Failed to apply translations: {e}")

    # ---- Save output to disk ----
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

    cleanup_files(tmp_in_path)

    return FileResponse(
        path=tmp_out_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=out_name,
        background=BackgroundTask(cleanup_files, tmp_out_path),
    )
