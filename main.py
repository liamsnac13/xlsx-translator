import io
import os
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from openpyxl import load_workbook
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Grec moderne + polytonique
GREEK_RE = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")

def is_greek_text(v) -> bool:
    return isinstance(v, str) and bool(GREEK_RE.search(v))

def is_formula_cell(cell) -> bool:
    # openpyxl: data_type "f" = formula, mais on garde aussi un check sur string
    if getattr(cell, "data_type", None) == "f":
        return True
    v = cell.value
    return isinstance(v, str) and v.lstrip().startswith("=")

def chunked(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate")
async def translate(file: UploadFile = File(...), model: str = "gpt-4.1-mini"):
    fname = (file.filename or "").lower()
    if not (fname.endswith(".xlsx") or fname.endswith(".xlsm")):
        raise HTTPException(status_code=400, detail="Upload a .xlsx or .xlsm file")

    raw = await file.read()
    wb = load_workbook(io.BytesIO(raw), data_only=False, keep_vba=False)

    # Collecte des cellules à traduire : (sheet, addr, original)
    targets = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                if v is None:
                    continue
                # sécurité formules
                if is_formula_cell(cell):
                    continue
                # uniquement texte grec
                if is_greek_text(v):
                    targets.append((ws.title, cell.coordinate, v))

    # Rien à traduire -> renvoie le fichier tel quel
    if not targets:
        out = io.BytesIO()
        wb.save(out)
        out.seek(0)
        return StreamingResponse(
            out,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{file.filename[:-5]}_EN.xlsx"'}
        )

    # Cache pour répétitions (gros gain sur modèles financiers)
    cache = {}

    BATCH_SIZE = 80

    for batch in chunked(targets, BATCH_SIZE):
        # On garde seulement ceux non présents dans le cache
        to_translate = []
        idx_map = []  # (sheet, addr, orig) pour réinjecter

        for wsname, addr, orig in batch:
            if orig in cache:
                wb[wsname][addr].value = cache[orig]
            else:
                to_translate.append(orig)
                idx_map.append((wsname, addr, orig))

        if not to_translate:
            continue

        # Une ligne = une cellule, pour mapping 1:1
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
                        "Return ONLY the translated lines, EXACTLY one line per input line, same order, no numbering."
                    ),
                },
                {"role": "user", "content": user_text},
            ],
        )

        out_lines = resp.output_text.splitlines()

        # Vérif stricte : si mismatch, on stoppe plutôt que de corrompre le fichier
        if len(out_lines) != len(to_translate):
            raise HTTPException(
                status_code=500,
                detail=f"Translation line mismatch: expected {len(to_translate)}, got {len(out_lines)}"
            )

        for (wsname, addr, orig), translated in zip(idx_map, out_lines):
            cache[orig] = translated
            wb[wsname][addr].value = translated

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)

    out_name = f"{file.filename[:-5]}_EN.xlsx"
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'}
    )
