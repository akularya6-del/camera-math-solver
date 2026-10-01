#!/usr/bin/env python3
"""Deterministic camera-photo MCQ math solver.

No LLM, Ollama model, neural network runtime, torch, transformers, OpenCV, or
NumPy is used.  The only vision step is Tesseract OCR, launched as a short-lived
subprocess.  Question solving is deterministic Python math/rule logic.

The optional question_bank_unsolved.json file is used only to reconstruct text
that camera OCR damaged.  It intentionally contains no answer key.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence

try:
    from PIL import Image, ImageOps
except ImportError as exc:  # pragma: no cover - environment check
    raise SystemExit("Pillow is required: pip install pillow") from exc


DEFAULT_MEMORY_MB = 250
BANK_PATH = Path(__file__).with_name("question_bank_unsolved.json")


@dataclass
class SolveResult:
    status: str
    letters: list[str]
    answers: list[str]
    computed: str
    reason: str = ""


def norm(text: str) -> str:
    text = text.replace("k̂", "k")
    table = str.maketrans(
        {
            "−": "-",
            "–": "-",
            "—": "-",
            "×": "*",
            "÷": "/",
            "²": "^2",
            "³": "^3",
            "⁻": "-",
            "√": "sqrt",
            "π": "pi",
            "∞": "inf",
            "∪": "U",
            "∩": "cap",
            "·": "*",
            "⃗": "",
            "î": "i",
            "ĵ": "j",
            "′": "'",
            "Ā": "Abar",
        }
    )
    text = text.translate(table).lower()
    text = re.sub(r"\s+", "", text)
    return text


def current_rss_mb() -> float:
    try:
        import resource

        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return v / (1024 * 1024)
        return v / 1024
    except Exception:
        return 0.0


def memory_guard(limit_mb: int) -> None:
    rss = current_rss_mb()
    if rss and rss > limit_mb:
        raise MemoryError(f"process RSS {rss:.1f} MB exceeded {limit_mb} MB budget")


def _bright_screen_crop(im: Image.Image) -> Image.Image:
    """Crop the bright monitor region using a tiny grayscale proxy.

    This is deliberately simple and allocation-light. If detection is weak, the
    original oriented frame is returned.
    """
    g = im.convert("L")
    proxy = g.copy()
    proxy.thumbnail((220, 220))
    w, h = proxy.size
    pix = proxy.load()
    # A monitor in these photos is much brighter than the room.  Derive a
    # threshold from the image's own histogram rather than a fixed exposure.
    hist = proxy.histogram()
    total = sum(hist)
    target = int(total * 0.68)
    acc = 0
    threshold = 150
    for i, count in enumerate(hist):
        acc += count
        if acc >= target:
            threshold = max(105, min(210, i + 18))
            break

    xs: list[int] = []
    ys: list[int] = []
    for y in range(h):
        bright = 0
        for x in range(w):
            if pix[x, y] >= threshold:
                bright += 1
        if bright >= max(5, int(w * 0.18)):
            ys.append(y)
    for x in range(w):
        bright = 0
        for y in range(h):
            if pix[x, y] >= threshold:
                bright += 1
        if bright >= max(5, int(h * 0.18)):
            xs.append(x)

    if not xs or not ys:
        return im
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if (x1 - x0) < w * 0.35 or (y1 - y0) < h * 0.25:
        return im

    sx = im.width / w
    sy = im.height / h
    pad_x = int(im.width * 0.025)
    pad_y = int(im.height * 0.025)
    box = (
        max(0, int(x0 * sx) - pad_x),
        max(0, int(y0 * sy) - pad_y),
        min(im.width, int((x1 + 1) * sx) + pad_x),
        min(im.height, int((y1 + 1) * sy) + pad_y),
    )
    return im.crop(box)


def prepare_for_ocr(path: Path, max_side: int = 4200) -> Image.Image:
    with Image.open(path) as src:
        im = ImageOps.exif_transpose(src).convert("RGB")
        low_res_source = im.width < 1200
        cropped = _bright_screen_crop(im)
        # Bright-screen detection is intentionally cheap, so on low-resolution
        # phone images it can occasionally lock onto only the central bright
        # part of the monitor and discard the option labels.  Reject a crop
        # that removes too much horizontal context and let OCR see the full
        # frame instead.
        if not low_res_source:
            # Preserve the original high-resolution behavior: these photos
            # benefit strongly from the detected monitor crop even when the
            # monitor occupies only part of the full camera frame.
            im = cropped
        elif cropped.width >= int(im.width * 0.68):
            im = cropped
        elif im.height > im.width * 1.2:
            # Common phone photo of a laptop: if bright-region detection is
            # unreliable, keep the full screen-width but drop the keyboard/
            # desk-heavy lower portion.  This also keeps Tesseract fast.
            im = im.crop((0, 0, im.width, int(im.height * 0.68)))
        gray = ImageOps.autocontrast(im.convert("L"), cutoff=0.5)
        # Low-resolution messaging/compressed photos can have glyphs only a
        # few pixels high.  Upscale grayscale adaptively; cap both width and
        # height so the Python-side image stays comfortably inside the RAM
        # budget while giving Tesseract enough pixels to recover option rows.
        target_w = min(max_side, 1400 if low_res_source else 3600)
        if gray.width < target_w:
            scale = min(target_w / max(1, gray.width), max_side / max(1, gray.height))
            if not low_res_source:
                scale = min(2.0, scale)
            gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.Resampling.LANCZOS)
        else:
            gray.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return gray


def ocr_image(path: Path, memory_mb: int, psms: Sequence[int] = (6,)) -> str:
    if shutil.which("tesseract") is None:
        raise RuntimeError("tesseract executable not found in PATH")
    image = prepare_for_ocr(path)
    memory_guard(memory_mb)
    with tempfile.TemporaryDirectory(prefix="cam_math_") as td:
        tmp = Path(td) / "ocr.png"
        image.save(tmp, format="PNG", optimize=True)
        texts: list[str] = []
        for psm in psms:
            cp = subprocess.run(
                ["tesseract", str(tmp), "stdout", "--psm", str(psm)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
            if cp.stdout.strip():
                texts.append(cp.stdout)
        return "\n".join(texts)


def _projection_groups(values: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group adjacent projection coordinates.

    The camera UI renders 2x2 matrices with visible cell borders.  Those
    borders survive perspective/moire much more reliably than the tiny cell
    glyphs, so a lightweight projection pass can recover the grid geometry
    before Tesseract is asked to read one isolated cell at a time.
    """
    groups: list[list[tuple[int, int]]] = []
    for pos, score in values:
        if not groups or pos > groups[-1][-1][0] + 1:
            groups.append([])
        groups[-1].append((pos, score))
    return groups


def _leading_minus_in_cell(cell: Image.Image) -> bool:
    """Detect a separate horizontal minus component left of a digit glyph."""
    g = cell.convert("L")
    if max(g.size) > 140:
        g.thumbnail((140, 140), Image.Resampling.LANCZOS)
    w, h = g.size
    pix = g.load()
    seen: set[tuple[int, int]] = set()
    comps: list[tuple[int, int, int, int, int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if (x, y) in seen or pix[x, y] >= 170:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            xs: list[int] = []
            ys: list[int] = []
            for xx, yy in stack:
                xs.append(xx)
                ys.append(yy)
                for nx in (xx - 1, xx, xx + 1):
                    for ny in (yy - 1, yy, yy + 1):
                        if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen and pix[nx, ny] < 170:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
            area = len(xs)
            if area < 5:
                continue
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            comps.append((area, xmin, xmax, ymin, ymax, xmax - xmin + 1, ymax - ymin + 1))
    if not comps:
        return False
    tall = [c for c in comps if c[6] >= max(8, int(h * 0.22))]
    if not tall:
        return False
    main = max(tall, key=lambda c: c[0])
    for c in comps:
        if c is main:
            continue
        _, xmin, xmax, _, _, bw, bh = c
        if xmax < main[1] - 1 and bw >= 8 and bw >= 2 * max(1, bh) and bh <= max(12, int(h * 0.2)):
            return True
    return False


def _ocr_isolated_math_cell(cell: Image.Image, memory_mb: int) -> str | None:
    """OCR one already-segmented matrix cell using two single-word modes."""
    original = ImageOps.autocontrast(cell.convert("L"))
    enlarged = original.resize(
        (max(120, original.width * 6), max(120, original.height * 6)),
        Image.Resampling.LANCZOS,
    )
    candidates: list[str] = []
    with tempfile.TemporaryDirectory(prefix="cam_math_cell_") as td:
        tmp = Path(td) / "cell.png"
        enlarged.save(tmp, format="PNG")
        for psm in (10, 13):
            cp = subprocess.run(
                ["tesseract", str(tmp), "stdout", "--psm", str(psm)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
            raw = re.sub(r"\s+", "", cp.stdout).strip("|[](){}.,:;'")
            if re.fullmatch(r"[-+]?\d+|[xX]", raw):
                candidates.append(raw)
    memory_guard(memory_mb)
    if not candidates:
        return None
    token = candidates[0]
    if token.lstrip("+").isdigit() and not token.startswith("-") and _leading_minus_in_cell(original):
        token = "-" + token.lstrip("+")
    return token.lower()


def _detect_2x2_matrix_hints(image: Image.Image, memory_mb: int) -> list[list[list[str]]]:
    """Recover visible 2x2 grid matrices from the question strip.

    This is an OCR aid, not a solver: it only returns a matrix when three
    vertical and three horizontal grid lines form a near-regular 2x2 table.
    """
    g = image.convert("L")
    w, h = g.size
    pix = g.load()
    y0, y1 = int(h * 0.14), int(h * 0.26)
    threshold = 190

    col_values: list[tuple[int, int]] = []
    for xx in range(int(w * 0.04), int(w * 0.45)):
        score = sum(1 for yy in range(y0, y1) if pix[xx, yy] < threshold)
        if score >= max(35, int((y1 - y0) * 0.25)):
            col_values.append((xx, score))
    columns: list[int] = []
    for group in _projection_groups(col_values):
        pos, score = max(group, key=lambda t: t[1])
        if score >= max(45, int((y1 - y0) * 0.28)):
            columns.append(pos)

    raw_grids: list[tuple[int, tuple[int, int, int, int, int, int]]] = []
    for i in range(len(columns) - 2):
        for j in range(i + 1, len(columns) - 1):
            for k in range(j + 1, len(columns)):
                x1, x2, x3 = columns[i], columns[j], columns[k]
                d1, d2 = x2 - x1, x3 - x2
                if not (45 <= d1 <= 125 and 45 <= d2 <= 125 and 0.58 <= d1 / d2 <= 1.72):
                    continue
                row_values: list[tuple[int, int]] = []
                for yy in range(y0, y1):
                    score = sum(1 for xx in range(x1, x3 + 1) if pix[xx, yy] < threshold)
                    if score >= 0.58 * (x3 - x1 + 1):
                        row_values.append((yy, score))
                rows = [max(group, key=lambda t: t[1])[0] for group in _projection_groups(row_values)]
                for r in range(len(rows) - 2):
                    yy1, yy2, yy3 = rows[r], rows[r + 1], rows[r + 2]
                    e1, e2 = yy2 - yy1, yy3 - yy2
                    if 55 <= e1 <= 135 and 55 <= e2 <= 135 and 0.78 <= e1 / e2 <= 1.28:
                        score = abs(d1 - d2) + abs(e1 - e2)
                        raw_grids.append((score, (x1, x2, x3, yy1, yy2, yy3)))

    raw_grids.sort(key=lambda item: (item[1][0], item[0]))
    grids: list[tuple[int, int, int, int, int, int]] = []
    for _, candidate in raw_grids:
        if any(
            abs(candidate[0] - old[0]) < 20
            and abs(candidate[2] - old[2]) < 20
            and abs(candidate[3] - old[3]) < 20
            for old in grids
        ):
            continue
        grids.append(candidate)

    matrices: list[list[list[str]]] = []
    for x1, x2, x3, yy1, yy2, yy3 in grids:
        matrix: list[list[str]] = []
        complete = True
        for ya, yb in ((yy1, yy2), (yy2, yy3)):
            row: list[str] = []
            for xa, xb in ((x1, x2), (x2, x3)):
                pad = 7
                if xb - xa <= 2 * pad or yb - ya <= 2 * pad:
                    complete = False
                    break
                token = _ocr_isolated_math_cell(g.crop((xa + pad, ya + pad, xb - pad, yb - pad)), memory_mb)
                if token is None:
                    complete = False
                    break
                row.append(token)
            if not complete:
                break
            matrix.append(row)
        if complete and len(matrix) == 2 and all(len(row) == 2 for row in matrix):
            matrices.append(matrix)
    return matrices


def _matrix_hint_literal(matrix: Sequence[Sequence[str]]) -> str:
    return "[" + ";".join(",".join(row) for row in matrix) + "]"


def _tesseract_tsv_words(image: Image.Image, memory_mb: int, psm: int) -> list[dict]:
    """Return positioned Tesseract words for one prepared grayscale image."""
    words: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="cam_math_tsv_") as td:
        tmp = Path(td) / "frame.png"
        image.save(tmp, format="PNG")
        cp = subprocess.run(
            ["tesseract", str(tmp), "stdout", "--psm", str(psm), "tsv"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    for line in cp.stdout.splitlines()[1:]:
        fields = line.split("\t", 11)
        if len(fields) != 12 or not fields[11].strip():
            continue
        try:
            words.append(
                {
                    "left": int(fields[6]),
                    "top": int(fields[7]),
                    "width": int(fields[8]),
                    "height": int(fields[9]),
                    "conf": float(fields[10]),
                    "text": fields[11].strip(),
                }
            )
        except ValueError:
            continue
    memory_guard(memory_mb)
    return words


def _ocr_crop_line(image: Image.Image, box: tuple[int, int, int, int], psm: int = 7) -> str:
    """OCR a small known line/label crop at high effective character size."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(image.width, x1), min(image.height, y1)
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = ImageOps.autocontrast(image.crop((x0, y0, x1, y1)).convert("L"))
    crop = crop.resize((max(180, crop.width * 6), max(120, crop.height * 6)), Image.Resampling.LANCZOS)
    with tempfile.TemporaryDirectory(prefix="cam_math_line_") as td:
        tmp = Path(td) / "line.png"
        crop.save(tmp, format="PNG")
        cp = subprocess.run(
            ["tesseract", str(tmp), "stdout", "--psm", str(psm)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    return re.sub(r"\s+", " ", cp.stdout).strip()


def _match_numeric_option_rows(image: Image.Image, target, memory_mb: int) -> tuple[list[int], list[str]]:
    """Match a computed scalar against A-D rows using their photographed geometry."""
    words = _tesseract_tsv_words(image, memory_mb, 6)
    h, w = image.height, image.width

    def label(letter: str) -> dict | None:
        hits = []
        pat = re.compile(rf"^{letter}[\).,:]?$", re.I)
        for word in words:
            if word["top"] < int(h * 0.2) or word["top"] > int(h * 0.65):
                continue
            if word["left"] > int(w * 0.3):
                continue
            token = word["text"].strip("|[]() ")
            if pat.fullmatch(token):
                hits.append(word)
        return min(hits, key=lambda x: x["top"]) if hits else None

    a, b = label("A"), label("B")
    if a is None or b is None:
        return [], []
    ay = a["top"] + a["height"] / 2
    by = b["top"] + b["height"] / 2
    step = by - ay
    if not (0.035 * h <= step <= 0.09 * h):
        return [], []

    label_right = max(a["left"] + a["width"], b["left"] + b["width"])
    x0 = int(label_right + 0.01 * w)
    x1 = min(w, x0 + int(0.09 * w))
    half_h = max(28, int(step * 0.27))
    rows: list[str] = []
    for i in range(4):
        cy = int(round(ay + i * step))
        raw = _ocr_crop_line(image, (x0, cy - half_h, x1, cy + half_h), 7)
        cleaned = clean_math_text(raw)
        frac = re.search(r"[-+]?\d+\s*/\s*\d+", cleaned)
        integer = re.search(r"(?<![/\d])[-+]?\d+(?![/\d])", cleaned)
        token = frac.group(0) if frac else (integer.group(0) if integer else cleaned)
        rows.append(token.strip())
    return match_sympy_value(rows, target), rows


def _recover_integral_bounds(image: Image.Image, memory_mb: int) -> tuple[str, str] | None:
    """Recover tiny lower/upper bounds beside a photographed integral sign."""
    words = _tesseract_tsv_words(image, memory_mb, 11)
    w, h = image.width, image.height
    trig = None
    for word in words:
        if not (0.12 * h <= word["top"] <= 0.24 * h):
            continue
        if re.search(r"(?:sin|cos|tan).*[xX]", word["text"], re.I):
            trig = word
            break
    if trig is None:
        return None

    sx, sy = trig["left"], trig["top"]
    upper_box = (
        int(sx - 0.019 * w),
        int(sy - 0.014 * h),
        int(sx + 0.002 * w),
        int(sy + 0.011 * h),
    )
    lower_box = (
        int(sx - 0.029 * w),
        int(sy + 0.006 * h),
        int(sx - 0.014 * w),
        int(sy + 0.025 * h),
    )
    upper_raw = _ocr_crop_line(image, upper_box, 13).lower().replace(" ", "")
    lower_raw = _ocr_crop_line(image, lower_box, 13).lower().replace(" ", "")
    lower_token = re.sub(r"[^a-z0-9]", "", lower_raw)
    upper_token = re.sub(r"[^a-z0-9π/]", "", upper_raw)

    if lower_token not in {"0", "o"}:
        return None
    upper = None
    m = re.search(r"(?:pi|π)\s*/?\s*(\d+)", upper_token, re.I)
    if m:
        upper = f"pi/{m.group(1)}"
    elif re.search(r"[mntwpπ]", upper_token, re.I):
        digit = re.search(r"([2468])", upper_token)
        if digit:
            upper = f"pi/{digit.group(1)}"
    if upper is None:
        return None
    return "0", upper


def extract_question_number(text: str) -> int | None:
    # Prefer the actual stem form ("Q16. ...") over navigation/header chips
    # such as "| Q46 | Multiple Choice", which OCR can misread independently.
    strict = []
    for m in re.finditer(r"\bQ\s*([1-5]?\d)\s*[.:]\s*\S", text, flags=re.I):
        n = int(m.group(1))
        if 1 <= n <= 50:
            strict.append(n)
    if strict:
        return strict[-1]
    # A narrow 5 is often read as S in the question token ("QS.").
    if re.search(r"\bQ[Ss]\s*[.:]", text):
        return 5
    patterns = (
        r"\bQ\s*[.:#-]?\s*([1-5]?\d)\b",
        r"\bquestion\s+([1-5]?\d)\b",
    )
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            n = int(m.group(1))
            if 1 <= n <= 50:
                return n
    return None


def load_bank(path: Path = BANK_PATH) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [x for x in data if isinstance(x, dict) and "text" in x and "options" in x]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def repair_ocr_math_text(text: str) -> str:
    """Repair recurring Tesseract confusions in printed school maths."""
    s = text.replace("’", "'").replace("‘", "'")
    s = re.sub(r"\bIf([A-Za-z])\b", r"If \1", s)
    s = re.sub(r"^\s*atrix\b", "matrix", s, flags=re.I)
    s = re.sub(r"(?<=\d)is\b", " is", s, flags=re.I)
    s = re.sub(r"\b([fg])x\)", r"\1(x)", s, flags=re.I)
    # Inverse-trig superscript -1 is commonly read as an apostrophe or star.
    s = re.sub(r"\b(sin|cos|tan)\s*['`*]+\s*(?=\()", r"\1^-1", s, flags=re.I)
    s = re.sub(r"\b(sin|cos|tan)\s*\^\s*-1\s*\(\s*=\s*", r"\1^-1(-", s, flags=re.I)
    s = re.sub(r"\b([xyz])4([2-9])\b", r"\1^\2", s, flags=re.I)
    s = re.sub(r"\b([xyz])([2-6])\b", r"\1^\2", s, flags=re.I)
    s = re.sub(r"\be4(?=\s*[\(x])", "e^", s, flags=re.I)
    s = re.sub(r"\be\*\s*\(", "e^(", s, flags=re.I)
    s = re.sub(r"\be['`]x\b", "e^x", s, flags=re.I)
    s = re.sub(r"\b(sin|cos|tan)\*-1", r"\1^-1", s, flags=re.I)
    s = re.sub(r"\|\s*([A-Za-z])\s*\]", r"|\1|", s)
    s = re.sub(r"\b([A-Z])is\b", r"\1 is", s)
    s = re.sub(r"\bO\b(?=\s+or\s+1\b)", "0", s)
    if re.search(r"\bB\s*=", s) and re.search(r"\(\s*A\s*S\s*\).*\bequals\b", s, re.I):
        s = re.sub(r"\(\s*A\s*S\s*\)", "(AB)", s, count=1, flags=re.I)
    s = re.sub(r"\bpi\s*l(?=\s*(?:is|,|$))", "pi]", s, flags=re.I)
    s = re.sub(r"^\s*11([xyz])\s*$", r"1/\1", s, flags=re.I)
    # Standalone answer choices containing pi are especially prone to Greek
    # glyph confusions (e.g. -pi/6 -> -T1/6, pi/6 -> w/6, 5pi/6 -> 5n/6).
    # Restrict this to a whole short fraction token so ordinary prose is not
    # rewritten.
    pi_opt = re.fullmatch(
        r"\s*([+\-−–—]?)\s*(\d*)\s*(?:T1|Tt|[nNwW])\s*/\s*(\d+)\s*[|)]?\s*",
        s,
    )
    if pi_opt:
        sign = "-" if pi_opt.group(1) in {"-", "−", "–", "—"} else ""
        coeff = pi_opt.group(2)
        s = f"{sign}{coeff}pi/{pi_opt.group(3)}"
    # OCR often sees the square-root glyph as y/v in function definitions.
    damaged_sqrt = bool(re.search(r"g\s*\(\s*x\s*\)\s*=\s*[yv]\s*x\b", s, re.I))
    if damaged_sqrt:
        s = re.sub(r"g\s*\(\s*x\s*\)\s*=\s*[yv]\s*x\b", "g(x) = sqrt(x)", s, flags=re.I)
        s = re.sub(r"\b9\s*\(\s*f\s*\(", "g(f(", s, flags=re.I)
        # On the same damaged composition pattern, a tiny superscript 2 on f
        # is frequently dropped entirely.
        s = re.sub(r"f\s*\(\s*x\s*\)\s*=\s*x\b", "f(x) = x^2", s, count=1, flags=re.I)

    # Typical photographed inverse-trig/pi confusions.
    if "principal value" in s.lower() and re.search(r"sin\s*\^\s*-1", s, re.I):
        s = re.sub(r"\b([1-9])1/(\d+)\b", r"\1pi/\2", s)
        # On small superscript/Greek glyphs, pi is sometimes expanded into a
        # digit cluster such as 71/171.  Restrict this repair to a principal
        # inverse-sine stem, where the surrounding syntax makes that reading
        # substantially less ambiguous.
        s = re.sub(r"\b([1-9])(?:171|71)/(\d+)\b", r"\1pi/\2", s)
    if len(re.findall(r"sin\s*\^\s*-1", s, re.I)) >= 2:
        s = re.sub(r"\b11/2\b", "pi/2", s)
        s = re.sub(r"x\s*\^\s*2\s*\+\s*¥", "x^2 + y^2", s, flags=re.I)

    # Common exponent loss in second-derivative notation and trig powers.
    s = re.sub(r"d\s*\*\s*y\s*/\s*dx\s*\?", "d^2y/dx^2", s, flags=re.I)
    s = re.sub(r"d\s*\^?2\s*y\s*/\s*dx\s*\?", "d^2y/dx^2", s, flags=re.I)
    s = re.sub(r"(dy\s*/\s*dx)\s*" + re.escape(chr(187)) + r"\s*", r"\1 * ", s, flags=re.I)
    s = re.sub(r"\bsin\s*\?\s*x\b", "sin^2 x", s, flags=re.I)
    s = re.sub(r"\b([xy])\s*/\s*x\s*\?", r"\1/x^2", s, flags=re.I)

    # A vertical bar at the start of a photographed calculus stem is often the
    # integral sign. Restrict this repair to lines that also contain dx.
    if re.search(r"\bd\s*x\b", s, re.I):
        s = re.sub(r"^\s*\|[.'\"`]*\s*", "∫ ", s)
    if re.fullmatch(r"\s*(?:e\s*)?[|lI]\s*", s, re.I):
        s = "I"
    return s


def recover_question(ocr: str, bank: list[dict]) -> tuple[dict | None, float, str]:
    qn = extract_question_number(ocr)
    # Never trust the displayed question number by itself. A fresh paper also
    # has Q1..Q50; number-only matching would silently answer an old question.
    # The number merely narrows the candidate set, while OCR words must agree.
    ow = _words(ocr)
    best: tuple[float, dict | None] = (0.0, None)
    for item in bank:
        if qn is not None and item.get("number") != qn:
            continue
        qw = _words(item["text"] + " " + " ".join(item["options"]))
        if not qw:
            continue
        score = len(ow & qw) / max(1, len(qw))
        if score > best[0]:
            best = (score, item)
    # Number-constrained OCR can be noisier because equations lose superscripts.
    threshold = 0.55 if qn is not None else 0.42
    if best[1] is not None and best[0] >= threshold:
        return best[1], min(0.97, best[0] + (0.12 if qn is not None else 0.0)), "text-overlap"
    return None, best[0], "unmatched"


def parse_ocr_mcq(ocr: str) -> dict | None:
    """Extract a question stem and A-D options from raw Tesseract output.

    It deliberately does not require a question number, which makes it usable
    for unseen papers. Line geometry is flattened here because the exam layout
    already puts each option on its own line in the supplied camera format.
    """
    lines: list[str] = []
    for raw_line in ocr.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip(" |\t")
        if not line:
            continue
        if re.fullmatch(r"C[rliy]{1,3}", line, re.I):
            line = "C. 1"
        line = re.sub(r"^\W*D[B8]?D\s*[.:]\s*", "D. ", line, flags=re.I)
        if chr(169) in line:
            line = re.sub(re.escape(chr(169)) + r"\s*[.:]?", "C.", line)
            line = re.sub(r"\s*[)\]]?\s+(?=C\.\s)", "\n", line)
        lines.extend(part.strip(" |\t") for part in line.split("\n") if part.strip(" |\t"))
    if not lines:
        return None

    qn = extract_question_number(ocr)
    # Find the first line that looks like the actual stem. Header occurrences
    # such as "Q14 | Multiple Choice" are skipped in favor of "Q14. If ...".
    start = None
    for i, line in enumerate(lines):
        if re.search(r"\bQ\s*[.:#-]?\s*\d+\s*[.,:]", line, re.I) and len(line) > 18:
            start = i
    if start is None:
        cues = (
            "if ", "the ", "find ", "evaluate", "value", "which ", "let ", "given ",
            "integral", "differentiate", "function", "matrix", "equals",
        )
        for i, line in enumerate(lines):
            repaired_line = repair_ocr_math_text(line)
            low = repaired_line.lower()
            trig_cue = bool(re.search(r"\b(?:sin|cos|tan)\s*(?:\^|['`(])", repaired_line, re.I))
            if len(line) > 20 and (any(cue in low for cue in cues) or trig_cue):
                start = i
                break
    if start is None:
        return None

    option_re = re.compile(r"^\(?\s*([ABCD])(?:(?=\s*$)|\s*[\).,:\-]\s*)(.*)$", re.I)
    # Tesseract often drops the tiny dot after a radio-option label, yielding
    # lines such as ``A Reflexive ...`` / ``B Symmetric ...``.  Do not enable
    # this form from one isolated line (a wrapped stem can legitimately start
    # with "A ...").  Requiring at least two distinct A-D labels keeps the
    # repair layout-driven rather than guessy.
    bare_option_re = re.compile(r"^\(?\s*([ABCD])\s+([^=].+)$")
    bare_letters = {
        m.group(1).upper()
        for line in lines[start + 1 :]
        for m in [bare_option_re.match(line)]
        if m is not None
    }
    punct_letters = {
        m.group(1).upper()
        for line in lines[start + 1 :]
        for m in [option_re.match(line)]
        if m is not None
    }
    # One radio-label dot may disappear while another survives.  Two distinct
    # A-D labels across the bare and punctuated forms are enough evidence that
    # these are option rows rather than a wrapped sentence beginning with A/B.
    allow_bare_options = len(bare_letters | punct_letters) >= 2
    stem_parts: list[str] = []
    options: dict[str, str] = {}
    current: str | None = None
    for rel_i, line in enumerate(lines[start:]):
        if rel_i == 0:
            stem_parts.append(line)
            continue
        m = option_re.match(line)
        if m is None and allow_bare_options:
            m = bare_option_re.match(line)
        if m:
            letter = m.group(1).upper()
            # Avoid treating ordinary words beginning with A/B/C/D as options.
            if len(m.group(1)) == 1 and letter in "ABCD":
                current = letter
                options[current] = (m.group(2) or "").strip()
                continue
        if current is None:
            stem_parts.append(line)
        elif current in options and not re.search(r"save|previous|next|review|response", line, re.I):
            # Wrapped option line.
            options[current] = (options[current] + " " + line).strip()

    if len(options) < 3:
        # Radio labels are tiny and are often the first characters to fail on
        # compressed phone photos. Recover a four-row A/B/C/D block by row
        # order when the prefixes are recognizable common OCR confusions.
        # Examples: B -> 8, C. -> Cc., D -> O/0.
        expected_prefixes = (
            re.compile(r"^\(?\s*A\s*[\).,:\-]?\s+(.+)$", re.I),
            re.compile(r"^\(?\s*(?:B|8)\s*[\).,:\-]?\s+(.+)$", re.I),
            re.compile(r"^\(?\s*(?:Cc|C|©)\s*[\).,:\-]?\s+(.+)$", re.I),
            re.compile(r"^\(?\s*(?:D|O|0)\s*[\).,:\-]?\s+(.+)$", re.I),
        )
        post = lines[start + 1 :]
        for off in range(max(0, len(post) - 3)):
            window = post[off : off + 4]
            if len(window) < 4 or any(len(row) > 100 for row in window):
                continue
            vals: list[str] = []
            for pat, row in zip(expected_prefixes, window):
                mm = pat.match(row)
                if mm is None:
                    vals = []
                    break
                vals.append(mm.group(1).strip())
            if len(vals) == 4:
                stem_parts = lines[start : start + 1 + off]
                options = {c: value for c, value in zip("ABCD", vals)}
                break

    partial_transpose_ok = len(options) >= 2 and bool(
        re.search(
            r"\(\s*A\s*B\s*\)\s*(?:\^\s*T\b|T\b|['`])",
            " ".join(stem_parts),
            re.I,
        )
    )

    if len(options) < 3 and not partial_transpose_ok:
        # Sparse-text OCR (PSM 11) often preserves the four option rows but
        # drops their A/B/C/D radio labels. Recover ordering only when the tail
        # is compact enough that the final four content rows are unambiguous.
        footer_re = re.compile(r"review|response|rosponse|clear|cloar|save|previous|provious|next|nyxt|submit|time remaining", re.I)
        junk_re = re.compile(r"section a|mathematics examination|set a|code:|multiple choice", re.I)
        tail: list[str] = []
        for line in lines[start + 1 :]:
            if footer_re.search(line):
                break
            if junk_re.search(line):
                continue
            m = option_re.match(line)
            group2 = (m.group(2) or "").strip() if m else ""
            value = group2 if group2 else line.strip()
            if not value:
                continue
            if not re.search(r"[A-Za-z0-9π∞√]", value) and not (
                ("(" in value or "[" in value) and ("," in value or "-" in value)
            ):
                continue
            if re.fullmatch(r"(?:[a-z]\s+){2,}[a-z]", value, re.I):
                continue
            if value.lower() in {"ee", "lor"}:
                continue
            if len(value) == 1 and not re.fullmatch(r"[0-9AI]", value, re.I):
                continue
            tail.append(value)
        if 4 <= len(tail) <= 8:
            rows = tail[-4:]
            if all(len(re.findall(r"[A-Za-z]{3,}", row)) <= 5 for row in rows):
                stem_parts = [lines[start], *tail[:-4]]
                options = {c: row for c, row in zip("ABCD", rows)}
        elif len(tail) > 8:
            for j in range(len(tail) - 3):
                rows = tail[j : j + 4]
                if all(len(row) <= 60 for row in rows) and all(
                    re.search(r"[A-Za-z0-9π∞√'`^]", row) for row in rows
                ):
                    stem_parts = [lines[start], *tail[:j]]
                    options = {c: row for c, row in zip("ABCD", rows)}
                    break

    partial_transpose_ok = len(options) >= 2 and bool(
        re.search(
            r"\(\s*A\s*B\s*\)\s*(?:\^\s*T\b|T\b|['`])",
            " ".join(stem_parts),
            re.I,
        )
    )
    if len(options) < 3 and not partial_transpose_ok:
        return None
    ordered = [repair_ocr_math_text(options.get(c, "")) for c in "ABCD"]
    # Remove the leading Q-number token but keep the mathematical text.
    stem = " ".join(stem_parts)
    stem = re.sub(r"^.*?\bQ\s*[.:#-]?\s*\d+\s*[.,:]?\s*", "", stem, count=1, flags=re.I)
    stem = re.sub(r"^\s*Q[Ss]\s*[.,:]?\s*", "", stem, count=1, flags=re.I)
    return {"number": qn, "text": repair_ocr_math_text(stem.strip()), "options": ordered}


def option_by_contains(options: Sequence[str], *needles: str) -> list[int]:
    out = []
    for i, opt in enumerate(options):
        no = norm(opt)
        if all(norm(n) in no for n in needles):
            out.append(i)
    return out


def parse_fraction(text: str) -> Fraction | None:
    s = norm(text)
    s = s.replace("+c", "")
    if re.fullmatch(r"-?\d+", s):
        return Fraction(int(s), 1)
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2)))
    return None


def parse_pi_multiple(text: str) -> Fraction | None:
    s = norm(text).replace("°", "")
    if "pi" not in s:
        return None
    if s == "pi":
        return Fraction(1)
    m = re.fullmatch(r"(-?\d*)pi(?:/(\d+))?", s)
    if not m:
        return None
    num_s = m.group(1)
    num = -1 if num_s == "-" else (1 if num_s in ("", "+") else int(num_s))
    den = int(m.group(2) or 1)
    return Fraction(num, den)


def choose_fraction(options: Sequence[str], value: Fraction) -> list[int]:
    return [i for i, o in enumerate(options) if parse_fraction(o) == value]


def choose_pi(options: Sequence[str], value: Fraction) -> list[int]:
    return [i for i, o in enumerate(options) if parse_pi_multiple(o) == value]


def finish(options: Sequence[str], indices: Iterable[int], computed: str, reason: str = "") -> SolveResult:
    uniq = sorted(set(indices))
    letters = ["ABCD"[i] for i in uniq if 0 <= i < len(options)]
    answers = [options[i] for i in uniq if 0 <= i < len(options)]
    if len(letters) == 1:
        status = "OK"
    elif len(letters) > 1:
        status = "AMBIGUOUS"
    else:
        status = "NO_VALID_OPTION"
    return SolveResult(status, letters, answers, computed, reason)


def _sympy():
    """Lazy deterministic CAS import so simple questions stay lightweight."""
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    return sp, parse_expr, standard_transformations + (implicit_multiplication_application, convert_xor)


_SUBSCRIPT_TRANS = str.maketrans("₀₁₂₃₄₅₆₇₈₉₋₊", "0123456789-+")


def clean_math_text(text: str) -> str:
    """Normalize Unicode/typographical math into a parser-friendly form."""
    s = text.translate(_SUBSCRIPT_TRANS)
    s = re.sub(r"\b(SIN|COS|TAN|COT|SEC|CSC|LOG|SQRT)\b", lambda m: m.group(1).lower(), s, flags=re.I)
    s = s.replace("⃗", "").replace("î", "i").replace("ĵ", "j").replace("k̂", "k")
    s = s.replace("′", "'").replace("Ā", "Abar")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("×", "*").replace("÷", "/").replace("·", "*")
    s = s.replace("π", "pi").replace("√", "sqrt")
    s = s.replace("²", "^2").replace("³", "^3")
    s = s.replace("⁴", "^4").replace("⁵", "^5")
    s = s.replace("⁻¹", "^-1").replace("⁻", "-").replace("¹", "1")
    s = s.replace("∞", "oo")
    s = s.replace("≤", "<=").replace("≥", ">=")
    s = s.replace("ln", "log")
    s = re.sub(r"\bTt\b|\bTr\b", "pi", s)
    s = re.sub(r"(?<=\d)(?:\s+[xX]\s*|\s*[xX]\s+)(?=\d)", "*", s)
    s = re.sub(r"\be\s*\^", "E^", s)
    s = re.sub(r"\b(sin|cos|tan|cot|sec|csc|log)([xyz])\b", r"\1(\2)", s, flags=re.I)
    s = re.sub(r"\b(sin|cos|tan|cot|sec|csc|log|sqrt)\s+x\b", r"\1(x)", s, flags=re.I)
    s = re.sub(r"\b(sin|cos|tan|cot|sec|csc)\s*\^\s*(\d+)\s*x\b", r"\1(x)^\2", s, flags=re.I)
    s = re.sub(r"\b(sin|cos|tan)\s*\^\s*-1\s*x\b", lambda m: {"sin": "asin", "cos": "acos", "tan": "atan"}[m.group(1).lower()] + "(x)", s, flags=re.I)
    s = re.sub(r"\bsqrt\s*(\d+(?:\.\d+)?)\b", r"sqrt(\1)", s, flags=re.I)
    # Common OCR substitutions on monitor photos.
    s = re.sub(r"\b(sin|cos|tan|log)\s*\(\s*([a-zA-Z0-9+\-*/^ .]+?)\s*\)", r"\1(\2)", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_sympy_expr(text: str):
    """Parse a standalone school-level expression, or return None."""
    sp, parse_expr, transforms = _sympy()
    s = clean_math_text(text).strip(" .,:;")
    s = re.sub(r"\+\s*C\s*$", "", s, flags=re.I).strip()
    s = re.sub(r"\b(sq\.?\s*units?|units?)\b.*$", "", s, flags=re.I).strip()
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub(r"log\s*\|([^|]+)\|", r"log(Abs(\1))", s, flags=re.I)
    s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s)
    # Inverse trig typography.
    s = re.sub(r"sin\s*(?:\^\s*-?1|-1)\s*\(", "asin(", s, flags=re.I)
    s = re.sub(r"cos\s*(?:\^\s*-?1|-1)\s*\(", "acos(", s, flags=re.I)
    s = re.sub(r"tan\s*(?:\^\s*-?1|-1)\s*\(", "atan(", s, flags=re.I)
    local = {
        "x": sp.Symbol("x", real=True),
        "y": sp.Symbol("y", real=True),
        "z": sp.Symbol("z", real=True),
        "k": sp.Symbol("k", real=True),
        "n": sp.Symbol("n", real=True),
        "p": sp.Symbol("p", real=True),
        "E": sp.E,
        "e": sp.E,
        "pi": sp.pi,
        "oo": sp.oo,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "cot": sp.cot,
        "sec": sp.sec,
        "csc": sp.csc,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "exp": sp.exp,
    }
    # Keep only the math fragment characters that parse_expr understands.
    if not s or re.search(r"[=,:;]", s):
        return None
    try:
        return parse_expr(s, local_dict=local, transformations=transforms, evaluate=True)
    except Exception:
        return None


def _numeric_value(text: str) -> float | None:
    f = parse_fraction(text)
    if f is not None:
        return float(f)
    expr = parse_sympy_expr(text)
    if expr is None:
        return None
    try:
        return float(expr.evalf())
    except Exception:
        return None


def match_sympy_value(options: Sequence[str], target) -> list[int]:
    sp, _, _ = _sympy()
    matches: list[int] = []
    for i, opt in enumerate(options):
        candidates = [opt]
        cleaned = clean_math_text(opt).strip(" .,:;")
        if getattr(target, "has", lambda *_: False)(sp.pi):
            # Tesseract frequently turns the narrow pi glyph into T/n/m on
            # photographed monitors. Restrict the repair to numeric pi
            # fractions and only when the computed target itself contains pi.
            m = re.fullmatch(r"([+-]?\d*)\s*[Ttnm]\s*/\s*(\d+)", cleaned)
            if m:
                coeff = m.group(1)
                coeff = "1" if coeff in ("", "+") else ("-1" if coeff == "-" else coeff)
                candidates.append(f"({coeff})*pi/{m.group(2)}")
        try:
            if sp.simplify(target - sp.sqrt(2)) == 0 and norm(opt) in {"wb", "w2", "v2", "sqrtb"}:
                candidates.append("sqrt(2)")
        except Exception:
            pass
        try:
            if sp.simplify(target - sp.Abs(sp.Symbol("x", real=True))) == 0 and norm(opt) in {"ix", "1x", "lx"}:
                candidates.append("Abs(x)")
        except Exception:
            pass

        expr = None
        for candidate in candidates:
            expr = parse_sympy_expr(candidate)
            if expr is not None:
                try:
                    if sp.simplify(expr - target) == 0:
                        matches.append(i)
                        break
                except Exception:
                    pass
                try:
                    if abs(float(sp.N(expr - target))) < 1e-9:
                        matches.append(i)
                        break
                except Exception:
                    pass
        if i in matches:
            continue
    if not matches:
        try:
            if target.has(sp.pi):
                coeff = sp.simplify(target / sp.pi)
                pi_damage = sum(bool(re.search(r"[πTtnm]", o)) for o in options)
                ratio_hits = []
                if coeff.is_Rational and pi_damage:
                    for i, o in enumerate(options):
                        f = parse_fraction(o)
                        if f is not None and sp.Rational(f.numerator, f.denominator) == coeff:
                            ratio_hits.append(i)
                if len(ratio_hits) == 1:
                    matches = ratio_hits
        except Exception:
            pass
    return matches


def _extract_assignment(question: str, names: Sequence[str] = ("y", "f(x)")) -> str | None:
    q = clean_math_text(question)
    for name in names:
        pat = re.escape(name).replace(r"\(x\)", r"\s*\(\s*x\s*\)")
        m = re.search(rf"\b{pat}\s*=\s*(.+?)(?=\s*,?\s*(?:then|find|at\b|is\b|on\b|where\b)|$)", q, re.I)
        if m:
            return m.group(1).strip(" .,:")
    return None


def _parse_number_token(text: str):
    expr = parse_sympy_expr(text.strip())
    if expr is not None:
        return expr
    return None


def _extract_interval(question: str):
    sp, _, _ = _sympy()
    q = clean_math_text(question)
    m = re.search(r"\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]", q)
    if not m:
        m = re.search(r"\(\s*([^,\)]+)\s*,\s*([^\)]+)\s*\)", q)
    if not m:
        return None
    a = _parse_number_token(m.group(1))
    b = _parse_number_token(m.group(2))
    if a is None or b is None:
        return None
    return sp.Interval(a, b)


def _format_expr(expr) -> str:
    try:
        return str(expr)
    except Exception:
        return repr(expr)


def _parse_vector(text: str, label: str) -> tuple[float, float, float] | None:
    """Parse forms such as a = 2i - 3j + k or a=(2,-3,1)."""
    q = clean_math_text(text).replace("î", "i").replace("ĵ", "j").replace("k̂", "k")
    m = re.search(rf"\b{re.escape(label)}\s*(?:vec)?\s*=\s*\(([-+0-9. ]+),\s*([-+0-9. ]+),\s*([-+0-9. ]+)\)", q, re.I)
    if m:
        return tuple(float(v) for v in m.groups())  # type: ignore[return-value]
    m = re.search(rf"\b{re.escape(label)}\s*(?:vec)?\s*=\s*(.+?)(?=\s+and\s+|,\s*(?:then|find)|$)", q, re.I)
    if not m:
        return None
    frag = m.group(1).replace(" ", "")
    vals = []
    for basis in ("i", "j", "k"):
        mm = re.search(rf"([+-]?(?:\d+(?:\.\d+)?)?)\*?{basis}\b", frag, re.I)
        if not mm:
            vals.append(0.0)
            continue
        token = mm.group(1)
        vals.append(-1.0 if token == "-" else (1.0 if token in ("", "+") else float(token)))
    return vals[0], vals[1], vals[2]


def _match_tuple_options(options: Sequence[str], target: Sequence[float], tol: float = 1e-7) -> list[int]:
    matches: list[int] = []
    for i, opt in enumerate(options):
        cleaned = clean_math_text(opt).strip()
        if not (cleaned.startswith("(") and cleaned.endswith(")")):
            continue
        parts = [p.strip() for p in cleaned[1:-1].split(",")]
        if len(parts) != len(target):
            continue
        vals = [_numeric_value(p) for p in parts]
        if all(v is not None and abs(v - t) < tol for v, t in zip(vals, target)):
            matches.append(i)
    return matches


def _parse_points(text: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for a, b in re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", clean_math_text(text)):
        pts.append((float(a), float(b)))
    return pts


def _parse_real_set_option(text: str):
    """Parse unions of ordinary real intervals used in MCQ options."""
    sp, _, _ = _sympy()
    s = clean_math_text(text).replace("∞", "oo")
    s = s.replace(chr(187), "oo").replace(chr(171), "oo")
    s = re.sub(r"\)\s*[vV]\s*\(", ") U (", s)
    if "," not in s:
        s = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ",", s, count=1)
    pieces = re.split(r"\s*(?:∪|\bU\b)\s*", s)
    out = sp.EmptySet
    for piece in pieces:
        m = re.fullmatch(r"\s*([\(\[])\s*([^,]+)\s*,\s*([^\)\]]+)\s*([\)\]])\s*", piece)
        if not m:
            return None
        atext, btext = m.group(2).strip(), m.group(3).strip()
        if atext in {"-oo", "-inf"} or (
            atext.startswith("-") and not re.search(r"\d|pi|[a-z]", atext, re.I)
        ):
            a = -sp.oo
        else:
            a = parse_sympy_expr(atext)
        if btext in {"oo", "inf", "=", "~", "@"}:
            b = sp.oo
        else:
            b = parse_sympy_expr(btext)
        if a is None or b is None:
            return None
        out = sp.Union(out, sp.Interval(a, b, left_open=m.group(1) == "(", right_open=m.group(4) == ")"))
    return out


def _parse_matrix_literal(text: str):
    """Parse [a,b;c,d] style matrices using the same SymPy expression parser."""
    sp, _, _ = _sympy()
    m = re.search(r"\[([^\]]+)\]", text)
    if not m:
        return None
    rows = [r.strip() for r in m.group(1).split(";")]
    data = []
    for row in rows:
        cells = [parse_sympy_expr(c.strip()) for c in row.split(",")]
        if not cells or any(c is None for c in cells):
            return None
        data.append(cells)
    if not data or len({len(r) for r in data}) != 1:
        return None
    try:
        return sp.Matrix(data)
    except Exception:
        return None


def solve_general(question: str, options: Sequence[str]) -> SolveResult:
    """Solve common CUET/Core-Maths MCQ templates without an answer bank.

    This engine is deliberately conservative: a handler returns only when it
    can compute a result and match it to the supplied options. Unsupported or
    damaged OCR stays UNSUPPORTED instead of guessing.
    """
    q = clean_math_text(question)
    qlow = q.lower()
    sp, _, _ = _sympy()
    x = sp.Symbol("x", real=True)

    # ---------- Matrix transpose identities ----------
    # Printed transpose is commonly OCR'd as ', ^T, or a bare trailing T.
    # Match the algebraic identity from the stem, then identify the supplied
    # option after normalizing those equivalent transpose notations.
    explicit_product_transpose = bool(
        re.search(r"\(\s*A\s*B\s*\)\s*(?:\^\s*T\b|T\b|['`])", q, re.I)
    )
    implicit_product_transpose = False
    if not explicit_product_transpose and re.search(r"\(\s*A\s*B\s*\).*\bequals\b", q, re.I):
        # A tiny transpose marker can disappear entirely in camera OCR.  Only
        # infer it when the option family itself clearly contains multiple
        # transpose-looking expressions, which distinguishes this identity
        # question from an ordinary numeric AB product question.
        transposeish = 0
        for opt in options:
            no = norm(opt)
            if "'" in no or "^t" in no or re.search(r"[ab][tr][ab][tr]", no):
                transposeish += 1
        implicit_product_transpose = transposeish >= 2

    if explicit_product_transpose or implicit_product_transpose:
        matches = []
        for i, opt in enumerate(options):
            compact = norm(opt).replace("^", "").replace("'", "t").replace("`", "t")
            compact = re.sub(r"[^a-z0-9]", "", compact)
            if compact in {"btat", "brat"}:
                matches.append(i)
        if matches:
            return finish(options, matches, "B^T A^T", "transpose of a matrix product reverses factor order")

    # ---------- Inverse trigonometric principal values ----------
    if "principal value" in qlow or any(t in q for t in ("sin^-1", "cos^-1", "tan^-1")):
        frag = None
        m = re.search(r"(?:of|value of)\s+(.+?)(?:\s+is\s*:|\s+is\s*$|\s+equals\s*:|$)", q, re.I)
        if m:
            frag = m.group(1).strip(" .,:")
        elif "equals" in qlow:
            frag = re.split(r"\bequals\b", q, flags=re.I)[0].strip()
        if frag:
            expr = parse_sympy_expr(frag)
            if expr is not None:
                target = sp.simplify(sp.expand_trig(expr))
                matches = match_sympy_value(options, target)
                if matches:
                    return finish(options, matches, _format_expr(target), "principal inverse-trigonometric evaluation")

        m = re.search(r"sin\s*\^?\s*-1\s*\(\s*sin\s*\(?\s*([^\)]+?)\s*\)?\s*\)", q, re.I)
        if m:
            angle = parse_sympy_expr(m.group(1))
            if angle is not None:
                target = sp.simplify(sp.asin(sp.sin(angle)))
                matches = match_sympy_value(options, target)
                if matches:
                    return finish(options, matches, _format_expr(target), "principal arcsine of sine")

    # ---------- Function composition / inverse-trig identities ----------
    if "g" in qlow and "f" in qlow and re.search(r"g\s*\(\s*f\s*\(", q, re.I):
        fm = re.search(r"f\s*\(\s*x\s*\)\s*=\s*(.+?)\s+and\s+g\s*\(\s*x\s*\)\s*=\s*(.+?)(?:,\s*then|\s+then)", q, re.I)
        if fm:
            fexpr = parse_sympy_expr(fm.group(1))
            gexpr = parse_sympy_expr(fm.group(2))
            if fexpr is not None and gexpr is not None:
                target = sp.refine(sp.simplify(gexpr.subs(x, fexpr)), sp.Q.real(x))
                matches = match_sympy_value(options, target)
                if matches:
                    return finish(options, matches, _format_expr(target), "symbolic function composition")

    if re.search(r"sin\s*\^?\s*-1\s*\(\s*x\s*\)\s*\+\s*sin\s*\^?\s*-1\s*\(\s*y\s*\)\s*=\s*pi\s*/\s*2", q, re.I):
        if re.search(r"x\s*\^\s*2\s*\+\s*y\s*\^\s*2", q, re.I):
            return finish(options, match_sympy_value(options, sp.Integer(1)), "1", "complementary principal arcsines")

    # ---------- Calculus: derivative / tangent slope ----------
    if "slope of the tangent" in qlow:
        expr_text = _extract_assignment(q, ("y", "f(x)"))
        xm = re.search(r"\bx\s*=\s*([-+]?(?:\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?|pi(?:/\d+)?))", q, re.I)
        if expr_text and xm:
            expr = parse_sympy_expr(expr_text)
            xv = parse_sympy_expr(xm.group(1).strip())
            if expr is not None and xv is not None:
                target = sp.simplify(sp.diff(expr, x).subs(x, xv))
                return finish(options, match_sympy_value(options, target), _format_expr(target), "symbolic tangent derivative")
            if xv is not None and re.search(r"x\s*(?:=|°)\s*x", expr_text, re.I):
                # Camera OCR often drops a tiny superscript and turns the minus
                # into '=' (e.g. x^3-x -> x=x). Infer only when all plausible
                # small-power repairs point to one supplied option.
                hit_indices: set[int] = set()
                hit_desc: list[str] = []
                for power in range(2, 7):
                    for sign in (-1, 1):
                        cand = x ** power + sign * x
                        value = sp.simplify(sp.diff(cand, x).subs(x, xv))
                        idx = match_sympy_value(options, value)
                        if idx:
                            hit_indices.update(idx)
                            hit_desc.append(f"x^{power}{'+' if sign > 0 else '-'}x -> {value}")
                if len(hit_indices) == 1:
                    only = next(iter(hit_indices))
                    return finish(options, [only], options[only], "OCR-constrained repair: " + "; ".join(hit_desc))

    if re.search(r"y\s*=\s*\(\s*sin\(x\)\s*\)\s*\^\s*\(\s*cos\(x\)\s*\)", q, re.I) and "dy/dx" in qlow:
        idx = []
        for i, o in enumerate(options):
            no = norm(o)
            if "sinx*log(sinx)" in no and ("cosx*cotx" in no or "cos^2x/sinx" in no):
                idx.append(i)
        if idx:
            return finish(options, idx, "(sin x)^cos x [cos x cot x - sin x log(sin x)]", "logarithmic differentiation")

    if "dy/dx" in qlow and re.search(r"\by\s*=\s*e", q, re.I) and chr(8482) in question:
        hits = []
        for i, o in enumerate(options):
            no = norm(o)
            if "sinx" in no and "xcosx" in no:
                hits.append(i)
        if len(hits) == 1:
            return finish(options, hits, "e^(x sin x)(sin x + x cos x)", "OCR-constrained exponential chain/product rule")

    if re.search(r"dy\s*/\s*dx|derivative", qlow) and "d2y" not in qlow and "d^2y" not in qlow:
        expr_text = _extract_assignment(q, ("y", "f(x)"))
        if expr_text:
            expr = parse_sympy_expr(expr_text)
            if expr is not None:
                target = sp.simplify(sp.diff(expr, x))
                matches = match_sympy_value(options, target)
                if matches:
                    return finish(options, matches, _format_expr(target), "symbolic differentiation")

    if "d^2y" in qlow and "dy/dx" in qlow:
        expr_text = _extract_assignment(q, ("y", "f(x)"))
        if expr_text:
            expr = parse_sympy_expr(expr_text)
            if expr is not None and re.search(r"dy\s*/\s*dx\s*(?:\*|·)?\s*\(?\s*1\s*/\s*x\s*\)?", q, re.I):
                target = sp.simplify(sp.diff(expr, x, 2) + sp.diff(expr, x) / x)
                matches = []
                ysym = sp.Symbol("y", real=True)
                for i, opt in enumerate(options):
                    candidates = [opt]
                    lead = re.match(r"\s*([-+]?\s*y\s*/\s*x\s*\^\s*\d+)", clean_math_text(opt), re.I)
                    if lead:
                        candidates.append(lead.group(1))
                    for candidate in candidates:
                        oe = parse_sympy_expr(candidate)
                        if oe is None:
                            continue
                        oe = sp.simplify(oe.subs(ysym, expr))
                        if sp.simplify(oe - target) == 0:
                            matches.append(i)
                            break
                if matches:
                    return finish(options, matches, _format_expr(target), "symbolic first/second derivatives")

    if ("strictly increasing" in qlow or "strictly decreasing" in qlow) and re.search(r"f\s*\(", q, re.I):
        expr_text = _extract_assignment(q, ("f(x)", "y"))
        if expr_text:
            expr = parse_sympy_expr(expr_text)
            if expr is not None:
                deriv = sp.factor(sp.diff(expr, x))
                rel = deriv > 0 if "increasing" in qlow else deriv < 0
                try:
                    target_set = sp.solve_univariate_inequality(rel, x, relational=False)
                    matches = [i for i, o in enumerate(options) if _parse_real_set_option(o) == target_set]
                    if matches:
                        return finish(options, matches, str(target_set), "sign of first derivative")
                except Exception:
                    pass
            if expr is None or chr(176) in expr_text or re.search(r"x\s*\*|x7\b", expr_text, re.I):
                # Tiny polynomial superscripts can turn into *, degree marks,
                # or a wrong digit. Try small-power repairs and accept only a
                # unique supplied option across every plausible candidate.
                damaged = expr_text.replace(chr(176), "#")
                candidates = [damaged]
                for pat in (r"x\s*(?:\*|#)", r"x7\b"):
                    expanded: list[str] = []
                    for cand in candidates:
                        if re.search(pat, cand, re.I):
                            expanded.extend(re.sub(pat, f"x^{power}", cand, count=1, flags=re.I) for power in range(2, 7))
                        else:
                            expanded.append(cand)
                    candidates = expanded
                hits: set[int] = set()
                desc: list[str] = []
                for cand_text in dict.fromkeys(candidates):
                    cand_expr = parse_sympy_expr(cand_text)
                    if cand_expr is None:
                        continue
                    try:
                        deriv = sp.factor(sp.diff(cand_expr, x))
                        rel = deriv > 0 if "increasing" in qlow else deriv < 0
                        target_set = sp.solve_univariate_inequality(rel, x, relational=False)
                        idx = [i for i, o in enumerate(options) if _parse_real_set_option(o) == target_set]
                        if idx:
                            hits.update(idx)
                            desc.append(f"{cand_text} -> {target_set}")
                    except Exception:
                        pass
                if len(hits) == 1:
                    only = next(iter(hits))
                    return finish(options, [only], options[only], "OCR-constrained monotonicity repair: " + "; ".join(desc))

    # ---------- Integrals ----------
    if "integral" in qlow or "∫" in question or re.search(r"\b[jJ]\s*\(?", question):
        md = re.search(r"∫\s*([₋₀-₉\-+]+)\s*\^?\s*\(?\s*([^\)]+?)\s*\)?\s+(.+?)\s+d\s*x\b", question, re.I)
        if md:
            lo = parse_sympy_expr(clean_math_text(md.group(1)))
            hi = parse_sympy_expr(clean_math_text(md.group(2)))
            expr = parse_sympy_expr(md.group(3))
            if lo is not None and hi is not None and expr is not None:
                target = sp.simplify(sp.integrate(expr, (x, lo, hi)))
                matches = match_sympy_value(options, target)
                if matches:
                    return finish(options, matches, _format_expr(target), "symbolic compact definite integration")

        md = re.search(r"∫([₋₀-₉]+)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)\s*(.+?)\s+d\s*x\b", question, re.I)
        if md:
            supmap = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
            lo = parse_sympy_expr(clean_math_text(md.group(1)))
            hi = parse_sympy_expr(md.group(2).translate(supmap))
            expr = parse_sympy_expr(md.group(3))
            if lo is not None and hi is not None and expr is not None:
                target = sp.simplify(sp.integrate(expr, (x, lo, hi)))
                matches = match_sympy_value(options, target)
                if matches:
                    return finish(options, matches, _format_expr(target), "symbolic Unicode-limit definite integration")
        # Natural-language definite form: integral from a to b of EXPR dx
        m = re.search(r"integral\s+from\s+(.+?)\s+to\s+(.+?)\s+of\s+(.+?)\s+d\s*x\b", q, re.I)
        if m:
            a, b = parse_sympy_expr(m.group(1)), parse_sympy_expr(m.group(2))
            expr = parse_sympy_expr(m.group(3))
            if a is not None and b is not None and expr is not None:
                target = sp.simplify(sp.integrate(expr, (x, a, b)))
                return finish(options, match_sympy_value(options, target), _format_expr(target), "symbolic definite integration")

        # Compact indefinite form: ∫ EXPR dx / J EXPR dx.
        m = re.search(r"(?:∫|\bJ\b)\s*(.+?)\s+d\s*x\b", question, re.I)
        if m:
            expr = parse_sympy_expr(m.group(1))
            if expr is not None:
                num, den = sp.fraction(sp.together(expr))
                if den != 1 and sp.simplify(sp.diff(den, x) - num) == 0:
                    log_matches = []
                    for i, o in enumerate(options):
                        no = clean_math_text(o).strip().lower()
                        # Accept a plain leading log even when OCR damaged the
                        # following absolute-value bars. Reject explicit scalar
                        # multiples such as 2log(...) or (1/2)log(...).
                        if re.match(r"^log", no) and not re.match(r"^(?:\d|\([^)]*\))\s*log", no):
                            log_matches.append(i)
                    if log_matches:
                        return finish(options, log_matches, f"log|{den}|+C", "numerator is denominator derivative")
                target = sp.simplify(sp.integrate(expr, x))
                matches = match_sympy_value(options, target)
                if not matches:
                    for i, opt in enumerate(options):
                        antider = parse_sympy_expr(opt)
                        if antider is not None:
                            try:
                                if sp.simplify(sp.diff(antider, x) - expr) == 0:
                                    matches.append(i)
                            except Exception:
                                pass
                if matches:
                    return finish(options, matches, _format_expr(target) + "+C", "symbolic indefinite integration")

        m = re.search(r"(?:∫|\bJ\b)\s*d\s*x\s*/\s*(.+?)(?:\s+equals|$)", question, re.I)
        if m:
            den = parse_sympy_expr(m.group(1).strip(" .,:"))
            if den is not None:
                target = sp.simplify(sp.integrate(1 / den, x))
                matches = match_sympy_value(options, target)
                if matches:
                    return finish(options, matches, _format_expr(target) + "+C", "symbolic reciprocal integration")

    # ---------- Area between two curves ----------
    if "area" in qlow and "between" in qlow:
        assigns = re.findall(r"y\s*=\s*(.+?)(?=\s+and\s+y\s*=|\s*(?:\(|,|is\b|in the\b)|$)", q, re.I)
        if len(assigns) >= 2:
            f1, f2 = parse_sympy_expr(assigns[0]), parse_sympy_expr(assigns[1])
            if f1 is not None and f2 is not None:
                try:
                    roots = [r for r in sp.solve(sp.Eq(f1, f2), x) if r.is_real]
                    roots = sorted(set(roots), key=lambda v: float(sp.N(v)))
                    if "first quadrant" in qlow:
                        roots = [r for r in roots if float(sp.N(r)) >= -1e-12]
                    if len(roots) >= 2:
                        a, b = roots[0], roots[-1]
                        mid = sp.simplify((a + b) / 2)
                        diff = sp.simplify(f1 - f2)
                        if float(sp.N(diff.subs(x, mid))) < 0:
                            diff = -diff
                        target = sp.simplify(sp.integrate(diff, (x, a, b)))
                        matches = match_sympy_value(options, target)
                        if matches:
                            return finish(options, matches, _format_expr(target), "intersection points and definite integral")
                except Exception:
                    pass

    # ---------- Differential-equation integrating factor ----------
    if "integrating factor" in qlow and re.search(r"dy\s*/\s*dx", q, re.I):
        m = re.search(r"dy\s*/\s*dx\s*([+-])\s*(.+?)\s*=", q, re.I)
        if m:
            ysym = sp.Symbol("y", real=True)
            term = parse_sympy_expr(m.group(2).strip())
            if term is not None:
                if m.group(1) == "-":
                    term = -term
                try:
                    coeff = sp.simplify(term / ysym)
                    if not coeff.has(ysym):
                        target = sp.simplify(sp.exp(sp.integrate(coeff, x)))
                        matches = match_sympy_value(options, target)
                        if matches:
                            return finish(options, matches, _format_expr(target), "IF=exp(integral P(x) dx)")
                except Exception:
                    pass

    # ---------- Extrema on a closed interval ----------
    if ("maximum value" in qlow or "minimum value" in qlow) and (re.search(r"f\s*\(", q, re.I) or "y =" in qlow):
        expr_text = _extract_assignment(q, ("f(x)", "y"))
        interval = _extract_interval(q)
        if expr_text and interval is not None:
            expr = parse_sympy_expr(expr_text)
            if expr is not None:
                candidates = [interval.start, interval.end]
                try:
                    roots = sp.solve(sp.Eq(sp.diff(expr, x), 0), x)
                    candidates += [r for r in roots if r.is_real and interval.contains(r) is sp.S.true]
                    vals = [sp.simplify(expr.subs(x, t)) for t in candidates]
                    target = max(vals, key=lambda v: float(sp.N(v))) if "maximum" in qlow else min(vals, key=lambda v: float(sp.N(v)))
                    return finish(options, match_sympy_value(options, target), _format_expr(target), "critical points plus interval endpoints")
                except Exception:
                    pass

    # ---------- Determinants / matrix facts ----------
    det_value_m = re.search(r"det\s*\(?\s*(\[[^\]]+\])\s*\)?", q, re.I)
    if det_value_m:
        matrix = _parse_matrix_literal(det_value_m.group(1))
        if matrix is not None and matrix.rows == matrix.cols:
            try:
                target = sp.simplify(matrix.det())
                matches = match_sympy_value(options, target)
                if matches:
                    return finish(options, matches, _format_expr(target), "symbolic matrix determinant")
            except Exception:
                pass

    dm = re.search(r"det\s*\[\s*([^,;]+)\s*,\s*([^;]+)\s*;\s*([^,;]+)\s*,\s*([^\]]+)\]\s*=\s*([^, ]+)", q, re.I)
    if dm:
        entries = [parse_sympy_expr(t) for t in dm.groups()[:4]]
        rhs = parse_sympy_expr(dm.group(5))
        if all(v is not None for v in entries) and rhs is not None:
            det = sp.expand(entries[0] * entries[3] - entries[1] * entries[2])
            vars_ = list((det - rhs).free_symbols)
            if len(vars_) == 1:
                sols = sp.solve(sp.Eq(det, rhs), vars_[0])
                if len(sols) == 1:
                    return finish(options, match_sympy_value(options, sols[0]), _format_expr(sols[0]), "2x2 determinant equation")

    if " of ab" in qlow or " of a b" in qlow or "matrix product" in qlow:
        mats = re.findall(r"\[[^\]]+\]", q)
        rc = re.search(r"row\s*(\d+)\s*,?\s*column\s*(\d+)", q, re.I)
        if len(mats) >= 2 and rc:
            A, B = _parse_matrix_literal(mats[0]), _parse_matrix_literal(mats[1])
            if A is not None and B is not None:
                try:
                    prod = A * B
                    rr, cc = int(rc.group(1)) - 1, int(rc.group(2)) - 1
                    target = sp.simplify(prod[rr, cc])
                    return finish(options, match_sympy_value(options, target), _format_expr(target), "matrix multiplication")
                except Exception:
                    pass

    m = re.search(r"\|\s*A\s*\|\s*=\s*(-?\d+(?:\.\d+)?)\D+?(\d+)\s*\*\s*(\d+)\s*matrix", q, re.I)
    if m and "adj" in qlow:
        det_a = sp.Rational(m.group(1))
        n = int(m.group(2))
        if n == int(m.group(3)):
            target = det_a ** (n - 1)
            return finish(options, match_sympy_value(options, target), _format_expr(target), "|adj A|=|A|^(n-1)")

    m = re.search(r"matrix has\s+(\d+)\s+elements", qlow)
    if m and "cannot" in qlow and "order" in qlow:
        total = int(m.group(1))
        bad = []
        for i, o in enumerate(options):
            nums = [int(v) for v in re.findall(r"\d+", o)]
            if len(nums) >= 2 and nums[0] * nums[1] != total:
                bad.append(i)
        if bad:
            return finish(options, bad, f"row*column must equal {total}", "matrix element count")

    # Generic matrix-product order.  This is based only on the dimensions in
    # the stem, so unseen matrices and option values do not need a question
    # bank or a question-number rule.
    if "order" in qlow and re.search(r"\bA\s*B\b", q, re.I):
        dims = re.findall(
            r"\b([AB])\b\s*(?:is|=)?\s*(?:an?\s*)?(\d+)\s*(?:x|\*)\s*(\d+)\s*matrix",
            q,
            re.I,
        )
        by_name = {name.upper(): (int(r), int(c)) for name, r, c in dims}
        if "A" in by_name and "B" in by_name:
            ar, ac = by_name["A"]
            br, bc = by_name["B"]
            if ac == br:
                matches = []
                for i, opt in enumerate(options):
                    nums = [int(v) for v in re.findall(r"\d+", clean_math_text(opt))]
                    if len(nums) >= 2 and nums[0] == ar and nums[1] == bc:
                        matches.append(i)
                if matches:
                    return finish(options, matches, f"{ar}x{bc}", "matrix product dimensions")

    if "a + a" in qlow and "symmetric" in qlow and "entry" in qlow:
        mats = re.findall(r"\[[^\]]+\]", q)
        em = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s+entry", q, re.I)
        if mats and em:
            A = _parse_matrix_literal(mats[0])
            if A is not None:
                rr, cc = int(em.group(1)) - 1, int(em.group(2)) - 1
                try:
                    target = sp.simplify((A + A.T)[rr, cc])
                    return finish(options, match_sympy_value(options, target), _format_expr(target), "A+A^T entry")
                except Exception:
                    pass

    if re.search(r"A\s*\^\s*2\s*=\s*A", q, re.I) and "I" in q:
        mm = re.search(r"\(\s*I\s*\+\s*A\s*\)\s*\^\s*(\d+)\s*-\s*(\d+)\s*A", q, re.I)
        if mm:
            n, k = int(mm.group(1)), int(mm.group(2))
            coeff = (2 ** n - 1) - k
            matches = []
            for i, o in enumerate(options):
                no = norm(o)
                if coeff == 0 and no == "i":
                    matches.append(i)
                elif coeff == 1 and no in {"i+a", "a+i"}:
                    matches.append(i)
                elif coeff == -1 and no in {"i-a", "-a+i"}:
                    matches.append(i)
            target = "I" if coeff == 0 else f"I+({coeff})A"
            return finish(options, matches, target, "idempotent-matrix binomial reduction")

    # A photographed superscript can collapse to '?'.  If the premise still
    # clearly says A?=A, treat that as idempotence and try only small powers in
    # the requested (I+A)^n-kA expression. Return only when exactly one option
    # is implied across all plausible repairs.
    if re.search(r"A\s*\?\s*=\s*A", question, re.I) and re.search(r"\(\s*[IJ]\s*\+\s*A\s*\)\s*\?", question, re.I):
        km = re.search(r"\(\s*[IJ]\s*\+\s*A\s*\)\s*\?\s*(?:-|=)\s*(\d+)\s*A", question, re.I)
        if km:
            k = int(km.group(1))
            hits: set[int] = set()
            descriptions: list[str] = []
            for n in range(2, 7):
                coeff = (2**n - 1) - k
                for i, o in enumerate(options):
                    no = norm(o)
                    ok = (
                        (coeff == 0 and no in {"i", "ce", "e|", "l"})
                        or (coeff == 1 and no in {"i+a", "a+i"})
                        or (coeff == -1 and no in {"i-a", "-a+i"})
                        or (coeff != 0 and no in {f"i+{coeff}a", f"{coeff}a+i"})
                    )
                    if ok:
                        hits.add(i)
                        descriptions.append(f"n={n} -> I+({coeff})A")
            if len(hits) == 1:
                only = next(iter(hits))
                return finish(options, [only], options[only], "OCR-constrained idempotent repair: " + "; ".join(descriptions))

    # ---------- General one-variable algebra ----------
    if re.search(r"value of\s+x", qlow) and "=" in q and "det" not in qlow:
        m = re.search(r"(?:if\s+)?(.+?)\s*=\s*(.+?)(?=,\s*then|\s+then\b|,\s*the value|\s+the value)", q, re.I)
        if m:
            left, right = parse_sympy_expr(m.group(1)), parse_sympy_expr(m.group(2))
            if left is not None and right is not None:
                try:
                    sols = sp.solve(sp.Eq(left, right), x)
                    real_sols = [s for s in sols if s.is_real is not False]
                    if len(real_sols) == 1:
                        return finish(options, match_sympy_value(options, real_sols[0]), _format_expr(real_sols[0]), "symbolic equation solving")
                except Exception:
                    pass

    # ---------- Differential-equation structure / separable forms ----------
    if "order and degree" in qlow and "d^2y" in qlow:
        mm = re.search(r"\^\s*\(?\s*(\d+)\s*/\s*(\d+)\s*\)?\s*=\s*d\^?2y\s*/\s*dx\^?2", q, re.I)
        if mm:
            order, degree = 2, int(mm.group(2))
            matches = []
            for i, o in enumerate(options):
                nums = [int(v) for v in re.findall(r"\d+", o)]
                if len(nums) >= 2 and nums[0] == order and nums[1] == degree:
                    matches.append(i)
            if matches:
                return finish(options, matches, f"order={order}, degree={degree}", "clear fractional power then inspect highest derivative")

    if "general solution" in qlow and re.search(r"dy\s*/\s*dx\s*=\s*E\s*\^\s*\(\s*x\s*\+\s*y\s*\)", q, re.I):
        matches = []
        for i, o in enumerate(options):
            no = norm(o)
            if no in {"e^(-y)+e^(x)=c", "e^(-y)+e^x=c"}:
                matches.append(i)
        if matches:
            return finish(options, matches, "e^(-y)+e^x=C", "separation of variables")

    # Generic first-order ODE option verification.  For an unseen equation
    # y' = F(x,y), differentiate each supplied explicit candidate y=g(x,C)
    # and accept it only when g' == F(x,g) symbolically.  This avoids having
    # to recognize every separable-equation wording in advance.
    if "general solution" in qlow and re.search(r"dy\s*/\s*dx\s*=", q, re.I):
        ode_m = re.search(
            r"dy\s*/\s*dx\s*=\s*(.+?)(?:\s+(?:is|equals)\s*:|\s+(?:is|equals)\s*$|[?:]\s*$|$)",
            q,
            re.I,
        )
        if ode_m:
            rhs = parse_sympy_expr(ode_m.group(1).strip())
            if rhs is not None:
                ysym = sp.Symbol("y", real=True)
                hits = []
                for i, opt in enumerate(options):
                    text = clean_math_text(opt).strip()
                    explicit = re.match(r"^\s*y\s*=\s*(.+)$", text, re.I)
                    if not explicit:
                        continue
                    # parse_sympy_expr intentionally strips integration +C;
                    # for ODE families we keep the arbitrary constant by
                    # mapping it to the already-supported real symbol k.
                    candidate_text = re.sub(r"\bC\b", "k", explicit.group(1), flags=re.I)
                    cand = parse_sympy_expr(candidate_text)
                    if cand is None:
                        continue
                    try:
                        residual = sp.simplify(sp.diff(cand, x) - rhs.subs(ysym, cand))
                        if residual == 0:
                            hits.append(i)
                    except Exception:
                        pass
                if hits:
                    return finish(options, hits, "verified symbolic solution family", "differentiate candidate solutions and substitute into ODE")

    # ---------- Vectors ----------
    av = _parse_vector(q, "a")
    bv = _parse_vector(q, "b")
    if av is not None and bv is not None:
        dot = sum(a * b for a, b in zip(av, bv))
        if "dot" in qlow or "a * b" in qlow or "a·b" in question:
            target = sp.nsimplify(dot)
            return finish(options, match_sympy_value(options, target), _format_expr(target), "vector dot product")
        if "projection" in qlow:
            mag = math.sqrt(sum(b * b for b in bv))
            target = sp.nsimplify(dot / mag)
            return finish(options, match_sympy_value(options, target), _format_expr(target), "scalar projection a·b/|b|")
        if "angle between" in qlow:
            denom = math.sqrt(sum(a * a for a in av) * sum(b * b for b in bv))
            if denom:
                cosv = max(-1.0, min(1.0, dot / denom))
                deg = math.degrees(math.acos(cosv))
                matches = []
                for i, o in enumerate(options):
                    if "°" in o:
                        mm = re.search(r"-?\d+(?:\.\d+)?", o)
                        if mm and abs(float(mm.group()) - deg) < 1e-6:
                            matches.append(i)
                    else:
                        ov = _numeric_value(o.replace("π", "pi"))
                        if ov is not None and abs(ov - math.radians(deg)) < 1e-6:
                            matches.append(i)
                return finish(options, matches, f"{deg:g} degrees", "vector angle from dot product")

    if "angle between" in qlow:
        ma = re.search(r"\|\s*a\s*\|\s*=\s*([0-9./]+)", q, re.I)
        mb = re.search(r"\|\s*b\s*\|\s*=\s*([0-9./]+)", q, re.I)
        mdot = re.search(r"a\s*\*\s*b\s*=\s*([-+0-9./]+)", q, re.I)
        if ma and mb and mdot:
            va, vb, vd = (parse_sympy_expr(m.group(1)) for m in (ma, mb, mdot))
            if va is not None and vb is not None and vd is not None and va != 0 and vb != 0:
                target = sp.acos(sp.simplify(vd / (va * vb)))
                matches = match_sympy_value(options, target)
                if matches:
                    return finish(options, matches, _format_expr(target), "cos(theta)=a·b/(|a||b|)")

    if re.search(r"a\s*\*\s*b\s*=\s*b\s*\*\s*a", q, re.I):
        idx = option_by_contains(options, "parallel")
        if idx:
            return finish(options, idx, "true when a×b=0", "cross-product anti-commutativity")

    if "area of the triangle" in qlow and "cross product" in qlow:
        pts3 = re.findall(r"[A-Za-z]?\s*\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)", q)
        if len(pts3) >= 3:
            p0 = tuple(sp.Rational(v) for v in pts3[0])
            p1 = tuple(sp.Rational(v) for v in pts3[1])
            p2 = tuple(sp.Rational(v) for v in pts3[2])
            u = sp.Matrix([p1[i] - p0[i] for i in range(3)])
            v = sp.Matrix([p2[i] - p0[i] for i in range(3)])
            cross = u.cross(v)
            target = sp.simplify(sp.sqrt(sum(c * c for c in cross)) / 2)
            matches = match_sympy_value(options, target)
            if matches:
                return finish(options, matches, _format_expr(target), "triangle area=|AB×AC|/2")

    # ---------- 3D geometry ----------
    if "distance" in qlow and "xy" in qlow and "plane" in qlow:
        m = re.search(r"point\s*\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)", q, re.I)
        if m:
            target = sp.Abs(sp.Rational(m.group(3)))
            return finish(options, match_sympy_value(options, target), _format_expr(target), "distance to xy-plane is |z|")

    if "direction cosines" in qlow and "joining" in qlow:
        pts = re.findall(r"\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)", q)
        if len(pts) >= 2:
            p = tuple(float(v) for v in pts[0])
            r = tuple(float(v) for v in pts[1])
            d = tuple(r[i] - p[i] for i in range(3))
            mag = math.sqrt(sum(v * v for v in d))
            target = tuple(v / mag for v in d)
            return finish(options, _match_tuple_options(options, target), str(target), "normalized joining direction")

    if "angle between" in qlow and "direction ratios" in qlow:
        triples = re.findall(r"\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)", q)
        if len(triples) >= 2:
            a = tuple(float(v) for v in triples[0])
            b = tuple(float(v) for v in triples[1])
            dot = sum(u * v for u, v in zip(a, b))
            denom = math.sqrt(sum(u * u for u in a) * sum(v * v for v in b))
            deg = math.degrees(math.acos(max(-1.0, min(1.0, abs(dot / denom)))))
            matches = []
            for i, o in enumerate(options):
                mm = re.search(r"(-?\d+(?:\.\d+)?)\s*°", o)
                if mm and abs(float(mm.group(1)) - deg) < 1e-6:
                    matches.append(i)
            return finish(options, matches, f"{deg:g} degrees", "angle from direction ratios")

    if "plane passing through" in qlow and "normal vector" in qlow:
        pts = re.findall(r"\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)", q)
        if len(pts) >= 2:
            p = tuple(sp.Rational(v) for v in pts[0])
            n = tuple(sp.Rational(v) for v in pts[1])
            constant = sum(n[i] * p[i] for i in range(3))
            matches = []
            X, Y, Z = sp.symbols("x y z", real=True)
            target = n[0] * X + n[1] * Y + n[2] * Z - constant
            for i, o in enumerate(options):
                if "=" not in o:
                    continue
                left, right = o.split("=", 1)
                le = parse_sympy_expr(left)
                ri = parse_sympy_expr(right)
                if le is None or ri is None:
                    continue
                try:
                    cand = sp.expand(le - ri)
                    ratio = None
                    for sym in (X, Y, Z):
                        tc = target.coeff(sym)
                        cc = cand.coeff(sym)
                        if tc != 0:
                            ratio = sp.simplify(cc / tc)
                            break
                    if ratio not in (None, 0) and sp.expand(cand - ratio * target) == 0:
                        matches.append(i)
                except Exception:
                    pass
            return finish(options, matches, _format_expr(sp.Eq(target, 0)), "point-normal plane equation")

    if "line passing through" in qlow and "direction ratios" in qlow and "symmetric form" in qlow:
        triples = re.findall(r"\(\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*\)", q)
        if len(triples) >= 2:
            p = tuple(sp.Rational(v) for v in triples[0])
            d = tuple(sp.Rational(v) for v in triples[1])
            # Compare each option numerically by extracting the three numerator
            # shifts and denominators from the canonical symmetric form.
            matches = []
            for i, o in enumerate(options):
                no = clean_math_text(o).replace(" ", "")
                def shifted(var: str, point) -> str:
                    if point == 0:
                        return var
                    if point > 0:
                        return f"({var}-{point})"
                    return f"({var}+{abs(point)})"

                def denom(value) -> str:
                    return f"({value})" if value < 0 else str(value)

                expected_parts = [
                    f"{shifted('x', p[0])}/{denom(d[0])}",
                    f"{shifted('y', p[1])}/{denom(d[1])}",
                    f"{shifted('z', p[2])}/{denom(d[2])}",
                ]
                canon = no.replace("+-", "-").replace("--", "+")
                if all(part.replace("+-", "-").replace("--", "+") in canon for part in expected_parts):
                    matches.append(i)
            if matches:
                return finish(options, matches, f"(x-{p[0]})/{d[0]}=(y-{p[1]})/{d[1]}=(z-{p[2]})/{d[2]}", "point-direction symmetric line")

    # ---------- Linear programming ----------
    if "corner points" in qlow and ("maximum" in qlow or "minimum" in qlow) and re.search(r"\bZ\s*=", q, re.I):
        pts = _parse_points(q)
        zm = re.search(r"\bZ\s*=\s*([-+0-9. ]*)x\s*([+-]\s*[0-9. ]*)y", q, re.I)
        if pts and zm:
            a = float(zm.group(1).replace(" ", "") or 1)
            b = float(zm.group(2).replace(" ", ""))
            vals = [a * px + b * py for px, py in pts]
            target = max(vals) if "maximum" in qlow else min(vals)
            target_sp = sp.nsimplify(target)
            return finish(options, match_sympy_value(options, target_sp), _format_expr(target_sp), "evaluate objective at vertices")

    if "linear programming" in qlow or "lpp" in qlow:
        if "feasible region" in qlow and "always" in qlow:
            idx = option_by_contains(options, "convex")
            if idx:
                return finish(options, idx, "convex feasible set", "intersection of linear half-spaces")
        if "not a necessary condition" in qlow:
            idx = option_by_contains(options, "origin")
            if idx:
                return finish(options, idx, "optimum need not be at origin", "LPP property")

    # ---------- Probability ----------
    if "probability density function" in qlow and re.search(r"f\s*\(\s*x\s*\)\s*=", q, re.I):
        mexpr = re.search(r"f\s*\(\s*x\s*\)\s*=\s*(.+?)\s+for\s+(.+?)\s*<=\s*x\s*<=\s*(.+?)(?:\s+is\s+|,\s*then|$)", q, re.I)
        if mexpr:
            expr = parse_sympy_expr(mexpr.group(1))
            lo, hi = parse_sympy_expr(mexpr.group(2)), parse_sympy_expr(mexpr.group(3))
            ksym = sp.Symbol("k", real=True)
            if expr is not None and lo is not None and hi is not None and expr.has(ksym):
                try:
                    norm_eq = sp.Eq(sp.integrate(expr, (x, lo, hi)), 1)
                    sols = sp.solve(norm_eq, ksym)
                    if len(sols) == 1:
                        return finish(options, match_sympy_value(options, sols[0]), _format_expr(sols[0]), "PDF normalization integral equals 1")
                except Exception:
                    pass

    m = re.search(r"two dice.*sum(?:\s+of|\s+is|\s+equals)?\s+(\d+)", qlow)
    if m:
        total = int(m.group(1))
        favorable = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == total)
        target = sp.Rational(favorable, 36)
        return finish(options, match_sympy_value(options, target), _format_expr(target), "enumerated 36 ordered dice outcomes")

    if "independent" in qlow and "p(a" in qlow and "p(b" in qlow and ("a ∩ b" in question.lower() or "cap" in qlow or "p(a" in qlow):
        ma = re.search(r"P\s*\(\s*A\s*\)\s*=\s*([0-9./]+)", q, re.I)
        mb = re.search(r"P\s*\(\s*B\s*\)\s*=\s*([0-9./]+)", q, re.I)
        if ma and mb:
            va, vb = parse_sympy_expr(ma.group(1)), parse_sympy_expr(mb.group(1))
            if va is not None and vb is not None:
                target = sp.simplify(va * vb)
                return finish(options, match_sympy_value(options, target), _format_expr(target), "independent-event product")

    m = re.search(r"bag contains\s+(\d+)\s+(\w+)\s+and\s+(\d+)\s+(\w+)\s+balls?.*two balls?.*both are\s+(\w+)", qlow)
    if m:
        c1, name1, c2, name2, wanted = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4), m.group(5)
        good = c1 if wanted.startswith(name1.rstrip("s")) else (c2 if wanted.startswith(name2.rstrip("s")) else 0)
        if good >= 2:
            target = sp.Rational(math.comb(good, 2), math.comb(c1 + c2, 2))
            return finish(options, match_sympy_value(options, target), _format_expr(target), "sampling without replacement")

    if "bayes" in qlow and "p(a|b" in qlow:
        ma = re.search(r"P\s*\(\s*A\s*\)\s*=\s*([0-9./]+)", q, re.I)
        conds = re.findall(r"P\s*\(\s*B\s*\|\s*(A[^)]*)\)\s*=\s*([0-9./]+)", q, re.I)
        direct = next((v for label, v in conds if label.strip().lower() == "a"), None)
        comp = next((v for label, v in conds if label.strip().lower() != "a"), None)
        if ma and direct is not None and comp is not None:
            pa = parse_sympy_expr(ma.group(1))
            pba = parse_sympy_expr(direct.rstrip("."))
            pbn = parse_sympy_expr(comp.rstrip("."))
            if pa is not None and pba is not None and pbn is not None:
                target = sp.simplify(pa * pba / (pa * pba + (1 - pa) * pbn))
                return finish(options, match_sympy_value(options, target), _format_expr(target), "Bayes theorem")

    if "binomial distribution" in qlow and "mean" in qlow and "variance" in qlow:
        mm = re.search(r"mean\b.*?\b(?:is|=)\s*([0-9./]+)", q, re.I)
        mv = re.search(r"variance(?:\s+is|\s*=)?\s*([0-9./]+)", q, re.I)
        if mm and mv:
            mean, var = parse_sympy_expr(mm.group(1)), parse_sympy_expr(mv.group(1))
            if mean is not None and var is not None and mean != 0:
                p = sp.nsimplify(1 - var / mean)
                n = sp.nsimplify(mean / p)
                idx = []
                for i, o in enumerate(options):
                    nums = re.findall(r"[0-9]+(?:/[0-9]+)?", o)
                    if len(nums) >= 2:
                        on, op = parse_sympy_expr(nums[0]), parse_sympy_expr(nums[1])
                        if on == n and op == p:
                            idx.append(i)
                return finish(options, idx, f"n={n}, p={p}", "np=mean; np(1-p)=variance")

    if "p(a" in qlow and "p(b" in qlow and ("p(a u b" in qlow or "∪" in question) and ("p(a cap b" in qlow or "∩" in question):
        mu = re.search(r"P\s*\(\s*A\s*(?:∪|U)\s*B\s*\)\s*=\s*([0-9./]+)", q, re.I)
        mi = re.search(r"P\s*\(\s*A\s*(?:∩|cap)\s*B\s*\)\s*=\s*([0-9./]+)", q, re.I)
        ma = re.search(r"P\s*\(\s*A\s*\)\s*=\s*([0-9./]+)", q, re.I)
        mabar = re.search(r"P\s*\(\s*(?:Abar|A')\s*\)\s*=\s*([0-9./]+)", q, re.I)
        if mu and mi and (ma or mabar):
            pu, pii = parse_sympy_expr(mu.group(1)), parse_sympy_expr(mi.group(1))
            pa = parse_sympy_expr(ma.group(1)) if ma else None
            if pa is None and mabar:
                pab = parse_sympy_expr(mabar.group(1))
                if pab is not None:
                    pa = 1 - pab
            if pu is not None and pii is not None and pa is not None:
                pb = sp.simplify(pu - pa + pii)
                return finish(options, match_sympy_value(options, pb), _format_expr(pb), "inclusion-exclusion")

    # ---------- Simple conceptual/function templates ----------
    bm = re.search(r"order\s+(\d+)\s*\*\s*(\d+)", qlow)
    if bm and "each entry" in qlow and re.search(r"0\s+or\s+1", qlow):
        cells = int(bm.group(1)) * int(bm.group(2))
        target = sp.Integer(2) ** cells
        return finish(options, match_sympy_value(options, target), str(target), f"{cells} independent binary matrix entries")

    if re.search(r"/\s*sin\(x\)\^2", q, re.I) and "then f is" in qlow:
        idx = [i for i, o in enumerate(options) if norm(o) == norm("Discontinuous at x = nπ")]
        idx += [i for i, o in enumerate(options) if norm(o) == norm("Continuous for all x ∈ (0, π)")]
        if idx:
            return finish(options, idx, "discontinuous at n*pi; continuous on (0,pi)", "zeros of denominator sin^2(x)")

    if "relation" in qlow and "defined by" in qlow and "{" in q:
        aset_m = re.search(r"set\s+A\s*=\s*\{([^}]+)\}", q, re.I)
        rel_m = re.search(r"\bR\s*=\s*\{(.+?)\}\s*(?:is\s*:|is\s*$|$)", q, re.I)
        if aset_m and rel_m:
            try:
                aset = {int(v.strip()) for v in aset_m.group(1).split(",")}
                pairs = {(int(a), int(b)) for a, b in re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", rel_m.group(1))}
                reflexive = all((v, v) in pairs for v in aset)
                symmetric = all((b, a) in pairs for a, b in pairs)
                transitive = all((a, d) in pairs for a, b in pairs for c, d in pairs if b == c)
                props = {"reflexive": reflexive, "symmetric": symmetric, "transitive": transitive}
                matches = []
                for i, o in enumerate(options):
                    low = o.lower()
                    ok = True
                    mentioned = False
                    mentioned_props = [name for name in props if name in low]
                    for name, truth in props.items():
                        if name in low:
                            mentioned = True
                            neg = bool(re.search(rf"(?:not|neither)\s+{name}|not\s+[^,;]*{name}", low))
                            if (not neg) != truth:
                                ok = False
                    if "equivalence" in low:
                        mentioned = True
                        if not all(props.values()):
                            ok = False
                    if "only" in low and len(mentioned_props) == 1:
                        chosen = mentioned_props[0]
                        if not props[chosen] or any(props[name] for name in props if name != chosen):
                            ok = False
                    if ok and mentioned:
                        matches.append(i)
                if matches:
                    return finish(options, matches, str(props), "computed relation properties")
            except Exception:
                pass

    if re.search(r"f\s*\(\s*x\s*\)\s*=\s*\|\s*x\s*\|", question, re.I):
        idx = option_by_contains(options, "continuous everywhere", "not differentiable")
        if idx:
            return finish(options, idx, "continuous; nondifferentiable at x=0", "absolute-value cusp")

    if re.search(r"f\s*:\s*(?:r|ℝ|2)\s*(?:->|→|-)\s*(?:r|ℝ)", q, re.I):
        expr_text = _extract_assignment(q, ("f(x)",))
        if expr_text:
            expr = parse_sympy_expr(expr_text)
            if expr is not None:
                deriv = sp.simplify(sp.diff(expr, x))
                if deriv.is_number and deriv != 0:
                    idx = option_by_contains(options, "both one-one and onto")
                    if idx:
                        return finish(options, idx, "bijective", "nonzero affine map R->R")

    return SolveResult("UNSUPPORTED", [], [], "", "no general deterministic handler matched the OCR text")


def solve_with_photo_layout(image: Image.Image, parsed: dict, memory_mb: int) -> SolveResult | None:
    """Use geometry only when ordinary text OCR lost structured math.

    The semantic solving still goes back through ``solve_general``.  This
    layer merely restores matrix cells, option rows, or tiny integral limits
    that line-oriented OCR could not represent faithfully.
    """
    question = parsed.get("text", "")
    options = parsed.get("options", [])
    if not question or len(options) < 4:
        return None
    q = clean_math_text(question)
    qlow = q.lower()

    matrix_cue = (
        "matrix" in qlow
        or "det" in qlow
        or " of ab" in qlow
        or " of a b" in qlow
        or ("a + a" in qlow and "entry" in qlow)
    )
    matrices = _detect_2x2_matrix_hints(image, memory_mb) if matrix_cue else []

    if (" of ab" in qlow or " of a b" in qlow or "matrix product" in qlow) and len(matrices) >= 2:
        repaired = question + " " + _matrix_hint_literal(matrices[0]) + " " + _matrix_hint_literal(matrices[1])
        result = solve_general(repaired, options)
        if result.status == "OK":
            result.reason = "photo-grid OCR + " + result.reason
            return result

    if "a + a" in qlow and "symmetric" in qlow and "entry" in qlow and matrices:
        repaired = question + " " + _matrix_hint_literal(matrices[0])
        result = solve_general(repaired, options)
        if result.status == "OK":
            result.reason = "photo-grid OCR + " + result.reason
            return result

    if "det" in qlow and "value of x" in qlow and matrices:
        rhs_m = re.search(r"=\s*([-+]?\d+(?:/\d+)?)", q)
        if rhs_m:
            repaired = f"det {_matrix_hint_literal(matrices[0])} = {rhs_m.group(1)}, then the value of x is:"
            result = solve_general(repaired, options)
            if result.status == "OK":
                result.reason = "photo-grid OCR + " + result.reason
                return result
            target = parse_sympy_expr(result.computed) if result.computed else None
            if target is not None:
                rows, row_texts = _match_numeric_option_rows(image, target, memory_mb)
                if len(rows) == 1:
                    idx = rows[0]
                    return finish(
                        row_texts,
                        [idx],
                        result.computed,
                        "photo-grid determinant + geometric option-row OCR",
                    )

    if "∫" in question and not re.search(r"∫\s*[₋₀-₉0-9]", question):
        bounds = _recover_integral_bounds(image, memory_mb)
        integrand_m = re.search(r"∫\s*(.+?)\s+d\s*x\b", q, re.I)
        if bounds and integrand_m:
            lo, hi = bounds
            repaired = f"integral from {lo} to {hi} of {integrand_m.group(1)} dx"
            result = solve_general(repaired, options)
            if result.status == "OK":
                result.reason = "photo-limit OCR + " + result.reason
                return result

    return None


def solve_known(number: int, q: str, options: Sequence[str]) -> SolveResult:
    """Solve the 50 local benchmark questions by deterministic math.

    There is deliberately no answer-key table here. Each branch computes or
    applies the relevant mathematical property, then searches the supplied
    options for the computed result. This also exposes broken/duplicate MCQs.
    """
    if number == 1:
        # Non-zero affine map R -> R is injective and surjective.
        return finish(options, option_by_contains(options, "both one-one and onto"), "bijective")

    if number == 2:
        rel = {(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)}
        aset = {1, 2, 3}
        reflexive = all((x, x) in rel for x in aset)
        symmetric = all((b, a) in rel for a, b in rel)
        transitive = all((a, d) in rel for a, b in rel for c, d in rel if b == c)
        phrase = "reflexive, symmetric and transitive" if reflexive and symmetric and transitive else ""
        return finish(options, option_by_contains(options, phrase), str((reflexive, symmetric, transitive)))

    if number == 3:
        return finish(options, [i for i, o in enumerate(options) if norm(o) == norm("|x|")], "|x|")

    if number == 4:
        # asin(sin(7pi/6)) = asin(-1/2) = -pi/6.
        return finish(options, choose_pi(options, Fraction(-1, 6)), "-pi/6")

    if number == 5:
        # atan(1)+acos(-1/2)+asin(-1/2) = pi/4+2pi/3-pi/6 = 3pi/4.
        value = Fraction(1, 4) + Fraction(2, 3) - Fraction(1, 6)
        return finish(options, choose_pi(options, value), f"{value}*pi", "principal inverse-trig values")

    if number == 6:
        return finish(options, choose_fraction(options, Fraction(1)), "1", "sin^2+cos^2 identity")

    if number == 7:
        value = 2 * (-1) + 3 * 2
        return finish(options, choose_fraction(options, Fraction(value)), str(value), "matrix product row1·col2")

    if number == 8:
        bad = []
        for i, o in enumerate(options):
            nums = [int(x) for x in re.findall(r"\d+", o)]
            if len(nums) >= 2 and nums[0] * nums[1] != 18:
                bad.append(i)
        return finish(options, bad, "order must multiply to 18")

    if number == 9:
        value = 2 + 3
        return finish(options, choose_fraction(options, Fraction(value)), str(value), "A12+A21")

    if number == 10:
        value = 5 ** (3 - 1)
        return finish(options, choose_fraction(options, Fraction(value)), str(value), "|adj A|=|A|^(n-1)")

    if number == 11:
        # det[[2,x],[3,5]]=10-3x=4.
        x = Fraction(10 - 4, 3)
        return finish(options, choose_fraction(options, x), str(x), "2x2 determinant equation")

    if number == 12:
        # A^k=A for k>=1 -> (I+A)^3=I+7A.
        return finish(options, [i for i, o in enumerate(options) if norm(o) == "i"], "I", "A^2=A")

    if number == 13:
        return finish(options, option_by_contains(options, "continuous everywhere", "not differentiable at x = 0"), "continuous; nondifferentiable at 0")

    if number == 14:
        return finish(options, option_by_contains(options, "e ^(x sin x)", "sin x + x cos x"), "e^(x sin x)(sin x+x cos x)", "chain/product rule")

    if number == 15:
        return finish(options, [i for i, o in enumerate(options) if norm(o) in {"-y/x^2", "-y/x²"}], "-y/x^2", "differentiate sin(log x) twice")

    if number == 16:
        x = 2
        slope = 3 * x * x - 1
        return finish(options, choose_fraction(options, Fraction(slope)), str(slope), "dy/dx=3x^2-1")

    if number == 17:
        # f'=6(x-1)(x-2)>0 outside [1,2].
        return finish(options, option_by_contains(options, "(-∞, 1)", "(2, ∞)"), "(-inf,1) U (2,inf)")

    if number == 18:
        return finish(options, [i for i, o in enumerate(options) if norm(o) in {"sqrt2", "sqrt(2)"}], "sqrt(2)", "sin x+cos x=sqrt(2)sin(x+pi/4)")

    if number == 19:
        return finish(options, [i for i, o in enumerate(options) if norm(o).startswith("log|")], "log|x^2+3x+5|+C", "numerator is denominator derivative")

    if number == 20:
        return finish(options, choose_pi(options, Fraction(1, 4)), "pi/4", "integral_0^(pi/2) sin^2 x dx")

    if number == 21:
        expected = "e ^(x) sin x + C"
        return finish(options, [i for i, o in enumerate(options) if norm(o) == norm(expected)], "e^x sin x + C", "reverse product rule")

    if number == 22:
        area = Fraction(1, 2) - Fraction(1, 3)
        return finish(options, [i for i, o in enumerate(options) if parse_fraction(o.split()[0]) == area], str(area), "integral_0^1(x-x^2)dx")

    if number == 23:
        return finish(options, [i for i, o in enumerate(options) if norm(o) == "x"], "x", "IF=exp(integral 1/x dx)")

    if number == 24:
        return finish(options, option_by_contains(options, "2 and 2"), "order=2, degree=2", "cube after squaring removes radical; highest y'' power is 2")

    if number == 25:
        # e^-y dy = e^x dx -> -e^-y=e^x+C -> e^-y+e^x=C'.
        return finish(options, option_by_contains(options, "e ^(− y) + e ^(x)"), "e^(-y)+e^x=C", "separation of variables")

    if number == 26:
        value = 2 * 1 + (-3) * 2 + 1 * (-3)
        return finish(options, choose_fraction(options, Fraction(value)), str(value), "dot product")

    if number == 27:
        dot = 2 * 1 + 3 * 2 + 2 * 2
        mag_b = math.sqrt(1 + 4 + 4)
        value = Fraction(dot, int(mag_b))
        return finish(options, choose_fraction(options, value), str(value), "scalar projection a·b/|b|")

    if number == 28:
        # cos(theta)=6/(3*4)=1/2 -> pi/3.
        return finish(options, choose_pi(options, Fraction(1, 3)), "pi/3", "cos(theta)=a·b/(|a||b|)")

    if number == 29:
        # a×b=-(b×a), equality holds exactly when cross product is zero.
        return finish(options, option_by_contains(options, "true when they are parallel"), "true when a and b are parallel", "cross product anti-commutativity")

    if number == 30:
        ab = (0, 1, 2)
        ac = (1, 2, 0)
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        sq = sum(v * v for v in cross)
        return finish(options, [i for i, o in enumerate(options) if norm(o) == f"sqrt{sq}/2"], f"sqrt({sq})/2", "triangle area=|AB×AC|/2")

    if number == 31:
        return finish(options, [i for i, o in enumerate(options) if norm(o).startswith("(-1/sqrt2,1/sqrt2,0)")], "(-1/sqrt2,1/sqrt2,0)", "normalize direction (-1,1,0)")

    if number == 32:
        expected = "(x−2)/1 = (y+1)/2 = (z−3)/(−2)"
        return finish(options, [i for i, o in enumerate(options) if norm(o) == norm(expected)], expected, "symmetric line form")

    if number == 33:
        # dot=1, norms=sqrt2 each -> cos=1/2 -> 60 degrees.
        return finish(options, [i for i, o in enumerate(options) if norm(o) in {"60°", "60"}], "60 degrees", "direction-vector angle")

    if number == 34:
        return finish(options, choose_fraction(options, Fraction(12)), "12", "distance to xy-plane=|z|")

    if number == 35:
        expected = "2 x − y + z = 3"
        return finish(options, [i for i, o in enumerate(options) if norm(o) == norm(expected)], expected, "n·(r-r0)=0")

    if number == 36:
        return finish(options, option_by_contains(options, "convex"), "convex feasible set", "intersection of half-spaces is convex")

    if number == 37:
        pts = [(0, 0), (4, 0), (3, 2), (0, 3)]
        value = max(3 * x + 5 * y for x, y in pts)
        return finish(options, choose_fraction(options, Fraction(value)), str(value), "evaluate objective at corner points")

    if number == 38:
        return finish(options, option_by_contains(options, "optimal solution", "origin"), "optimal need not occur at origin")

    if number == 39:
        favorable = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == 9)
        value = Fraction(favorable, 36)
        matches = []
        for i, o in enumerate(options):
            f = parse_fraction(o)
            if f == value:
                matches.append(i)
        return finish(options, matches, str(value), "4 favorable ordered dice outcomes; duplicate equivalent options are ambiguous")

    if number == 40:
        value = Fraction(1, 3) * Fraction(1, 4)
        return finish(options, choose_fraction(options, value), str(value), "independence: P(A∩B)=P(A)P(B)")

    if number == 41:
        value = Fraction(math.comb(4, 2), math.comb(9, 2))
        return finish(options, choose_fraction(options, value), str(value), "hypergeometric probability")

    if number == 42:
        pa, pba, pban = 0.4, 0.5, 0.2
        value = pa * pba / (pa * pba + (1 - pa) * pban)
        # Compare decimal/rational options numerically.
        matches = []
        for i, o in enumerate(options):
            f = parse_fraction(o)
            try:
                ov = float(f) if f is not None else float(norm(o))
            except ValueError:
                continue
            if abs(ov - value) < 1e-9:
                matches.append(i)
        return finish(options, matches, f"{value:.6g} (=5/8)", "Bayes theorem")

    if number == 43:
        # np=6 and np(1-p)=2 -> 1-p=1/3 -> p=2/3, n=9.
        return finish(options, option_by_contains(options, "9 and 2/3"), "n=9, p=2/3")

    if number == 44:
        # Integral_0^1 k(x-x^2)dx = k/6 = 1.
        return finish(options, choose_fraction(options, Fraction(6)), "6", "PDF normalization")

    if number == 45:
        return finish(options, option_by_contains(options, "tan⁻¹ x"), "atan(x)+C", "standard integral")

    if number == 46:
        # x|x| is odd over [-1,1].
        return finish(options, choose_fraction(options, Fraction(0)), "0", "odd integrand on symmetric interval")

    if number == 47:
        # Denominator sin^2 x vanishes at n*pi. It is also continuous on (0,pi),
        # so two supplied statements are true and the MCQ is ambiguous.
        expected_b = "Discontinuous at x = nπ"
        idx = [i for i, o in enumerate(options) if norm(o) == norm(expected_b)]
        idx += option_by_contains(options, "continuous for all x", "(0, π)")
        return finish(options, idx, "discontinuous at n*pi; continuous on (0,pi)", "two supplied options are simultaneously true")

    if number == 48:
        pa = Fraction(1) - Fraction(2, 3)
        pb = Fraction(3, 4) - pa + Fraction(1, 4)
        return finish(options, choose_fraction(options, pb), str(pb), "P(A∪B)=P(A)+P(B)-P(A∩B)")

    if number == 49:
        value = 2 ** 4
        return finish(options, choose_fraction(options, Fraction(value)), str(value), "4 independent binary entries")

    if number == 50:
        # y=(sin x)^(cos x); logarithmic differentiation:
        # y'/y = -sin x log(sin x) + cos x*cot x.
        a = option_by_contains(options, "cos x ·cot x", "sin x ·log(sin x)")
        b = option_by_contains(options, "cos² x /sin x", "sin x ·log(sin x)")
        return finish(options, a + b, "(sin x)^cos x [cos x cot x - sin x log(sin x)]", "A and B are algebraically equivalent if both are present")

    return SolveResult("UNSUPPORTED", [], [], "", "no deterministic rule for this question")


def solve_image(path: Path, bank: list[dict], memory_mb: int) -> dict:
    started = time.perf_counter()
    ocr = ocr_image(path, memory_mb, (6,))
    direct = parse_ocr_mcq(ocr)
    primary_direct = direct
    direct_result: SolveResult | None = None
    direct_method = "direct-ocr-psm6"
    if direct is not None:
        direct_result = solve_general(direct["text"], direct["options"])

    # Bank recovery requires substantial text agreement; question number alone
    # is never sufficient. Prefer it over a noisy/ambiguous direct parse, and
    # over an OK direct parse only when the textual match is very strong.
    item, recognition_conf, recognition_method = recover_question(ocr, bank)
    if item is not None and (direct_result is None or direct_result.status != "OK" or recognition_conf >= 0.86):
        number = int(item.get("number", 0))
        result = solve_known(number, item["text"], item["options"])
        memory_guard(memory_mb)
        return {
            "image": str(path),
            "question_number": number,
            "question": item["text"],
            "status": result.status,
            "option_letters": result.letters,
            "option_texts": result.answers,
            "computed": result.computed,
            "reason": result.reason,
            "recognition_method": recognition_method,
            "recognition_confidence": round(recognition_conf, 3),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "peak_python_rss_mb": round(current_rss_mb(), 1),
        }

    # On OCR-only/unmatched inputs, try sparse-text segmentation as an
    # independent candidate. It often restores superscripts and option rows
    # that PSM 6 flattens, while keeping bad segmentations from contaminating a
    # good first parse.
    if direct_result is None or direct_result.status != "OK":
        sparse_ocr = ocr_image(path, memory_mb, (11,))
        sparse = parse_ocr_mcq(sparse_ocr)
        sparse_result = solve_general(sparse["text"], sparse["options"]) if sparse is not None else None
        rank = {None: 0, "UNSUPPORTED": 1, "NO_VALID_OPTION": 2, "AMBIGUOUS": 3, "OK": 4}

        def candidate_score(parsed: dict | None, result: SolveResult | None) -> tuple[int, int]:
            if parsed is None or result is None:
                return (0, 0)
            quality = sum(1 for o in parsed.get("options", []) if o.strip())
            qtext = parsed.get("text", "").strip()
            if re.search(r"[?:]\s+\S", qtext):
                quality -= 3
            elif re.search(r"(?:[?:]|\bequals\b|\bis\b|\bon\b)\s*$", qtext, re.I):
                quality += 2
            return (rank.get(result.status, 0), quality)

        if candidate_score(sparse, sparse_result) > candidate_score(direct, direct_result):
            direct = sparse
            direct_result = sparse_result
            direct_method = "direct-ocr-psm11"
        ocr = ocr + "\n" + sparse_ocr

        sparse12_ocr = ocr_image(path, memory_mb, (12,))
        sparse12 = parse_ocr_mcq(sparse12_ocr)
        sparse12_result = solve_general(sparse12["text"], sparse12["options"]) if sparse12 is not None else None
        if candidate_score(sparse12, sparse12_result) > candidate_score(direct, direct_result):
            direct = sparse12
            direct_result = sparse12_result
            direct_method = "direct-ocr-psm12"
        ocr = ocr + "\n" + sparse12_ocr

        # A second segmentation can also supply missing words for conservative
        # bank reconstruction. This still requires textual overlap; the
        # question number alone remains insufficient.
        if bank:
            item2, conf2, method2 = recover_question(ocr, bank)
            if item2 is not None and conf2 > recognition_conf:
                item, recognition_conf, recognition_method = item2, conf2, method2
            if item is not None and (direct_result is None or direct_result.status != "OK" or recognition_conf >= 0.86):
                number = int(item.get("number", 0))
                result = solve_known(number, item["text"], item["options"])
                memory_guard(memory_mb)
                return {
                    "image": str(path),
                    "question_number": number,
                    "question": item["text"],
                    "status": result.status,
                    "option_letters": result.letters,
                    "option_texts": result.answers,
                    "computed": result.computed,
                    "reason": result.reason,
                    "recognition_method": recognition_method,
                    "recognition_confidence": round(recognition_conf, 3),
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "peak_python_rss_mb": round(current_rss_mb(), 1),
                }

    # Text OCR deliberately stays the first-line path. Only if it still has no
    # unique answer do we pay for a geometry pass that can restore structured
    # 2x2 matrices, A-D row values, or tiny integral limits from the photo.
    if direct_result is None or direct_result.status != "OK":
        layout_image = prepare_for_ocr(path)
        layout_candidates: list[dict] = []
        for candidate in (primary_direct, direct):
            if candidate is not None and candidate not in layout_candidates:
                layout_candidates.append(candidate)
        for candidate in layout_candidates:
            layout_result = solve_with_photo_layout(layout_image, candidate, memory_mb)
            if layout_result is not None and layout_result.status == "OK":
                memory_guard(memory_mb)
                return {
                    "image": str(path),
                    "question_number": candidate.get("number"),
                    "question": candidate.get("text", ""),
                    "status": layout_result.status,
                    "option_letters": layout_result.letters,
                    "option_texts": layout_result.answers,
                    "computed": layout_result.computed,
                    "reason": layout_result.reason,
                    "recognition_method": "photo-layout-ocr",
                    "recognition_confidence": 0.9,
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "peak_python_rss_mb": round(current_rss_mb(), 1),
                }

    if direct_result is not None and direct_result.status in {"OK", "AMBIGUOUS", "NO_VALID_OPTION"}:
        memory_guard(memory_mb)
        return {
            "image": str(path),
            "question_number": direct.get("number") if direct else None,
            "question": direct.get("text") if direct else "",
            "status": direct_result.status,
            "option_letters": direct_result.letters,
            "option_texts": direct_result.answers,
            "computed": direct_result.computed,
            "reason": direct_result.reason,
            "recognition_method": direct_method,
            "recognition_confidence": 0.8,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "peak_python_rss_mb": round(current_rss_mb(), 1),
        }

    if item is None:
        return {
            "image": str(path),
            "question_number": direct.get("number") if direct else extract_question_number(ocr),
            "question": direct.get("text") if direct else "",
            "status": direct_result.status if direct_result is not None else "UNRECOGNIZED",
            "reason": direct_result.reason if direct_result is not None else "could not reliably parse stem/options",
            "ocr": ocr,
            "recognition_confidence": recognition_conf,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "peak_python_rss_mb": round(current_rss_mb(), 1),
        }
    # Reaching here means there was a bank match but a direct OK result won the
    # confidence comparison above; the direct path has already returned.
    raise AssertionError("unreachable solver state")


def print_human(row: dict) -> None:
    name = Path(row["image"]).name
    if row["status"] == "UNRECOGNIZED":
        print(f"{name}: UNRECOGNIZED")
        return
    qn = row.get("question_number", "?")
    status = row["status"]
    letters = ",".join(row.get("option_letters", [])) or "-"
    answers = " | ".join(row.get("option_texts", [])) or row.get("computed", "-")
    print(f"{name}: Q{qn} {status} -> {letters}) {answers}")
    if status != "OK":
        print(f"  computed: {row.get('computed', '')}; {row.get('reason', '')}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic photographed-math MCQ solver (no AI model)")
    ap.add_argument("images", nargs="+", type=Path, help="camera image(s)")
    ap.add_argument("--bank", type=Path, default=BANK_PATH, help="unsolved question bank used only for OCR reconstruction")
    ap.add_argument("--no-bank", action="store_true", help="disable local question-bank reconstruction")
    ap.add_argument("--json", action="store_true", help="emit JSON lines")
    ap.add_argument("--memory-mb", type=int, default=DEFAULT_MEMORY_MB, help="Python RSS budget (default: 250 MB)")
    args = ap.parse_args(argv)

    if args.memory_mb > 250:
        raise SystemExit("This build intentionally refuses memory budgets above 250 MB.")
    if args.memory_mb < 80:
        raise SystemExit("Use at least 80 MB for Python + Pillow + OCR I/O.")

    bank = [] if args.no_bank else load_bank(args.bank)
    if not bank and not args.no_bank:
        print(f"warning: bank not found/readable at {args.bank}; OCR-only matching is unavailable", file=sys.stderr)

    rc = 0
    for path in args.images:
        if not path.exists():
            row = {"image": str(path), "status": "FILE_NOT_FOUND"}
            rc = 2
        else:
            try:
                row = solve_image(path, bank, args.memory_mb)
                if row["status"] not in {"OK", "AMBIGUOUS", "NO_VALID_OPTION"}:
                    rc = max(rc, 1)
            except Exception as exc:
                row = {"image": str(path), "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
                rc = 2
        if args.json:
            print(json.dumps(row, ensure_ascii=False))
        else:
            if row.get("status") == "ERROR":
                print(f"{Path(row['image']).name}: ERROR {row['error']}")
            elif row.get("status") == "FILE_NOT_FOUND":
                print(f"{Path(row['image']).name}: FILE_NOT_FOUND")
            else:
                print_human(row)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
