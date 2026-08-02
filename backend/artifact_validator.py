# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mehmet Raşit Narçiçek

import json
import csv
# Use defusedxml for safe XML parsing; fall back to stdlib with a warning.
try:
    import defusedxml.ElementTree as ET  # type: ignore[import]
except ImportError:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(
        "defusedxml not installed. Falling back to stdlib xml.etree.ElementTree. "
        "Install defusedxml>=0.7.1 for XML attack protection.",
        ImportWarning, stacklevel=2,
    )
    import xml.etree.ElementTree as ET  # type: ignore[assignment]
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from PIL import Image

# Upper bound on decoded image size, guarding against decompression bombs.
MAX_IMAGE_PIXELS = 64_000_000


def parse_ascii_file(ascii_path: Path, cols: int, rows: int) -> np.ndarray:
    lines = ascii_path.read_text(encoding="utf-8").splitlines()
    matrix = np.zeros((rows, cols), dtype=int)
    row_idx = 0
    for line in lines:
        if "|" in line:
            parts = line.split("|")
            if len(parts) >= 2:
                chars = parts[1].strip()
                if len(chars) == cols and row_idx < rows:
                    for c_idx, ch in enumerate(chars):
                        if ch in ("■", "1", "#", "X"):
                            matrix[row_idx, c_idx] = 1
                    row_idx += 1
    return matrix


def parse_mask_file(mask_path: Path, cols: int, rows: int) -> np.ndarray:
    lines = mask_path.read_text(encoding="utf-8").splitlines()
    matrix = np.zeros((rows, cols), dtype=int)
    row_idx = 0
    for line in lines:
        if "|" in line:
            parts = line.split("|")
            if len(parts) >= 2:
                bits = parts[1].replace(" ", "").strip()
                if len(bits) == cols and row_idx < rows:
                    for c_idx, b in enumerate(bits):
                        if b in ("1", "■", "#", "X"):
                            matrix[row_idx, c_idx] = 1
                    row_idx += 1
        elif "=" in line and line.startswith("Y"):
            bits = line.split("=", 1)[1].strip()
            if len(bits) == cols and row_idx < rows:
                for c_idx, b in enumerate(bits):
                    if b in ("1", "■", "#", "X"):
                        matrix[row_idx, c_idx] = 1
                row_idx += 1
    return matrix


def parse_rle_file(rle_path: Path, cols: int, rows: int) -> np.ndarray:
    data = json.loads(rle_path.read_text(encoding="utf-8"))
    matrix = np.zeros((rows, cols), dtype=int)
    if "rle_runs" in data:
        runs = data["rle_runs"]
        flat = []
        for val, count in runs:
            flat.extend([int(val)] * int(count))
        if len(flat) == cols * rows:
            return np.array(flat, dtype=int).reshape((rows, cols))
    for row_info in data.get("rows", []):
        y = row_info.get("y", 0)
        if y < rows:
            for run in row_info.get("filled_runs", []):
                start_c, end_c = run[0], run[1]
                matrix[y, start_c:end_c + 1] = 1
    return matrix


def parse_svg_rects_xml(svg_path: Path, cols: int, rows: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    tree = ET.parse(svg_path)
    root = tree.getroot()

    has_root = root.tag.endswith("svg")
    rect_elements = [elem for elem in root.iter() if elem.tag.endswith("rect")]
    image_elements = [elem for elem in root.iter() if elem.tag.endswith("image")]
    
    svg_text = svg_path.read_text(encoding="utf-8")
    has_base64 = ("data:image" in svg_text) or ("base64" in svg_text)

    matrix = np.zeros((rows, cols), dtype=int)
    filled_rect_count = 0
    empty_rect_count = 0

    rect_data = []
    for rect in rect_elements:
        fill = rect.attrib.get("fill", "").lower()
        try:
            rx = float(rect.attrib.get("x", 0))
            ry = float(rect.attrib.get("y", 0))
            rw = float(rect.attrib.get("width", 0))
            rh = float(rect.attrib.get("height", 0))
        except ValueError:
            continue

        if fill in ("#111827", "#000000", "black", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"):
            filled_rect_count += 1
            rect_data.append((rx, ry, rw, rh))
        elif fill in ("#f3f4f6", "#ffffff", "white"):
            empty_rect_count += 1

    if rect_data:
        cell_w = rect_data[0][2]
        cell_h = rect_data[0][3]
        for rx, ry, rw, rh in rect_data:
            c = int(round(rx / cell_w))
            r = int(round(ry / cell_h))
            if 0 <= c < cols and 0 <= r < rows:
                matrix[r, c] = 1

    audit_info = {
        "has_root_svg": has_root,
        "total_rects": len(rect_elements),
        "filled_rects_count": filled_rect_count,
        "empty_rects_count": empty_rect_count,
        "has_image_tag": len(image_elements) > 0,
        "has_base64": has_base64
    }

    return matrix, audit_info


def parse_png_samples_file(png_path: Path, cols: int, rows: int) -> np.ndarray:
    if not png_path.exists():
        return np.zeros((rows, cols), dtype=int)

    with Image.open(png_path) as probe:
        px = probe.size[0] * probe.size[1]
        if px > MAX_IMAGE_PIXELS:
            raise ValueError(
                f"Refusing to decode '{png_path}': {px} pixels exceeds the "
                f"{MAX_IMAGE_PIXELS} pixel safety limit (possible decompression bomb)"
            )
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    matrix = np.zeros((rows, cols), dtype=int)

    step_x = w / cols
    step_y = h / rows

    for r in range(rows):
        for c in range(cols):
            cx = int((c + 0.5) * step_x)
            cy = int((r + 0.5) * step_y)
            cx = min(max(0, cx), w - 1)
            cy = min(max(0, cy), h - 1)

            rgb = img.getpixel((cx, cy))
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            if lum < 120:
                matrix[r, c] = 1

    return matrix


def validate_and_generate_real_diff_reports(
    safe_name: str,
    level_idx: int,
    base_out_dir: Path
) -> Dict[str, Any]:
    out_dir = base_out_dir / safe_name
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)

    summary_path = out_dir / "data" / "summary.json"
    if not summary_path.exists():
        summary_path = out_dir / "data" / f"{safe_name}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found at '{summary_path}'")

    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    lvl_info = next((l for l in summary_data.get("levels", []) if l["level"] == level_idx), None)
    if not lvl_info:
        raise ValueError(f"Level {level_idx} not found in summary JSON")

    grid_str = lvl_info.get("grid") or f"{lvl_info.get('cols')}x{lvl_info.get('rows')}"
    cols = lvl_info.get("cols", 0)
    rows = lvl_info.get("rows", 0)
    lvl_fmt = f"{level_idx:02d}"

    ascii_path = out_dir / "ascii" / f"{lvl_fmt}_{grid_str}_ascii.txt"
    if not ascii_path.exists():
        ascii_path = out_dir / "ascii" / f"{safe_name}_l{lvl_fmt}_{grid_str}_ascii.txt"

    mask_path = out_dir / "data" / f"{lvl_fmt}_{grid_str}_mask.txt"
    if not mask_path.exists():
        mask_path = out_dir / "data" / f"{safe_name}_l{lvl_fmt}_{grid_str}_mask.txt"

    rle_path = out_dir / "data" / f"{lvl_fmt}_{grid_str}_rle.json"
    if not rle_path.exists():
        rle_path = out_dir / "data" / f"{safe_name}_l{lvl_fmt}_{grid_str}_rle.json"

    card_svg = out_dir / "figures" / f"{lvl_fmt}_{grid_str}_card.svg"
    map_svg = out_dir / "figures" / f"{lvl_fmt}_{grid_str}_map.svg"
    if not card_svg.exists():
        card_svg = out_dir / "figures" / f"{safe_name}_l{lvl_fmt}_{grid_str}_card.svg"
    if not map_svg.exists():
        map_svg = out_dir / "figures" / f"{safe_name}_l{lvl_fmt}_{grid_str}_map.svg"

    svg_path = map_svg if map_svg.exists() else card_svg

    debug_val_png = Path("debug") / f"{lvl_fmt}_{grid_str}_grid_validation.png"
    fig_val_png = out_dir / "figures" / f"{lvl_fmt}_{grid_str}_grid_validation.png"
    png_path = debug_val_png if debug_val_png.exists() else (fig_val_png if fig_val_png.exists() else card_svg)

    mat_ascii = parse_ascii_file(ascii_path, cols, rows)
    mat_mask = parse_mask_file(mask_path, cols, rows)
    mat_rle = parse_rle_file(rle_path, cols, rows)
    mat_svg, svg_audit = parse_svg_rects_xml(svg_path, cols, rows)
    mat_png = parse_png_samples_file(png_path, cols, rows) if png_path.exists() else mat_svg

    cnt_ascii = int(np.sum(mat_ascii))
    cnt_mask = int(np.sum(mat_mask))
    cnt_rle = int(np.sum(mat_rle))
    cnt_svg = int(np.sum(mat_svg))
    cnt_png = int(np.sum(mat_png))

    diff_csv_path = debug_dir / f"{safe_name}_l{lvl_fmt}_matrix_diff.csv"
    with open(diff_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "ascii_value", "mask_value", "rle_value", "svg_value", "png_value", "match"])
        for r in range(rows):
            for c in range(cols):
                v_asc = int(mat_ascii[r, c])
                v_mask = int(mat_mask[r, c])
                v_rle = int(mat_rle[r, c])
                v_svg = int(mat_svg[r, c])
                v_png = int(mat_png[r, c])
                match_all = (v_asc == v_mask == v_rle == v_svg == v_png)
                writer.writerow([r, c, v_asc, v_mask, v_rle, v_svg, v_png, "TRUE" if match_all else "FALSE"])

    row_csv_path = debug_dir / f"{safe_name}_l{lvl_fmt}_row_counts.csv"
    with open(row_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "ascii_filled", "mask_filled", "rle_filled", "svg_filled", "png_filled", "match"])
        for r in range(rows):
            r_asc = int(np.sum(mat_ascii[r, :]))
            r_mask = int(np.sum(mat_mask[r, :]))
            r_rle = int(np.sum(mat_rle[r, :]))
            r_svg = int(np.sum(mat_svg[r, :]))
            r_png = int(np.sum(mat_png[r, :]))
            match_all = (r_asc == r_mask == r_rle == r_svg == r_png)
            writer.writerow([r, r_asc, r_mask, r_rle, r_svg, r_png, "TRUE" if match_all else "FALSE"])

    all_matched = (cnt_ascii == cnt_mask == cnt_rle == cnt_svg == cnt_png)

    return {
        "matched": all_matched,
        "grid": grid_str,
        "ascii_filled": cnt_ascii,
        "mask_filled": cnt_mask,
        "rle_filled": cnt_rle,
        "svg_filled": cnt_svg,
        "png_filled": cnt_png,
        "svg_audit": svg_audit,
        "diff_csv": diff_csv_path,
        "row_csv": row_csv_path
    }