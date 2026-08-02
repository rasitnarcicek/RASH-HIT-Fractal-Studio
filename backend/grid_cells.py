# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mehmet Raşit Narçiçek

"""
grid_cells.py — Shared grid cell labelling, coordinate and SVG fragment helpers.

Exporters repeatedly derive the same artifact naming tokens (``01``, ``4x8``),
cell bounding boxes and cell centers from a grid level, and emit identical SVG
cell-fill / grid-overlay markup. These helpers keep those derivations in one place.
"""

from __future__ import annotations
from typing import Iterable, List, Set, Tuple

FILL_COLOR = "#60A5FA"


def level_tag(level: int) -> str:
    """Zero-padded level token used in artifact filenames and titles (``01``)."""
    return f"{level:02d}"


def grid_label(cols: int, rows: int) -> str:
    """Grid resolution token used in artifact filenames and titles (``4x8``)."""
    return f"{cols}x{rows}"


def artifact_tags(level: int, cols: int, rows: int) -> Tuple[str, str]:
    """Returns the ``(level_tag, grid_label)`` pair used to name per-level artifacts."""
    return level_tag(level), grid_label(cols, rows)


def cell_bounds(col: int, row: int, cell_w: float, cell_h: float) -> Tuple[float, float, float, float]:
    """Axis-aligned SVG bounding box ``(xmin, ymin, xmax, ymax)`` of a grid cell."""
    xmin = col * cell_w
    ymin = row * cell_h
    return xmin, ymin, xmin + cell_w, ymin + cell_h


def cell_center(col: int, row: int, cell_w: float, cell_h: float) -> Tuple[float, float]:
    """SVG center point ``(cx, cy)`` of a grid cell."""
    xmin, ymin, xmax, ymax = cell_bounds(col, row, cell_w, cell_h)
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0


def svg_cell_fill_lines(
    filled_cells: Iterable[Tuple[int, int]],
    cell_w: float,
    cell_h: float,
    indent: str
) -> List[str]:
    """``<rect>`` markup for every filled cell, given ``(row, col)`` index pairs."""
    lines = []
    for (r, c) in filled_cells:
        x = c * cell_w
        y = r * cell_h
        lines.append(
            f'{indent}<rect x="{x:.4f}" y="{y:.4f}" '
            f'width="{cell_w:.4f}" height="{cell_h:.4f}" fill="{FILL_COLOR}"/>'
        )
    return lines


def svg_grid_overlay_lines(
    cols: int,
    rows: int,
    cell_w: float,
    cell_h: float,
    view_w: float,
    view_h: float,
    indent: str
) -> List[str]:
    """``<line>`` markup for the full vertical + horizontal grid overlay."""
    lines = []
    for c in range(cols + 1):
        x = c * cell_w
        lines.append(f'{indent}<line x1="{x:.4f}" y1="0" x2="{x:.4f}" y2="{view_h:.4f}"/>')
    for r in range(rows + 1):
        y = r * cell_h
        lines.append(f'{indent}<line x1="0" y1="{y:.4f}" x2="{view_w:.4f}" y2="{y:.4f}"/>')
    return lines


def cell_record_row(
    cell_id: int,
    col: int,
    row: int,
    cell_w: float,
    cell_h: float,
    filled_set: Set[Tuple[int, int]]
) -> Tuple[int, str, str, int, int, str, int, float, float, float, float, float, float]:
    """One row of the 13-column per-cell technical table shared by CSV/XLSX/HTML exports."""
    is_filled = (col, row) in filled_set
    xmin, ymin, xmax, ymax = cell_bounds(col, row, cell_w, cell_h)
    cx, cy = cell_center(col, row, cell_w, cell_h)
    return (
        cell_id,
        f"R{row + 1:02d}",
        f"C{col + 1:02d}",
        row,
        col,
        "filled" if is_filled else "empty",
        1 if is_filled else 0,
        xmin, ymin, xmax, ymax, cx, cy,
    )
