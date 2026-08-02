# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mehmet Raşit Narçiçek

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List
from datetime import datetime

from backend.svg_loader import SVGLoader
from backend.geometry_engine import extract_node_geometries, ParsedGeometry
from backend.grid_planner import create_grid_plan
from backend.intersection_cpu_area import analyze_grid_cpu_area
from backend.academic_exporter import (
    export_academic_package_v3,
    AnalysisReportModel,
    LevelReportModel,
    sanitize_output_slug,
)
from backend.fractal_analyzer import compute_fractal_dimension

def process_single_file(input_file: str, engine: str, measure_mode: str, levels: int, profile: str, output_root: Path, export_high_level: bool):
    if levels < 1:
        print(f"Error: Invalid --levels '{levels}'. Number of grid levels must be >= 1.", file=sys.stderr)
        sys.exit(1)

    print("============================================================")
    print("RASH-HIT Fractal Studio v1.0.0 - Running Analysis")
    print("============================================================")
    print(f"Input File      : {os.path.basename(input_file)}")
    print(f"Grid Levels     : L01..L{levels:02d}")
    print(f"Measure Mode    : {measure_mode}")
    print("Selected Engine : CPU Exact Vector Geometry Engine")
    print("------------------------------------------------------------")

    # Load SVG
    t0_load = time.perf_counter()
    try:
        loader = SVGLoader(input_file)
        elements = loader.get_elements()
        geoms: List[ParsedGeometry] = []
        for node, style in elements:
            geoms.extend(extract_node_geometries(node, style))
    except Exception as e:
        raise RuntimeError(f"Failed to load or parse SVG file '{input_file}': {e}") from e
    t1_load = time.perf_counter()
    print(f"[+] Loaded {len(geoms)} geometries in {(t1_load - t0_load)*1000:.2f} ms")

    # Create Grid Plan
    vw, vh = loader.viewbox[2], loader.viewbox[3]
    grid_plan = create_grid_plan(loader.viewbox, vw, vh, num_levels=levels)

    # Execute Engine
    need_cell_indices = export_high_level or (levels <= 8)

    t0_calc = time.perf_counter()
    results = analyze_grid_cpu_area(geoms, grid_plan, return_cell_indices=need_cell_indices)
    t1_calc = time.perf_counter()

    calc_time_ms = (t1_calc - t0_calc) * 1000.0

    print("------------------------------------------------------------")
    print("SUMMARY RESULTS")
    print("------------------------------------------------------------")
    print(f"{'Level':<6} {'Grid':<10} {'Total Cells':>12} {'Filled Cells':>13} {'Empty Cells':>13} {'Fill %':>9} {'Time ms':>10}")
    for r in results:
        print(f"L{r.level.level_idx:02d}    {r.level.cols}x{r.level.rows:<6} {r.level.total_cells:>12,} {r.filled_count:>13,} {r.empty_count:>13,} {r.fill_ratio*100:>8.2f}% {r.execution_time_ms:>10.2f}")
    print("------------------------------------------------------------")
    print(f"Total Computation Time: {calc_time_ms:.2f} ms")

    # Export Full Package
    input_path = Path(input_file)
    output_root.mkdir(parents=True, exist_ok=True)
    print("------------------------------------------------------------")
    print(f"[+] Exporting Academic Package v3 to: {output_root}")

    try:
        level_models = []
        skipped_list = []
        for r in results:
            lvl_num = r.level.level_idx
            if lvl_num <= 7:
                is_export_safe = True
            elif lvl_num == 8:
                is_export_safe = export_high_level
            else:
                is_export_safe = False

            f_set = set(r.filled_cells_indices) if (r.filled_cells_indices and is_export_safe) else set()

            if not is_export_safe:
                if lvl_num == 8:
                    reason_msg = "Skipped by default policy (use --export-high-level-tables to include L08)."
                else:
                    reason_msg = "Skipped by safety policy (L09+ full table exports are permanently disabled)."
                skipped_list.append({
                    "level": lvl_num,
                    "artifact": "Full Cell Data Tables",
                    "reason": reason_msg
                })

            level_models.append(LevelReportModel(
                level=lvl_num,
                cols=r.level.cols,
                rows=r.level.rows,
                grid_label=f"{r.level.cols}x{r.level.rows}",
                total_cells=r.level.total_cells,
                filled_cells=r.filled_count,
                empty_cells=r.empty_count,
                fill_ratio=r.fill_ratio,
                occupancy_percent=r.fill_ratio * 100.0,
                cell_w=r.level.cell_w,
                cell_h=r.level.cell_h,
                execution_time_ms=r.execution_time_ms,
                mode="cpu",
                export_full_cells=is_export_safe,
                filled_set=f_set
            ))

        fractal_res = compute_fractal_dimension(results)
        safe_stem = sanitize_output_slug(input_path.stem)

        report_model = AnalysisReportModel(
            motif=safe_stem,
            safe_name=safe_stem,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_file=input_path.name,
            viewbox_width=vw,
            viewbox_height=vh,
            aspect_ratio=(vw / vh) if vh != 0 else 1.0,
            vector_geometry_count=len(geoms),
            analysis_engine="cpu",
            
            
            
            
            db=fractal_res.fractal_dimension_db,
            r2=fractal_res.r2_score,
            total_time_ms=calc_time_ms,
            hardware_info="CPU Exact Vector Geometry Engine",
            levels=level_models
        )

        export_academic_package_v3(
            model=report_model,
            output_root=output_root,
            profile=profile,
            skipped_outputs=skipped_list,
            measure_mode=measure_mode
        )
        print(f"[OK] Export Completed! Package generated at:\n     {output_root / safe_stem}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Academic package export failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(description="RASH-HIT Fractal Studio - Vector Geometry Analysis & Box-Counting Engine")
    _input_group = parser.add_mutually_exclusive_group(required=False)
    _input_group.add_argument("--input", type=str, help="Input SVG file path")
    _input_group.add_argument("--dir", type=str, help="Directory path for batch processing all SVG files")
    parser.add_argument("--engine", type=str, default="cpu", choices=["cpu"], help="Engine selection (default: cpu)")
    parser.add_argument("--measure", type=str, default="area", choices=["area"], help="Measurement mode (default: area)")
    parser.add_argument("--levels", type=int, default=7, help="Number of grid levels (default: 7)")
    parser.add_argument("--profile", type=str, default="lean", choices=["lean", "reproducible", "debug", "presentation"], help="Profiling mode (default: lean)")
    parser.add_argument("--output-dir", type=str, required=False, help="Custom output directory for exported reports")
    parser.add_argument("--export-high-level-tables", action="store_true", help="Force export cell tables for L08+")

    args = parser.parse_args()

    if args.levels < 1:
        print(f"Error: Invalid --levels '{args.levels}'. Number of grid levels must be >= 1.", file=sys.stderr)
        sys.exit(1)


    target_input = args.input or args.dir
    if not target_input:
        parser.print_help()
        return

    output_root = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "outputs"

    target_path = Path(target_input)
    if target_path.is_file():
        try:
            process_single_file(str(target_path), args.engine, args.measure, args.levels, args.profile, output_root, args.export_high_level_tables)
        except Exception as e:
            print(f"[ERROR] Analysis failed for '{target_path.name}': {e}", file=sys.stderr)
            sys.exit(1)
    elif target_path.is_dir():
        svg_files = sorted(list(target_path.glob("*.svg")))
        if not svg_files:
            print(f"[!] No SVG files found in directory: {target_path}")
            sys.exit(1)
        
        print("============================================================")
        print(f"BATCH PROCESSING MODE: {len(svg_files)} SVG Files Found")
        print("============================================================")
        failed: List[str] = []
        for idx, svg_f in enumerate(svg_files, start=1):
            print(f"\n>>> [{idx}/{len(svg_files)}] Processing: {svg_f.name}")
            try:
                process_single_file(str(svg_f), args.engine, args.measure, args.levels, args.profile, output_root, args.export_high_level_tables)
            except Exception as e:
                print(f"[ERROR] Failed to process '{svg_f.name}': {e}", file=sys.stderr)
                failed.append(svg_f.name)
        if failed:
            print(f"[ERROR] Batch completed with {len(failed)}/{len(svg_files)} failure(s): {', '.join(failed)}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[!] Error: Target path not found: {target_input}")
        sys.exit(1)

if __name__ == "__main__":
    main()
