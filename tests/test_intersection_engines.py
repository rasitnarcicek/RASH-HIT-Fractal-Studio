# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mehmet Raşit Narçiçek

"""Unit tests for the hierarchical quadtree engine and the CPU area wrapper."""

import unittest

from shapely.geometry import LineString, Polygon

from backend.geometry_engine import ParsedGeometry
from backend.grid_planner import GridLevel, create_grid_plan
from backend.intersection_cpu import CellDebugInfo, CPULevelResult
from backend.intersection_cpu_area import analyze_grid_cpu_area
from backend.intersection_hierarchical import (
    HierarchicalLevelResult,
    _bulk_fill_decision,
    analyze_grid_hierarchical,
)


def make_plan(num_levels=2, base_cells=2):
    return create_grid_plan(
        svg_viewbox=(0.0, 0.0, 100.0, 100.0),
        svg_width=100.0,
        svg_height=100.0,
        num_levels=num_levels,
        base_cells=base_cells,
    )


def fill_geom(poly):
    return ParsedGeometry('fill', poly)


def stroke_geom(line, width=0.0):
    return ParsedGeometry('stroke', line, stroke_width=width)


class TestBulkFillDecision(unittest.TestCase):

    def test_no_trees_leaves_cells_empty(self):
        mask = _bulk_fill_decision(
            [0.0], [0.0], [1.0], [1.0],
            None, [], None, [], []
        )
        self.assertEqual(mask.tolist(), [False])

    def test_fill_tree_marks_only_intersecting_cells(self):
        from shapely.strtree import STRtree
        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        tree = STRtree([poly])
        mask = _bulk_fill_decision(
            [0.0, 10.0], [0.0, 10.0], [1.0, 11.0], [1.0, 11.0],
            tree, [poly], None, [], []
        )
        self.assertEqual(mask.tolist(), [True, False])

    def test_zero_width_stroke_uses_intersection(self):
        import numpy as np
        from shapely.strtree import STRtree
        line = LineString([(0, 0), (0, 10)])
        tree = STRtree([line])
        mask = _bulk_fill_decision(
            [-1.0, 5.0], [0.0, 0.0], [1.0, 6.0], [1.0, 1.0],
            None, [], tree, [line], np.array([0.0])
        )
        self.assertEqual(mask.tolist(), [True, False])

    def test_wide_stroke_captures_nearby_cell_by_distance(self):
        import numpy as np
        from shapely.strtree import STRtree
        # Diagonal line so the cell overlaps its bbox (STRtree candidate) but not the line itself.
        line = LineString([(0, 0), (10, 10)])
        tree = STRtree([line])
        cell = ([8.0], [0.0], [9.0], [1.0])
        near = _bulk_fill_decision(*cell, None, [], tree, [line], np.array([20.0]))
        far = _bulk_fill_decision(*cell, None, [], tree, [line], np.array([1.0]))
        self.assertEqual(near.tolist(), [True])
        self.assertEqual(far.tolist(), [False])


class TestAnalyzeGridHierarchical(unittest.TestCase):

    def test_empty_geometry_list_returns_all_empty_levels(self):
        plan = make_plan()
        results, details = analyze_grid_hierarchical([], plan)
        self.assertEqual(len(results), len(plan.levels))
        for res, det, lvl in zip(results, details, plan.levels):
            self.assertEqual(res.filled_count, 0)
            self.assertEqual(res.empty_count, lvl.total_cells)
            self.assertEqual(res.fill_ratio, 0.0)
            self.assertIsInstance(det, HierarchicalLevelResult)
            self.assertEqual(det.exact_geos_tests, 0)

    def test_geometries_with_empty_shapes_are_ignored(self):
        plan = make_plan(num_levels=1)
        results, _ = analyze_grid_hierarchical([fill_geom(Polygon())], plan)
        self.assertEqual(results[0].filled_count, 0)

    def test_full_cover_fills_every_cell_on_all_levels(self):
        plan = make_plan(num_levels=3)
        geom = fill_geom(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
        results, details = analyze_grid_hierarchical([geom], plan)
        for res, lvl in zip(results, plan.levels):
            self.assertEqual(res.filled_count, lvl.total_cells)
            self.assertEqual(res.empty_count, 0)
        # Levels after the first subdivide every filled parent into 4 children.
        self.assertEqual(details[1].partial_parents_subdivided, plan.levels[0].total_cells)
        self.assertEqual(details[1].exact_geos_tests, plan.levels[0].total_cells * 4)

    def test_quadrant_fill_counts_and_indices(self):
        plan = make_plan(num_levels=1, base_cells=2)
        geom = fill_geom(Polygon([(0, 0), (49, 0), (49, 49), (0, 49)]))
        results, _ = analyze_grid_hierarchical([geom], plan, return_cell_indices=True)
        res = results[0]
        self.assertEqual(res.filled_count, 1)
        self.assertEqual(res.empty_count, plan.levels[0].total_cells - 1)
        self.assertEqual(res.filled_cells_indices, [(0, 0)])

    def test_indices_omitted_when_not_requested(self):
        plan = make_plan(num_levels=1)
        geom = fill_geom(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
        results, _ = analyze_grid_hierarchical([geom], plan)
        self.assertEqual(results[0].filled_cells_indices, [])

    def test_empty_parents_prune_children(self):
        plan = make_plan(num_levels=2, base_cells=2)
        geom = fill_geom(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
        _, details = analyze_grid_hierarchical([geom], plan)
        level0_filled = details[0].level_result.filled_count
        # Only filled parents are expanded, so level 2 tests 4 children per parent.
        self.assertEqual(details[1].exact_geos_tests, level0_filled * 4)

    def test_stroke_only_geometry_is_counted(self):
        plan = make_plan(num_levels=1, base_cells=2)
        geom = stroke_geom(LineString([(0, 0), (0, 100)]), width=1.0)
        results, _ = analyze_grid_hierarchical([geom], plan, return_cell_indices=True)
        self.assertGreater(results[0].filled_count, 0)

    def test_hierarchical_result_proxies_level_result(self):
        plan = make_plan(num_levels=1)
        inner = CPULevelResult(plan.levels[0], 3, 1, 1.5)
        det = HierarchicalLevelResult(inner, 1, 0, 2, 8, 0.5)
        self.assertIs(det.level, plan.levels[0])
        self.assertEqual(det.filled_count, 3)
        self.assertEqual(det.empty_count, 1)
        self.assertEqual(det.total_cells, plan.levels[0].total_cells)
        self.assertEqual(det.execution_time_ms, 1.5)
        self.assertAlmostEqual(det.fill_ratio, inner.fill_ratio)


class TestAnalyzeGridCPUArea(unittest.TestCase):

    def test_matches_hierarchical_counts(self):
        plan = make_plan(num_levels=2)
        geoms = [fill_geom(Polygon([(0, 0), (60, 0), (60, 60), (0, 60)]))]
        area_results = analyze_grid_cpu_area(geoms, plan)
        hier_results, _ = analyze_grid_hierarchical(geoms, plan)

        self.assertEqual(len(area_results), len(hier_results))
        for a, h in zip(area_results, hier_results):
            self.assertIsInstance(a, CPULevelResult)
            self.assertEqual(a.filled_count, h.filled_count)
            self.assertEqual(a.empty_count, h.empty_count)
            self.assertEqual(a.total_cells, h.total_cells)

    def test_propagates_cell_indices(self):
        plan = make_plan(num_levels=1, base_cells=2)
        geoms = [fill_geom(Polygon([(0, 0), (49, 0), (49, 49), (0, 49)]))]
        results = analyze_grid_cpu_area(geoms, plan, return_cell_indices=True)
        self.assertEqual(results[0].filled_cells_indices, [(0, 0)])

    def test_no_geometry_returns_empty_levels(self):
        plan = make_plan(num_levels=2)
        results = analyze_grid_cpu_area([], plan)
        for res, lvl in zip(results, plan.levels):
            self.assertEqual(res.filled_count, 0)
            self.assertEqual(res.empty_count, lvl.total_cells)


class TestCPUResultContainers(unittest.TestCase):

    def test_cpu_level_result_to_dict(self):
        lvl = GridLevel(1, 4, 2, 100.0, 50.0)
        res = CPULevelResult(lvl, filled_count=2, empty_count=6, execution_time_ms=1.234)
        d = res.to_dict()
        self.assertEqual(d['filled_cells'], 2)
        self.assertEqual(d['empty_cells'], 6)
        self.assertEqual(d['fill_ratio'], 0.25)
        self.assertEqual(d['execution_time_ms'], 1.23)
        self.assertEqual(d['mode'], 'CPU')
        self.assertEqual(d['cols'], 4)

    def test_fill_ratio_zero_when_no_cells(self):
        lvl = GridLevel(1, 1, 1, 10.0, 10.0)
        lvl.total_cells = 0
        res = CPULevelResult(lvl, 0, 0, 0.0)
        self.assertEqual(res.fill_ratio, 0.0)

    def test_cell_debug_info_to_dict_rounds_values(self):
        info = CellDebugInfo(
            level_idx=2, col=3, row=4, reason='fill-hit', geometry_id=7,
            geometry_type='path', fill_or_stroke='fill', stroke_width=1.234567,
            cell_bounds=(0.123456, 1.0, 2.0, 3.987654),
        )
        d = info.to_dict()
        self.assertEqual(d['level'], 2)
        self.assertEqual(d['stroke_width'], 1.2346)
        self.assertEqual(d['cell_bounds'], [0.1235, 1.0, 2.0, 3.9877])


if __name__ == '__main__':
    unittest.main()
