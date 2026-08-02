# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mehmet Raşit Narçiçek

"""Unit tests for grid planning, output profiles and fractal regression."""

import math
import unittest

from backend.fractal_analyzer import compute_fractal_dimension
from backend.grid_planner import GridLevel, GridPlan, create_grid_plan
from backend.intersection_cpu import CPULevelResult
from backend.output_profiles import PROFILES, load_output_profile


class TestGridLevel(unittest.TestCase):

    def test_cell_metrics(self):
        lvl = GridLevel(1, 4, 2, 100.0, 50.0)
        self.assertEqual(lvl.total_cells, 8)
        self.assertEqual((lvl.cell_w, lvl.cell_h), (25.0, 25.0))
        self.assertEqual(lvl.cell_aspect_ratio, 1.0)
        self.assertAlmostEqual(lvl.scale_epsilon, 0.25)
        self.assertAlmostEqual(lvl.log_inv_epsilon, math.log(4.0))

    def test_degenerate_dimensions_fall_back_to_defaults(self):
        lvl = GridLevel(1, 1, 1, 0.0, 0.0)
        self.assertEqual(lvl.cell_aspect_ratio, 1.0)
        self.assertEqual(lvl.scale_epsilon, 1.0)
        self.assertEqual(lvl.log_inv_epsilon, 0.0)

    def test_to_dict_rounds_values(self):
        d = GridLevel(3, 3, 3, 10.0, 10.0).to_dict()
        self.assertEqual(d['level'], 3)
        self.assertEqual(d['total_cells'], 9)
        self.assertAlmostEqual(d['cell_w'], 3.3333)


class TestCreateGridPlan(unittest.TestCase):

    def test_viewbox_takes_priority(self):
        plan = create_grid_plan((10.0, 20.0, 200.0, 100.0), 999.0, 999.0, num_levels=1)
        self.assertEqual((plan.xmin, plan.ymin, plan.xmax, plan.ymax), (10.0, 20.0, 210.0, 120.0))
        self.assertAlmostEqual(plan.aspect_ratio, 2.0)

    def test_width_height_used_when_viewbox_missing(self):
        plan = create_grid_plan(None, 80.0, 40.0, num_levels=1)
        self.assertEqual((plan.xmax, plan.ymax), (80.0, 40.0))

    def test_geometry_bounds_used_as_last_resort(self):
        plan = create_grid_plan(None, 0.0, 0.0, geometry_bounds=(1.0, 2.0, 11.0, 12.0), num_levels=1)
        self.assertEqual((plan.xmin, plan.ymin, plan.xmax, plan.ymax), (1.0, 2.0, 11.0, 12.0))

    def test_emergency_default_box(self):
        plan = create_grid_plan(None, 0.0, 0.0, num_levels=1)
        self.assertEqual((plan.xmax, plan.ymax), (100.0, 100.0))

    def test_zero_size_viewbox_is_rejected(self):
        plan = create_grid_plan((0.0, 0.0, 0.0, 0.0), 30.0, 15.0, num_levels=1)
        self.assertEqual((plan.xmax, plan.ymax), (30.0, 15.0))

    def test_levels_double_each_step_for_wide_canvas(self):
        plan = create_grid_plan((0.0, 0.0, 200.0, 100.0), 200.0, 100.0, num_levels=3, base_cells=4)
        self.assertEqual([(l.cols, l.rows) for l in plan.levels], [(8, 4), (16, 8), (32, 16)])
        self.assertEqual([l.level_idx for l in plan.levels], [1, 2, 3])

    def test_levels_for_tall_canvas(self):
        plan = create_grid_plan((0.0, 0.0, 100.0, 200.0), 100.0, 200.0, num_levels=2, base_cells=4)
        self.assertEqual([(l.cols, l.rows) for l in plan.levels], [(4, 8), (8, 16)])

    def test_manual_grids_override_automatic_planning(self):
        plan = create_grid_plan((0.0, 0.0, 100.0, 100.0), 100.0, 100.0,
                                manual_grids=[(3, 5), (0, 0)])
        self.assertEqual([(l.cols, l.rows) for l in plan.levels], [(3, 5), (1, 1)])

    def test_degenerate_bounds_are_clamped(self):
        plan = GridPlan((5.0, 5.0, 5.0, 5.0), [])
        self.assertEqual(plan.width, 1e-6)
        self.assertEqual(plan.height, 1e-6)


class TestOutputProfiles(unittest.TestCase):

    def test_default_profile_is_lean(self):
        self.assertEqual(load_output_profile().name, 'lean')
        self.assertEqual(load_output_profile('').name, 'lean')

    def test_named_profiles_are_case_insensitive(self):
        self.assertIs(load_output_profile('  DEBUG '), PROFILES['debug'])

    def test_reproducible_profile_enables_reproducibility_artifacts(self):
        p = load_output_profile('reproducible')
        self.assertTrue(p.generate_masks and p.generate_rle and p.generate_summary_json)
        self.assertFalse(p.generate_raw_csv)

    def test_unknown_profile_raises_with_valid_keys(self):
        with self.assertRaises(ValueError) as ctx:
            load_output_profile('nope')
        self.assertIn('lean', str(ctx.exception))


def level_result(level_idx, cols, rows, filled):
    lvl = GridLevel(level_idx, cols, rows, 100.0, 100.0)
    return CPULevelResult(lvl, filled, lvl.total_cells - filled, 0.0)


class TestFractalDimension(unittest.TestCase):

    def test_no_levels_returns_zeros(self):
        res = compute_fractal_dimension([])
        self.assertEqual((res.fractal_dimension_db, res.r2_score), (0.0, 0.0))
        self.assertEqual(res.level_results, [])

    def test_single_usable_level_cannot_regress(self):
        res = compute_fractal_dimension([level_result(1, 2, 2, 3)])
        self.assertEqual(res.fractal_dimension_db, 0.0)
        self.assertEqual(res.scaling_levels_used, [1])

    def test_space_filling_square_has_dimension_two(self):
        results = [level_result(i + 1, 2 ** (i + 1), 2 ** (i + 1), 4 ** (i + 1)) for i in range(4)]
        res = compute_fractal_dimension(results)
        self.assertAlmostEqual(res.fractal_dimension_db, 2.0, places=6)
        self.assertAlmostEqual(res.r2_score, 1.0, places=9)

    def test_selected_levels_filter_regression_input(self):
        results = [level_result(i + 1, 2 ** (i + 1), 2 ** (i + 1), 4 ** (i + 1)) for i in range(4)]
        res = compute_fractal_dimension(results, selected_levels=[2, 3])
        self.assertEqual(res.scaling_levels_used, [2, 3])
        self.assertEqual(len(res.level_results), 4)

    def test_empty_levels_are_skipped(self):
        results = [level_result(1, 2, 2, 0)] + [
            level_result(i + 2, 2 ** (i + 2), 2 ** (i + 2), 4 ** (i + 2)) for i in range(3)
        ]
        res = compute_fractal_dimension(results)
        self.assertNotIn(1, res.scaling_levels_used)

    def test_constant_fill_count_yields_nan_r2(self):
        results = [level_result(i + 1, 2 ** (i + 1), 2 ** (i + 1), 5) for i in range(3)]
        res = compute_fractal_dimension(results)
        self.assertTrue(math.isnan(res.r2_score))
        self.assertAlmostEqual(res.fractal_dimension_db, 0.0, places=9)

    def test_to_dict_contains_rounded_summary(self):
        results = [level_result(i + 1, 2 ** (i + 1), 2 ** (i + 1), 4 ** (i + 1)) for i in range(3)]
        d = compute_fractal_dimension(results).to_dict()
        self.assertEqual(d['fractal_dimension_db'], 2.0)
        self.assertEqual(d['scaling_levels_used'], [1, 2, 3])
        self.assertEqual(len(d['levels']), 3)


if __name__ == '__main__':
    unittest.main()
