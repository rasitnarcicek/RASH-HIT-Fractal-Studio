# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mehmet Raşit Narçiçek

"""Unit tests for path parsing, curve flattening, transforms and geometry extraction."""

import math
import unittest

import numpy as np

from backend.geometry_engine import (
    ParsedGeometry,
    extract_node_geometries,
    parse_svg_path,
    parse_transform_string,
    sample_cubic_bezier,
    sample_elliptical_arc,
    sample_quadratic_bezier,
    transform_points,
)
from backend.svg_loader import SVGNode


def node(tag, attribs, styles=None):
    return SVGNode(tag, attribs, styles if styles is not None else {'fill': 'black'}, '')


class TestTransforms(unittest.TestCase):

    def test_empty_transform_is_identity(self):
        np.testing.assert_allclose(parse_transform_string(''), np.eye(3))

    def test_translate_with_single_argument_defaults_ty(self):
        M = parse_transform_string('translate(5)')
        self.assertEqual(transform_points([(0.0, 0.0)], M), [(5.0, 0.0)])

    def test_uniform_scale_from_single_argument(self):
        M = parse_transform_string('scale(3)')
        self.assertEqual(transform_points([(2.0, 4.0)], M), [(6.0, 12.0)])

    def test_matrix_command(self):
        M = parse_transform_string('matrix(1 0 0 1 10 20)')
        self.assertEqual(transform_points([(1.0, 2.0)], M), [(11.0, 22.0)])

    def test_rotate_about_origin(self):
        M = parse_transform_string('rotate(90)')
        x, y = transform_points([(1.0, 0.0)], M)[0]
        self.assertAlmostEqual(x, 0.0, places=9)
        self.assertAlmostEqual(y, 1.0, places=9)

    def test_rotate_about_center_keeps_center_fixed(self):
        M = parse_transform_string('rotate(37, 5, 7)')
        x, y = transform_points([(5.0, 7.0)], M)[0]
        self.assertAlmostEqual(x, 5.0, places=9)
        self.assertAlmostEqual(y, 7.0, places=9)

    def test_composed_transforms_apply_left_to_right(self):
        M = parse_transform_string('translate(10,0) scale(2)')
        self.assertEqual(transform_points([(3.0, 1.0)], M), [(16.0, 2.0)])

    def test_unknown_command_is_ignored(self):
        np.testing.assert_allclose(parse_transform_string('bogus(1,2)'), np.eye(3))

    def test_transform_points_with_no_points(self):
        self.assertEqual(transform_points([], np.eye(3)), [])


class TestCurveSampling(unittest.TestCase):

    def test_cubic_bezier_endpoint_and_count(self):
        pts = sample_cubic_bezier((0, 0), (0, 10), (10, 10), (10, 0), num_steps=8)
        self.assertEqual(len(pts), 8)
        self.assertAlmostEqual(pts[-1][0], 10.0)
        self.assertAlmostEqual(pts[-1][1], 0.0)

    def test_quadratic_bezier_endpoint_and_count(self):
        pts = sample_quadratic_bezier((0, 0), (5, 10), (10, 0), num_steps=4)
        self.assertEqual(len(pts), 4)
        self.assertAlmostEqual(pts[-1][0], 10.0)
        self.assertAlmostEqual(pts[-1][1], 0.0)

    def test_arc_with_identical_endpoints_is_empty(self):
        self.assertEqual(sample_elliptical_arc((1, 1), 5, 5, 0, False, True, (1, 1)), [])

    def test_arc_with_zero_radius_degenerates_to_line_endpoint(self):
        self.assertEqual(sample_elliptical_arc((0, 0), 0, 5, 0, False, True, (4, 3)), [(4, 3)])

    def test_arc_endpoint_matches_target(self):
        pts = sample_elliptical_arc((0, 0), 5, 5, 0, False, True, (10, 0), num_steps=12)
        self.assertEqual(len(pts), 12)
        self.assertAlmostEqual(pts[-1][0], 10.0, places=6)
        self.assertAlmostEqual(pts[-1][1], 0.0, places=6)

    def test_arc_radii_are_scaled_up_when_too_small(self):
        # rx/ry too small to span the endpoints: implementation enlarges them.
        pts = sample_elliptical_arc((0, 0), 1, 1, 0, False, True, (10, 0), num_steps=8)
        self.assertAlmostEqual(pts[-1][0], 10.0, places=6)
        self.assertAlmostEqual(pts[-1][1], 0.0, places=6)

    def test_large_arc_flag_produces_longer_sweep(self):
        small = sample_elliptical_arc((0, 0), 5, 5, 0, False, True, (5, 5), num_steps=32)
        large = sample_elliptical_arc((0, 0), 5, 5, 0, True, True, (5, 5), num_steps=32)

        def length(pts):
            return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
        self.assertGreater(length(large), length(small))


class TestParseSVGPath(unittest.TestCase):

    def test_empty_string_returns_no_subpaths(self):
        self.assertEqual(parse_svg_path(''), [])

    def test_absolute_line_commands(self):
        self.assertEqual(
            parse_svg_path('M 0 0 L 10 0 L 10 10 Z'),
            [[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 0.0)]],
        )

    def test_relative_commands_accumulate(self):
        self.assertEqual(
            parse_svg_path('m 1 1 l 2 0 l 0 2'),
            [[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0)]],
        )

    def test_horizontal_and_vertical_commands(self):
        self.assertEqual(
            parse_svg_path('M 0 0 H 5 V 5 h -5 v -5'),
            [[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (0.0, 0.0)]],
        )

    def test_implicit_lineto_after_moveto(self):
        self.assertEqual(
            parse_svg_path('M 0 0 1 1 2 2'),
            [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]],
        )

    def test_multiple_subpaths(self):
        subs = parse_svg_path('M 0 0 L 1 0 M 5 5 L 6 5')
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[1][0], (5.0, 5.0))

    def test_smooth_cubic_reflects_previous_control_point(self):
        subs = parse_svg_path('M 0 0 C 0 5 5 5 5 0 S 10 -5 10 0', tolerance_steps=4)
        self.assertAlmostEqual(subs[0][-1][0], 10.0)
        self.assertAlmostEqual(subs[0][-1][1], 0.0)

    def test_smooth_quadratic_without_previous_control_point(self):
        subs = parse_svg_path('M 0 0 T 10 0', tolerance_steps=4)
        self.assertAlmostEqual(subs[0][-1][0], 10.0)

    def test_quadratic_then_smooth_quadratic(self):
        subs = parse_svg_path('M 0 0 Q 5 5 10 0 T 20 0', tolerance_steps=4)
        self.assertAlmostEqual(subs[0][-1][0], 20.0)
        self.assertAlmostEqual(subs[0][-1][1], 0.0)

    def test_arc_command_reaches_endpoint(self):
        subs = parse_svg_path('M 0 0 A 5 5 0 0 1 10 0', tolerance_steps=8)
        self.assertAlmostEqual(subs[0][-1][0], 10.0, places=6)

    def test_scientific_notation_numbers(self):
        subs = parse_svg_path('M 0 0 L 1e2 0')
        self.assertEqual(subs[0][-1], (100.0, 0.0))

    def test_path_not_starting_with_command_raises(self):
        with self.assertRaises(ValueError):
            parse_svg_path('10 10 L 20 20')

    def test_unknown_command_raises(self):
        with self.assertRaises(ValueError):
            parse_svg_path('M 0 0 K 5 5')

    def test_truncated_parameter_list_raises(self):
        with self.assertRaises(ValueError):
            parse_svg_path('M 0 0 L 10')

    def test_command_where_number_expected_raises(self):
        with self.assertRaises(ValueError):
            parse_svg_path('M 0 0 L 10 L')


class TestExtractNodeGeometries(unittest.TestCase):

    def test_rect_fill_area(self):
        geoms = extract_node_geometries(node('rect', {'x': '0', 'y': '0', 'width': '10', 'height': '4'}), [])
        self.assertEqual(len(geoms), 1)
        self.assertEqual(geoms[0].geom_type, 'fill')
        self.assertAlmostEqual(geoms[0].shapely_obj.area, 40.0)

    def test_zero_size_rect_yields_nothing(self):
        self.assertEqual(extract_node_geometries(node('rect', {'width': '0', 'height': '5'}), []), [])

    def test_circle_area_approximates_pi_r_squared(self):
        geoms = extract_node_geometries(node('circle', {'cx': '0', 'cy': '0', 'r': '10'}), [])
        self.assertAlmostEqual(geoms[0].shapely_obj.area, math.pi * 100, delta=5.0)

    def test_zero_radius_circle_yields_nothing(self):
        self.assertEqual(extract_node_geometries(node('circle', {'r': '0'}), []), [])

    def test_ellipse_bounds(self):
        geoms = extract_node_geometries(node('ellipse', {'cx': '0', 'cy': '0', 'rx': '10', 'ry': '5'}), [])
        xmin, ymin, xmax, ymax = geoms[0].shapely_obj.bounds
        self.assertAlmostEqual(xmax - xmin, 20.0, delta=0.5)
        self.assertAlmostEqual(ymax - ymin, 10.0, delta=0.5)

    def test_line_produces_stroke_only(self):
        n = node('line', {'x1': '0', 'y1': '0', 'x2': '10', 'y2': '0'},
                 styles={'fill': 'none', 'stroke': 'black', 'stroke-width': '2'})
        geoms = extract_node_geometries(n, [])
        self.assertEqual([g.geom_type for g in geoms], ['stroke'])
        self.assertAlmostEqual(geoms[0].stroke_width, 2.0)

    def test_polygon_is_closed_automatically(self):
        geoms = extract_node_geometries(node('polygon', {'points': '0,0 10,0 10,10'}), [])
        self.assertAlmostEqual(geoms[0].shapely_obj.area, 50.0)

    def test_polyline_with_single_point_yields_nothing(self):
        self.assertEqual(extract_node_geometries(node('polyline', {'points': '1,1'}), []), [])

    def test_unsupported_tag_yields_nothing(self):
        self.assertEqual(extract_node_geometries(node('text', {'x': '0'}), []), [])

    def test_transform_stack_scales_geometry_and_stroke_width(self):
        n = node('rect', {'x': '0', 'y': '0', 'width': '10', 'height': '10'},
                 styles={'fill': 'black', 'stroke': 'black', 'stroke-width': '2'})
        geoms = extract_node_geometries(n, ['scale(2)', 'translate(1,1)'])
        fill = next(g for g in geoms if g.geom_type == 'fill')
        stroke = next(g for g in geoms if g.geom_type == 'stroke')
        self.assertAlmostEqual(fill.shapely_obj.area, 400.0)
        self.assertAlmostEqual(fill.shapely_obj.bounds[0], 2.0)
        self.assertAlmostEqual(stroke.stroke_width, 4.0)

    def test_evenodd_fill_rule_creates_hole(self):
        d = 'M 0 0 L 10 0 L 10 10 L 0 10 Z M 3 3 L 7 3 L 7 7 L 3 7 Z'
        n = node('path', {'d': d}, styles={'fill': 'black', 'fill-rule': 'evenodd'})
        geoms = extract_node_geometries(n, [])
        self.assertAlmostEqual(geoms[0].shapely_obj.area, 100.0 - 16.0)

    def test_nonzero_fill_rule_unions_subpaths(self):
        d = 'M 0 0 L 10 0 L 10 10 L 0 10 Z M 3 3 L 7 3 L 7 7 L 3 7 Z'
        n = node('path', {'d': d}, styles={'fill': 'black'})
        geoms = extract_node_geometries(n, [])
        self.assertAlmostEqual(geoms[0].shapely_obj.area, 100.0)

    def test_self_intersecting_polygon_is_repaired(self):
        n = node('path', {'d': 'M 0 0 L 10 10 L 10 0 L 0 10 Z'}, styles={'fill': 'black'})
        geoms = extract_node_geometries(n, [])
        self.assertTrue(geoms[0].shapely_obj.is_valid)

    def test_parsed_geometry_bounds_default_for_empty_object(self):
        self.assertEqual(ParsedGeometry('fill', None).bounds, (0, 0, 0, 0))


if __name__ == '__main__':
    unittest.main()
