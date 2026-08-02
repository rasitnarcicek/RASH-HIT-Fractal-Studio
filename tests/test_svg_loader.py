# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mehmet Raşit Narçiçek

"""Unit tests for SVG loading, CSS resolution and style priority rules."""

import tempfile
import unittest
from pathlib import Path

from backend.svg_loader import (
    SVGLoader,
    SVGNode,
    parse_css_style_block,
    parse_length,
    parse_style_attribute,
)


def write_svg(content: str) -> Path:
    f = tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False, suffix='.svg')
    f.write(content)
    f.close()
    return Path(f.name)


class TestParseLength(unittest.TestCase):

    def test_empty_returns_default(self):
        self.assertEqual(parse_length('', default=7.0), 7.0)

    def test_plain_number_and_px(self):
        self.assertEqual(parse_length('12'), 12.0)
        self.assertEqual(parse_length('.24px'), 0.24)

    def test_point_and_em_units(self):
        self.assertAlmostEqual(parse_length('12pt'), 12 * 1.33333)
        self.assertAlmostEqual(parse_length('2em'), 32.0)

    def test_invalid_values_return_default(self):
        self.assertEqual(parse_length('abc', default=3.0), 3.0)
        self.assertEqual(parse_length('xxpt', default=3.0), 3.0)
        self.assertEqual(parse_length('xxem', default=3.0), 3.0)


class TestStyleParsing(unittest.TestCase):

    def test_inline_style_attribute(self):
        self.assertEqual(
            parse_style_attribute('Fill: #FFF; stroke-width: 2px; junk'),
            {'fill': '#fff', 'stroke-width': '2px'},
        )

    def test_empty_style_attribute(self):
        self.assertEqual(parse_style_attribute(''), {})

    def test_empty_css_block(self):
        self.assertEqual(parse_css_style_block(''), {})

    def test_css_block_multiple_selectors_and_comments(self):
        rules = parse_css_style_block("""
            /* comment */
            .cls-1 { fill: #1d1d1b; }
            .cls-2, .cls-3 { fill: none; }
            .cls-3 { stroke: #000; stroke-width: .24px; }
        """)
        self.assertEqual(rules['cls-1']['fill'], '#1d1d1b')
        self.assertEqual(rules['cls-2']['fill'], 'none')
        self.assertEqual(rules['cls-3'], {'fill': 'none', 'stroke': '#000', 'stroke-width': '.24px'})

    def test_non_class_selectors_are_ignored(self):
        self.assertEqual(parse_css_style_block('rect { fill: red; }'), {})


class TestSVGNode(unittest.TestCase):

    def test_defaults_are_filled_and_visible(self):
        n = SVGNode('path', {}, {}, '')
        self.assertTrue(n.has_fill)
        self.assertFalse(n.has_stroke)
        self.assertTrue(n.is_visible)

    def test_namespaced_tag_is_stripped(self):
        self.assertEqual(SVGNode('{http://www.w3.org/2000/svg}rect', {}, {}, '').tag, 'rect')

    def test_fill_none_with_stroke(self):
        n = SVGNode('path', {}, {'fill': 'none', 'stroke': '#000', 'stroke-width': '2'}, '')
        self.assertFalse(n.has_fill)
        self.assertTrue(n.has_stroke)
        self.assertTrue(n.is_visible)

    def test_zero_fill_opacity_disables_fill(self):
        n = SVGNode('path', {}, {'fill': 'black', 'fill-opacity': '0'}, '')
        self.assertFalse(n.has_fill)
        self.assertFalse(n.is_visible)

    def test_zero_stroke_opacity_disables_stroke(self):
        n = SVGNode('path', {}, {'fill': 'none', 'stroke': '#000', 'stroke-opacity': '0'}, '')
        self.assertFalse(n.has_stroke)

    def test_global_opacity_scales_channels(self):
        n = SVGNode('path', {}, {'fill': 'black', 'opacity': '0.5', 'fill-opacity': '0.5'}, '')
        self.assertAlmostEqual(n.effective_fill_alpha, 0.25)

    def test_zero_stroke_width_disables_stroke(self):
        n = SVGNode('path', {}, {'fill': 'none', 'stroke': '#000', 'stroke-width': '0'}, '')
        self.assertFalse(n.has_stroke)

    def test_display_none_and_visibility_hidden_are_invisible(self):
        self.assertFalse(SVGNode('path', {}, {'display': 'none'}, '').is_visible)
        self.assertFalse(SVGNode('path', {}, {'visibility': 'hidden'}, '').is_visible)


class TestSVGLoader(unittest.TestCase):

    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            p.unlink(missing_ok=True)

    def load(self, content):
        path = write_svg(content)
        self._paths.append(path)
        return SVGLoader(str(path))

    def test_viewbox_metadata(self):
        loader = self.load('<svg viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg"></svg>')
        self.assertEqual(loader.viewbox, (0.0, 0.0, 200.0, 100.0))
        self.assertEqual((loader.width, loader.height), (200.0, 100.0))

    def test_missing_viewbox_defaults_from_width_height_with_warning(self):
        loader = self.load('<svg width="50px" height="20px" xmlns="http://www.w3.org/2000/svg"></svg>')
        self.assertEqual(loader.viewbox, (0.0, 0.0, 50.0, 20.0))
        self.assertTrue(any('viewBox attribute missing' in w for w in loader.warnings))

    def test_css_class_styles_are_applied(self):
        loader = self.load(
            '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">'
            '<style>.a { fill: none; stroke: #000; stroke-width: 3; }</style>'
            '<rect class="a" x="0" y="0" width="5" height="5"/></svg>'
        )
        (node, _), = loader.get_elements()
        self.assertFalse(node.has_fill)
        self.assertTrue(node.has_stroke)
        self.assertEqual(node.stroke_width, 3.0)

    def test_inline_style_overrides_presentation_attribute(self):
        loader = self.load(
            '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">'
            '<rect fill="red" style="fill:none;stroke:#000" width="5" height="5"/></svg>'
        )
        (node, _), = loader.get_elements()
        self.assertFalse(node.has_fill)

    def test_group_transforms_accumulate_into_stack(self):
        loader = self.load(
            '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">'
            '<g transform="translate(1,2)"><rect transform="scale(2)" width="5" height="5"/></g></svg>'
        )
        (node, stack), = loader.get_elements()
        self.assertEqual(stack, ['translate(1,2)', 'scale(2)'])
        self.assertEqual(node.tag, 'rect')

    def test_defs_and_hidden_elements_are_skipped(self):
        loader = self.load(
            '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">'
            '<defs><rect width="5" height="5"/></defs>'
            '<rect width="5" height="5" display="none"/>'
            '<text>hello</text></svg>'
        )
        self.assertEqual(loader.get_elements(), [])

    def test_clip_path_mask_and_fill_rule_warnings(self):
        loader = self.load(
            '<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">'
            '<clipPath id="c"><rect width="1" height="1"/></clipPath>'
            '<mask id="m"><rect width="1" height="1"/></mask>'
            '<path d="M0 0 L1 1" fill-rule="evenodd"/></svg>'
        )
        loader.get_elements()
        joined = ' '.join(loader.warnings)
        self.assertIn('clipPath', joined)
        self.assertIn('mask', joined)
        self.assertIn('evenodd', joined)


if __name__ == '__main__':
    unittest.main()
