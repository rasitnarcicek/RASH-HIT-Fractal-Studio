# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mehmet Raşit Narçiçek

"""Unit tests for artifact parsers and the cross-artifact validation report."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.artifact_validator import (
    parse_ascii_file,
    parse_mask_file,
    parse_png_samples_file,
    parse_rle_file,
    parse_svg_rects_xml,
    validate_and_generate_real_diff_reports,
)


class ArtifactTempDirTestCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        return path


class TestParseAsciiFile(ArtifactTempDirTestCase):

    def test_hash_and_x_characters_are_filled(self):
        path = self.write('a.txt', '|#X|\n|.. |\n')
        np.testing.assert_array_equal(parse_ascii_file(path, 2, 2), [[1, 1], [0, 0]])

    def test_rows_with_wrong_width_are_skipped(self):
        path = self.write('a.txt', '|111|\n|10|\nheader line\n')
        np.testing.assert_array_equal(parse_ascii_file(path, 2, 2), [[1, 0], [0, 0]])

    def test_extra_rows_beyond_grid_are_ignored(self):
        path = self.write('a.txt', '|10|\n|01|\n|11|\n')
        np.testing.assert_array_equal(parse_ascii_file(path, 2, 2), [[1, 0], [0, 1]])


class TestParseMaskFile(ArtifactTempDirTestCase):

    def test_pipe_delimited_bits_with_spaces(self):
        path = self.write('m.txt', '|1 0 1|\n|0 1 0|\n')
        np.testing.assert_array_equal(parse_mask_file(path, 3, 2), [[1, 0, 1], [0, 1, 0]])

    def test_y_prefixed_rows(self):
        path = self.write('m.txt', 'Y0=101\nY1=010\n')
        np.testing.assert_array_equal(parse_mask_file(path, 3, 2), [[1, 0, 1], [0, 1, 0]])

    def test_mismatched_width_rows_are_ignored(self):
        path = self.write('m.txt', 'Y0=1010\n|11|\n')
        np.testing.assert_array_equal(parse_mask_file(path, 3, 1), [[0, 0, 0]])


class TestParseRLEFile(ArtifactTempDirTestCase):

    def test_flat_rle_runs(self):
        path = self.write('r.json', json.dumps({'rle_runs': [[1, 2], [0, 2]]}))
        np.testing.assert_array_equal(parse_rle_file(path, 2, 2), [[1, 1], [0, 0]])

    def test_rle_runs_with_wrong_total_falls_back_to_rows(self):
        payload = {'rle_runs': [[1, 3]], 'rows': [{'y': 1, 'filled_runs': [[0, 1]]}]}
        path = self.write('r.json', json.dumps(payload))
        np.testing.assert_array_equal(parse_rle_file(path, 2, 2), [[0, 0], [1, 1]])

    def test_row_runs_outside_grid_are_ignored(self):
        payload = {'rows': [{'y': 5, 'filled_runs': [[0, 1]]}]}
        path = self.write('r.json', json.dumps(payload))
        np.testing.assert_array_equal(parse_rle_file(path, 2, 2), np.zeros((2, 2), dtype=int))


class TestParseSVGRectsXML(ArtifactTempDirTestCase):

    def test_filled_and_empty_rects_are_classified(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect x="0" y="0" width="10" height="10" fill="#111827"/>'
            '<rect x="10" y="0" width="10" height="10" fill="#ffffff"/>'
            '<rect x="10" y="10" width="10" height="10" fill="black"/>'
            '</svg>'
        )
        matrix, audit = parse_svg_rects_xml(self.write('m.svg', svg), 2, 2)
        np.testing.assert_array_equal(matrix, [[1, 0], [0, 1]])
        self.assertTrue(audit['has_root_svg'])
        self.assertEqual(audit['total_rects'], 3)
        self.assertEqual((audit['filled_rects_count'], audit['empty_rects_count']), (2, 1))
        self.assertFalse(audit['has_image_tag'])
        self.assertFalse(audit['has_base64'])

    def test_embedded_raster_image_is_flagged(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<image href="data:image/png;base64,AAAA"/></svg>'
        )
        _, audit = parse_svg_rects_xml(self.write('m.svg', svg), 1, 1)
        self.assertTrue(audit['has_image_tag'])
        self.assertTrue(audit['has_base64'])

    def test_non_numeric_rect_attributes_are_skipped(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect x="bad" y="0" width="10" height="10" fill="#111827"/></svg>'
        )
        matrix, audit = parse_svg_rects_xml(self.write('m.svg', svg), 1, 1)
        np.testing.assert_array_equal(matrix, [[0]])
        self.assertEqual(audit['filled_rects_count'], 0)

    def test_rects_outside_grid_bounds_are_ignored(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect x="0" y="0" width="10" height="10" fill="black"/>'
            '<rect x="90" y="0" width="10" height="10" fill="black"/></svg>'
        )
        matrix, _ = parse_svg_rects_xml(self.write('m.svg', svg), 2, 1)
        np.testing.assert_array_equal(matrix, [[1, 0]])


class TestParsePNGSamplesFile(ArtifactTempDirTestCase):

    def test_missing_file_returns_zero_matrix(self):
        matrix = parse_png_samples_file(self.tmp / 'missing.png', 3, 2)
        np.testing.assert_array_equal(matrix, np.zeros((2, 3), dtype=int))

    def test_dark_pixels_are_marked_filled(self):
        img = Image.new('RGB', (2, 2), (255, 255, 255))
        img.putpixel((0, 0), (0, 0, 0))
        path = self.tmp / 'p.png'
        img.save(path)
        np.testing.assert_array_equal(parse_png_samples_file(path, 2, 2), [[1, 0], [0, 0]])


class TestValidateAndGenerateRealDiffReports(ArtifactTempDirTestCase):

    def setUp(self):
        super().setUp()
        self._cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self._cwd)
        super().tearDown()

    def build_artifacts(self, out_dir: Path, name='sample'):
        (out_dir / name / 'data').mkdir(parents=True)
        (out_dir / name / 'ascii').mkdir(parents=True)
        (out_dir / name / 'figures').mkdir(parents=True)
        summary = {'levels': [{'level': 1, 'grid': '2x2', 'cols': 2, 'rows': 2}]}
        (out_dir / name / 'data' / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        (out_dir / name / 'ascii' / '01_2x2_ascii.txt').write_text('|10|\n|00|\n', encoding='utf-8')
        (out_dir / name / 'data' / '01_2x2_mask.txt').write_text('Y0=10\nY1=00\n', encoding='utf-8')
        (out_dir / name / 'data' / '01_2x2_rle.json').write_text(
            json.dumps({'rle_runs': [[1, 1], [0, 3]]}), encoding='utf-8')
        (out_dir / name / 'figures' / '01_2x2_map.svg').write_text(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<rect x="0" y="0" width="10" height="10" fill="black"/>'
            '<rect x="10" y="0" width="10" height="10" fill="white"/></svg>',
            encoding='utf-8')

    def test_missing_summary_raises(self):
        out_dir = self.tmp / 'out'
        (out_dir / 'sample').mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            validate_and_generate_real_diff_reports('sample', 1, out_dir)

    def test_unknown_level_raises(self):
        out_dir = self.tmp / 'out'
        self.build_artifacts(out_dir)
        with self.assertRaises(ValueError):
            validate_and_generate_real_diff_reports('sample', 9, out_dir)

    def test_consistent_artifacts_report_a_match(self):
        out_dir = self.tmp / 'out'
        self.build_artifacts(out_dir)
        report = validate_and_generate_real_diff_reports('sample', 1, out_dir)
        self.assertTrue(report['matched'])
        self.assertEqual(report['grid'], '2x2')
        self.assertEqual(report['ascii_filled'], 1)
        self.assertEqual(report['mask_filled'], 1)
        self.assertEqual(report['rle_filled'], 1)
        self.assertEqual(report['svg_filled'], 1)
        self.assertTrue(Path(report['diff_csv']).exists())
        self.assertTrue(Path(report['row_csv']).exists())

    def test_divergent_mask_is_reported_as_mismatch(self):
        out_dir = self.tmp / 'out'
        self.build_artifacts(out_dir)
        (out_dir / 'sample' / 'data' / '01_2x2_mask.txt').write_text('Y0=11\nY1=00\n', encoding='utf-8')
        report = validate_and_generate_real_diff_reports('sample', 1, out_dir)
        self.assertFalse(report['matched'])
        self.assertEqual(report['mask_filled'], 2)


if __name__ == '__main__':
    unittest.main()
