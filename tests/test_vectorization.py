import json
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from handwrite.pngtosvg import (
    DEFAULT_VECTORIZATION,
    PNGtoSVG,
    THRESHOLD,
    VectorizationBudgetError,
    _min_stroke_width,
    build_design_svg,
    count_subpaths,
    measure_features,
    threshold_mask,
)
from handwrite.svgtottf import DESIGN_EM, FontGeometryError, SVGtoTTF
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"


def rgba_canvas(size, bg=(255, 255, 255, 255)):
    return np.full((size, size, 4), bg, dtype=np.uint8)


def save_png(path, pixels):
    Image.fromarray(pixels, "RGBA").save(path)


def glyph_config(path, codepoints):
    config = {
        "threshold_value": 200,
        "vectorization": dict(DEFAULT_VECTORIZATION),
        "props": {
            "ascent": 800,
            "descent": 200,
            "em": 1000,
            "encoding": "UnicodeFull",
            "filename": "TestFont",
            "style": "Regular",
        },
        "sfnt_names": {},
        "glyphs": list(codepoints),
        "typography_parameters": {
            "bearing_table": {},
            "kerning_table": {"rows": [], "cols": []},
        },
    }
    with open(path, "w") as f:
        json.dump(config, f)
    return config


class TestThresholdSemantics(unittest.TestCase):
    def test_white_requires_r_g_a_but_not_b(self):
        # R/G/A pass while B is zero: still white because B is never read.
        img = Image.fromarray(np.array([[(255, 255, 0, 255)]], dtype=np.uint8), "RGBA")
        self.assertFalse(threshold_mask(img, THRESHOLD)[0, 0])

        # Blue alone cannot rescue a pixel whose R/G fail the threshold.
        img = Image.fromarray(np.array([[(0, 0, 255, 255)]], dtype=np.uint8), "RGBA")
        self.assertTrue(threshold_mask(img, THRESHOLD)[0, 0])

    def test_alpha_participates(self):
        img = Image.fromarray(np.array([[(0, 0, 0, 199)]], dtype=np.uint8), "RGBA")
        # R/G below the threshold, so the pixel is ink regardless of alpha.
        self.assertTrue(threshold_mask(img, THRESHOLD)[0, 0])

        img = Image.fromarray(np.array([[(200, 200, 0, 199)]], dtype=np.uint8), "RGBA")
        # R/G pass but alpha does not: ink.
        self.assertTrue(threshold_mask(img, THRESHOLD)[0, 0])

        img = Image.fromarray(np.array([[(200, 200, 0, 200)]], dtype=np.uint8), "RGBA")
        self.assertFalse(threshold_mask(img, THRESHOLD)[0, 0])


class TestFeatureMeasurement(unittest.TestCase):
    def test_isolated_dots_components_and_areas(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[2:3, 2:3] = True  # 1 px dot
        mask[10:14, 10:14] = True  # 16 px dot
        features = measure_features(mask)
        self.assertEqual(features["components"], 2)
        self.assertEqual(features["holes"], 0)
        self.assertEqual(features["min_component_area"], 1)
        self.assertEqual(features["ink_pixels"], 17)

    def test_enclosed_hole_count(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[5:35, 5:35] = True
        mask[10:30, 10:30] = False
        features = measure_features(mask)
        self.assertEqual(features["components"], 1)
        self.assertEqual(features["holes"], 1)

    def test_corner_touch_is_one_contour(self):
        # Two blocks touching only at a corner: potrace traces one contour.
        mask = np.zeros((20, 20), dtype=bool)
        mask[3:10, 3:10] = True
        mask[10:17, 10:17] = True
        features = measure_features(mask)
        self.assertEqual(features["components"], 1)

    def test_min_stroke_width_detects_hairlines(self):
        line = np.zeros((60, 200), dtype=bool)
        line[30, 10:190] = True
        self.assertEqual(_min_stroke_width(line), 1)

        two_px = np.zeros((60, 200), dtype=bool)
        two_px[30:32, 10:190] = True
        self.assertEqual(_min_stroke_width(two_px), 2)

        dot = np.zeros((40, 40), dtype=bool)
        dot[18:22, 18:22] = True
        self.assertEqual(_min_stroke_width(dot), 4)


class TestAdaptiveSampling(unittest.TestCase):
    def setUp(self):
        self.converter = PNGtoSVG()
        self.temp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp)

    def _png(self, name, pixels):
        path = os.path.join(self.temp, name)
        save_png(path, pixels)
        return path

    def test_hairline_is_integer_magnified_but_same_design_size(self):
        # 1 px line at 100 px canvas vs the identical artwork pre-replicated
        # to 200 px: both must end up with the exact same design-space bbox.
        lo = rgba_canvas(100)
        lo[50, 10:90] = (0, 0, 0, 255)
        hi = np.repeat(np.repeat(lo, 2, axis=0), 2, axis=1)

        def design_bbox(pixels):
            path = self._png("g_%d.png" % pixels.shape[0], pixels)
            bmp, info = self.converter.pngToBmp(path, dict(DEFAULT_VECTORIZATION))
            svg = self.converter.bmpToSvg(bmp, info, dict(DEFAULT_VECTORIZATION))
            root = ET.parse(svg).getroot()
            meta = json.loads(root.find("{%s}metadata" % SVG_NS).text)
            return meta

        meta_lo = design_bbox(lo)
        meta_hi = design_bbox(hi)
        self.assertEqual(meta_lo["sample_factor"], 2)
        self.assertEqual(meta_hi["sample_factor"], 1)
        self.assertEqual(meta_lo["design"]["ink_bbox"], meta_hi["design"]["ink_bbox"])

    def test_proportional_artwork_is_resolution_independent(self):
        def bar(n):
            canvas = rgba_canvas(n)
            thickness = max(2, int(round(n * 0.02)))
            y = (n - thickness) // 2
            canvas[y : y + thickness, int(n * 0.1) : int(n * 0.9)] = (
                0,
                0,
                0,
                255,
            )
            return canvas

        bboxes = []
        for n in (100, 400):
            path = self._png("bar%d.png" % n, bar(n))
            bmp, info = self.converter.pngToBmp(path, dict(DEFAULT_VECTORIZATION))
            svg = self.converter.bmpToSvg(bmp, info, dict(DEFAULT_VECTORIZATION))
            root = ET.parse(svg).getroot()
            meta = json.loads(root.find("{%s}metadata" % SVG_NS).text)
            bboxes.append(meta["design"]["ink_bbox"])
        self.assertEqual(bboxes[0], bboxes[1])

    def test_budget_rejects_when_magnification_does_not_fit(self):
        canvas = rgba_canvas(200)
        canvas[100, 20:180] = (0, 0, 0, 255)
        path = self._png("hair.png", canvas)
        budget = dict(DEFAULT_VECTORIZATION)
        # 200 * 2 exceeds the cap, so faithful 2x tracing cannot be done.
        budget.update(max_dimension=300, max_pixels=90000)
        with self.assertRaises(VectorizationBudgetError):
            self.converter.pngToBmp(path, budget)

    def test_budget_rejects_downsample_that_loses_hole(self):
        canvas = rgba_canvas(400)
        canvas[50:350, 50:350] = (0, 0, 0, 255)
        canvas[199:201, 199:201] = (255, 255, 255, 255)  # 2 px hole
        path = self._png("hole.png", canvas)
        budget = dict(DEFAULT_VECTORIZATION)
        budget.update(max_dimension=100, max_pixels=10000)
        with self.assertRaises(VectorizationBudgetError):
            self.converter.pngToBmp(path, budget)

    def test_oversized_but_preservable_glyph_is_downsampled(self):
        canvas = rgba_canvas(2000)
        canvas[400:1600, 900:1100] = (0, 0, 0, 255)
        path = self._png("big.png", canvas)
        budget = dict(DEFAULT_VECTORIZATION)
        budget.update(max_dimension=500, max_pixels=250000)
        bmp, info = self.converter.pngToBmp(path, budget)
        self.assertLessEqual(info["width"], 500)
        self.assertEqual(info["features"]["components"], 1)
        self.assertGreaterEqual(info["features"]["min_stroke_width"], 2)

    def test_single_pixel_dot_survives_tracing(self):
        canvas = rgba_canvas(60)
        canvas[30, 30] = (0, 0, 0, 255)  # isolated 1 px dot
        canvas[10:50, 45:48] = (0, 0, 0, 255)  # plus a stroke
        path = self._png("dot.png", canvas)
        bmp, info = self.converter.pngToBmp(path, dict(DEFAULT_VECTORIZATION))
        svg = self.converter.bmpToSvg(bmp, info, dict(DEFAULT_VECTORIZATION))
        root = ET.parse(svg).getroot()
        paths = list(root.iter("{%s}path" % SVG_NS))
        # turdsize must be small enough that the 1 px dot stays a contour.
        self.assertEqual(count_subpaths(paths), 2)

    def test_svg_topology_mismatch_is_rejected(self):
        # Hand potrace a solid block but claim it has an extra component.
        canvas = rgba_canvas(40)
        canvas[10:30, 10:30] = (0, 0, 0, 255)
        path = self._png("solid.png", canvas)
        bmp, info = self.converter.pngToBmp(path, dict(DEFAULT_VECTORIZATION))
        tmp_svg = os.path.join(self.temp, "raw.svg")
        import subprocess

        subprocess.run(
            ["potrace", bmp, "-b", "svg", "-o", tmp_svg, "-t", "0"],
            check=True,
        )
        out_svg = os.path.join(self.temp, "solid.svg")
        info["features"]["components"] = 2
        with self.assertRaises(VectorizationBudgetError):
            build_design_svg(tmp_svg, out_svg, info)


class TestFinalFontValidation(unittest.TestCase):
    def setUp(self):
        self.converter = SVGtoTTF()
        self.temp = tempfile.mkdtemp()
        self.chars = os.path.join(self.temp, "characters")
        self.out = os.path.join(self.temp, "out")
        os.makedirs(self.chars)
        os.makedirs(self.out)
        self.codepoints = [65, 66, 67]
        self.config = os.path.join(self.temp, "config.json")
        glyph_config(self.config, self.codepoints)

        # A: isolated dot, B: true 1 px hairline, C: ring with one hole.
        glyphs = {}
        dot = rgba_canvas(200)
        dot[98:102, 100:104] = (0, 0, 0, 255)
        glyphs[65] = dot
        line = rgba_canvas(200)
        line[95, 20:180] = (0, 0, 0, 255)
        glyphs[66] = line
        ring = rgba_canvas(200)
        cv2.circle(ring, (100, 100), 70, (0, 0, 0, 255), thickness=5)
        glyphs[67] = ring
        for cp, pixels in glyphs.items():
            d = os.path.join(self.chars, str(cp))
            os.makedirs(d)
            save_png(os.path.join(d, "%d.png" % cp), pixels)

        PNGtoSVG().convert(self.chars, self.config)

    def tearDown(self):
        shutil.rmtree(self.temp)

    def test_dot_hairline_hole_survive_into_ttf(self):
        ttf = self.converter.convert(
            self.chars,
            self.out,
            self.config,
            {"filename": "TestFont", "family": "TestFont", "style": "Regular"},
        )
        self.assertTrue(os.path.exists(ttf))

    def test_rendered_topology_is_checked(self):
        ttf = self.converter.convert(
            self.chars,
            self.out,
            self.config,
            {"filename": "TestFont", "family": "TestFont", "style": "Regular"},
        )
        # Tamper with the dot's BMP after the font was built: validation must
        # notice the source/render topology mismatch instead of accepting it.
        dot_bmp = os.path.join(self.chars, "65", "65.bmp")
        frame = np.full((200, 200), 255, dtype=np.uint8)
        frame[50:150, 50:150] = 0
        frame[70:130, 70:130] = 255  # introduces a hole
        Image.fromarray(frame, "L").save(dot_bmp)
        with self.assertRaises(FontGeometryError):
            self.converter.validate_generated_font(ttf, self.chars, self.config)

    def test_validation_can_be_disabled(self):
        cfg = dict(
            glyph_config(os.path.join(self.temp, "unused.json"), self.codepoints)
        )
        cfg["vectorization"]["validate_font"] = False
        off_path = os.path.join(self.temp, "config_off.json")
        with open(off_path, "w") as f:
            json.dump(cfg, f)
        # Would raise on the tampered BMP if validation ran.
        dot_bmp = os.path.join(self.chars, "65", "65.bmp")
        frame = np.full((200, 200), 255, dtype=np.uint8)
        frame[50:150, 50:150] = 0
        Image.fromarray(frame, "L").save(dot_bmp)
        ttf = self.converter.convert(
            self.chars,
            self.out,
            off_path,
            {"filename": "NoValidate", "family": "NoValidate", "style": "Regular"},
        )
        self.assertTrue(os.path.exists(ttf))


class TestValidationHelpers(unittest.TestCase):
    def test_fit_canvas_center_pad_and_crop(self):
        small = np.ones((2, 2), dtype=bool)
        fitted = SVGtoTTF._fit_canvas(small, 4, 4, np)
        self.assertEqual(fitted.shape, (4, 4))
        self.assertEqual(fitted.sum(), 4)
        self.assertTrue(fitted[1:3, 1:3].all())

        big = np.ones((6, 6), dtype=bool)
        fitted = SVGtoTTF._fit_canvas(big, 4, 4, np)
        self.assertEqual(fitted.shape, (4, 4))
        self.assertTrue(fitted.all())

    def test_band_error_ignores_close_mismatch(self):
        expected = np.zeros((20, 20), dtype=bool)
        expected[5:15, 5:15] = True
        # Shifted by one pixel: entirely inside a generous band.
        actual = np.zeros((20, 20), dtype=bool)
        actual[5:15, 6:16] = True
        error = SVGtoTTF._band_error(actual, expected, 3, np, cv2)
        self.assertEqual(error, 0.0)


if __name__ == "__main__":
    unittest.main()
