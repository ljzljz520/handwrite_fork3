import sys
import os
import glob
import json
import uuid

import xml.etree.ElementTree as ET

# Fixed design space shared with pngtosvg (duplicated on purpose: this file is
# also executed by FontForge's own Python, which must stay stdlib-only).
SVG_NS = "http://www.w3.org/2000/svg"
DESIGN_EM = 1000


class FontGeometryError(Exception):
    """Raised when the generated TTF fails topology/geometry validation.

    The renderer must reproduce every independent dot, thin stroke and
    enclosed hole that the source bitmap contained; a glyph that does not
    survive integerization is reported instead of silently shipped.
    """


class SVGtoTTF:
    def convert(self, directory, outdir, config, metadata=None):
        """Convert a directory with SVG images to TrueType Font.

        Calls a subprocess to the run this script with Fontforge Python
        environment.

        Parameters
        ----------
        directory : str
            Path to directory with SVGs to be converted.
        outdir : str
            Path to output directory.
        config : str
            Path to config file.
        metadata : dict
            Dictionary containing the metadata (filename, family or style)
        """
        import subprocess
        import platform

        existing = set(glob.glob(os.path.join(outdir, "*.ttf")))
        subprocess.run(
            (
                ["ffpython"]
                if platform.system() == "Windows"
                else ["fontforge", "-script"]
            )
            + [
                os.path.abspath(__file__),
                config,
                directory,
                outdir,
                json.dumps(metadata),
            ],
            check=True,
        )

        produced = set(glob.glob(os.path.join(outdir, "*.ttf"))) - existing
        ttf_path = max(produced, key=os.path.getmtime, default=None)
        self.validate_generated_font(ttf_path, directory, config)
        return ttf_path

    @staticmethod
    def _read_svg_metadata(svg_path):
        """Return the embedded vectorization metadata, if present."""
        try:
            root = ET.parse(svg_path).getroot()
            node = root.find("{%s}metadata" % SVG_NS)
            if node is None or not node.text:
                return None
            data = json.loads(node.text)
            return data if data.get("handwrite_vectorization") == 1 else None
        except OSError, ET.ParseError, ValueError:
            return None

    @staticmethod
    def _topology(mask, np, cv2):
        """Count (ink components, enclosed holes) of a boolean mask.

        Uses 8-connectivity for both colors to mirror potrace's corner
        model: two regions touching at a pixel corner form one contour.
        """
        ink = (mask.astype(np.uint8)) * 255
        count, _, _, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        background = ((~mask).astype(np.uint8)) * 255
        bg_count, bg_labels, _, _ = cv2.connectedComponentsWithStats(
            background, connectivity=8
        )
        border = set(
            np.concatenate(
                (
                    bg_labels[0, :],
                    bg_labels[-1, :],
                    bg_labels[:, 0],
                    bg_labels[:, -1],
                )
            ).tolist()
        )
        holes = sum(1 for label in range(1, bg_count) if label not in border)
        return count - 1, holes

    @staticmethod
    def _fit_canvas(mask, height, width, np):
        """Center-crop/pad a boolean mask to exactly (height, width)."""
        out = np.zeros((height, width), dtype=bool)
        src_h, src_w = mask.shape
        h = min(src_h, height)
        w = min(src_w, width)
        sy = max(0, (src_h - h) // 2)
        sx = max(0, (src_w - w) // 2)
        dy = max(0, (height - h) // 2)
        dx = max(0, (width - w) // 2)
        out[dy : dy + h, dx : dx + w] = mask[sy : sy + h, sx : sx + w]
        return out

    @staticmethod
    def _band_error(actual, expected, distance, np, cv2):
        """Fraction of ink pixels mismatching farther than ``distance`` units."""
        union = np.logical_or(actual, expected)
        if not union.any():
            return 0.0
        dist_actual = cv2.distanceTransform(
            ((~actual).astype(np.uint8)) * 255, cv2.DIST_L2, 3
        )
        dist_expected = cv2.distanceTransform(
            ((~expected).astype(np.uint8)) * 255, cv2.DIST_L2, 3
        )
        wrong = ((actual & ~expected) & (dist_expected > distance)).sum()
        wrong += ((expected & ~actual) & (dist_actual > distance)).sum()
        return float(wrong) / float(union.sum())

    def validate_generated_font(self, ttf_path, directory, config):
        """Validate topology and geometry of every glyph in the final TTF.

        Renders each codepoint at 1 unit == 1 px and compares it against the
        bilevel source bitmap mapped through the explicit design transform.
        Independent dots (component count), enclosed holes (hole count), thin
        stroke widths and the geometric mismatch (distance-band filtered
        symmetric difference) are checked. A failure raises
        FontGeometryError so a degraded font is never accepted silently.
        Glyphs whose SVG carries no handwrite metadata (manual workflows) are
        skipped.
        """
        from handwrite.pngtosvg import DEFAULT_VECTORIZATION, _min_stroke_width

        if ttf_path is None:
            return
        with open(config) as f:
            cfg = json.load(f)
        budget = dict(DEFAULT_VECTORIZATION)
        budget.update(cfg.get("vectorization", {}))
        if not budget.get("validate_font", True):
            return

        import cv2
        import numpy as np
        from PIL import Image, ImageFont

        font = ImageFont.truetype(ttf_path, DESIGN_EM)
        failures = []
        for codepoint in cfg.get("glyphs", []):
            glyph_dir = os.path.join(directory, str(codepoint))
            meta = self._read_svg_metadata(
                os.path.join(glyph_dir, "{}.svg".format(codepoint))
            )
            if meta is None or meta["features"]["components"] == 0:
                continue
            try:
                self._validate_glyph(
                    codepoint,
                    meta,
                    glyph_dir,
                    font,
                    budget,
                    np,
                    cv2,
                    Image,
                    _min_stroke_width,
                )
            except FontGeometryError as exc:
                failures.append(str(exc))

        if failures:
            raise FontGeometryError(
                "Generated font failed topology/geometry validation:\n - "
                + "\n - ".join(failures[:20])
                + ("" if len(failures) <= 20 else "\n - ... %d more" % len(failures))
            )

    def _validate_glyph(
        self,
        codepoint,
        meta,
        glyph_dir,
        font,
        budget,
        np,
        cv2,
        Image,
        min_stroke_width,
    ):
        bmp_path = os.path.join(glyph_dir, "{}.bmp".format(codepoint))
        if not os.path.exists(bmp_path):
            return

        width = meta["sampled_width"]
        height = meta["sampled_height"]
        tr = meta["transform"]
        scale = tr["units_per_pixel"]
        xoff, yoff = tr["xoff"], tr["yoff"]

        source = np.asarray(Image.open(bmp_path).convert("L")) < 128

        # Expected ink in the fixed y-down 1000-unit design canvas.
        expected = np.zeros((DESIGN_EM, DESIGN_EM), dtype=bool)
        piece = cv2.resize(
            source.astype(np.uint8),
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        px, py = int(round(xoff)), int(round(yoff))
        ph, pw = piece.shape
        expected[py : py + ph, px : px + pw] = piece

        mask_core = font.getmask(chr(codepoint), mode="L")
        rendered = np.asarray(mask_core).reshape(mask_core.size[1], mask_core.size[0])
        actual_full = rendered >= 128
        if not actual_full.any():
            raise FontGeometryError("glyph renders no ink")
        ys, xs = np.where(actual_full)
        actual = actual_full[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

        bx0, by0, bx1, by1 = meta["design"]["ink_bbox"]
        ox, oy = int(round(bx0)), int(round(by0))
        ew, eh = max(1, int(round(bx1 - bx0))), max(1, int(round(by1 - by0)))
        ah, aw = actual.shape

        # The declared design bbox is the bitmap ink extent mapped through
        # the explicit transform. potrace's corner model bevels at most one
        # sampled pixel off each tip; rasterization/thresholding adds a
        # couple of font units of anti-aliasing slack.
        size_tol = max(0.07 * ew, 2 * scale + 4), max(0.07 * eh, 2 * scale + 4)
        if abs(aw - ew) > size_tol[0] or abs(ah - eh) > size_tol[1]:
            raise FontGeometryError(
                "U+{:04X} glyph size changed: rendered {}x{} units vs "
                "expected {}x{} (design transform mismatch)".format(
                    codepoint, aw, ah, ew, eh
                )
            )

        # Compare at 1:1 in design units (no non-uniform resampling, which
        # would distort stroke widths); small position/border rounding
        # differences are absorbed by centering.
        actual_norm = self._fit_canvas(actual, eh, ew, np)
        expected_crop = self._fit_canvas(
            expected[oy : oy + eh, ox : ox + ew], eh, ew, np
        )

        a_components, a_holes = self._topology(actual_norm, np, cv2)
        e_components = meta["features"]["components"]
        e_holes = meta["features"]["holes"]
        if (a_components, a_holes) != (e_components, e_holes):
            raise FontGeometryError(
                "U+{:04X} topology changed: rendered {} component(s)/{} "
                "hole(s) vs source {} component(s)/{} hole(s) (dots/holes "
                "lost or strokes merged)".format(
                    codepoint, a_components, a_holes, e_components, e_holes
                )
            )

        error = self._band_error(
            actual_norm, expected_crop, budget["tolerance_band_units"], np, cv2
        )
        if error > budget["max_area_error"]:
            raise FontGeometryError(
                "U+{:04X} geometric error {:.1%} exceeds budget {:.1%} "
                "outside the {}-unit tolerance band".format(
                    codepoint,
                    error,
                    budget["max_area_error"],
                    budget["tolerance_band_units"],
                )
            )

        expected_width = min_stroke_width(expected_crop) or 0
        actual_width = min_stroke_width(actual_norm) or 0
        floor = budget["min_stroke_design_units"]
        # A 2-unit integerization slack plus a relative retention floor:
        # catches a long stroke collapsing to a hairline without rejecting
        # ordinary anti-aliasing of a minimum-width (4-unit) stroke.
        required = max(min(expected_width, floor) - 2, int(0.6 * expected_width))
        if actual_width < required:
            raise FontGeometryError(
                "U+{:04X} thinnest stroke collapsed to {} units (source {} "
                "units, required {}, floor {})".format(
                    codepoint, actual_width, expected_width, required, floor
                )
            )

    def set_properties(self):
        """Set metadata of the font from config."""
        props = self.config["props"]
        lang = props.get("lang", "English (US)")
        fontname = self.metadata.get("filename", None) or props.get(
            "filename", "Example"
        )
        family = self.metadata.get("family", None) or fontname
        style = self.metadata.get("style", None) or props.get("style", "Regular")

        self.font.familyname = fontname
        self.font.fontname = fontname + "-" + style
        self.font.fullname = fontname + " " + style
        self.font.encoding = props.get("encoding", "UnicodeFull")

        for k, v in props.items():
            if hasattr(self.font, k):
                if isinstance(v, list):
                    v = tuple(v)
                setattr(self.font, k, v)

        if self.config.get("sfnt_names", None):
            self.config["sfnt_names"]["Family"] = family
            self.config["sfnt_names"]["Fullname"] = family + " " + style
            self.config["sfnt_names"]["PostScriptName"] = family + "-" + style
            self.config["sfnt_names"]["SubFamily"] = style

        self.config["sfnt_names"]["UniqueID"] = family + " " + str(uuid.uuid4())

        for k, v in self.config.get("sfnt_names", {}).items():
            self.font.appendSFNTName(str(lang), str(k), str(v))

    def add_glyphs(self, directory):
        """Read and add SVG images as glyphs to the font.

        Walks through the provided directory and uses each ord(character).svg file
        as glyph for the character. Then using the provided config, set the font
        parameters and export TTF file to outdir.

        Parameters
        ----------
        directory : str
            Path to directory with SVGs to be converted.
        """
        space = self.font.createMappedChar(ord(" "))
        space.width = 500

        for k in self.config["glyphs"]:
            # Create character glyph
            g = self.font.createMappedChar(k)
            self.unicode_mapping.setdefault(k, g.glyphname)
            # Get outlines
            src = "{}/{}.svg".format(k, k)
            src = directory + os.sep + src
            g.importOutlines(src, ("removeoverlap", "correctdir"))
            g.removeOverlap()

    def set_bearings(self, bearings):
        """Add left and right bearing from config

        Parameters
        ----------
        bearings : dict
            Map from character: [left bearing, right bearing]
        """
        default = bearings.get("Default", [60, 60])

        for k, v in bearings.items():
            if v[0] is None:
                v[0] = default[0]
            if v[1] is None:
                v[1] = default[1]

            if k != "Default":
                glyph_name = self.unicode_mapping[ord(str(k))]
                self.font[glyph_name].left_side_bearing = v[0]
                self.font[glyph_name].right_side_bearing = v[1]

    def set_kerning(self, table):
        """Set kerning values in the font.

        Parameters
        ----------
        table : dict
            Config dictionary with kerning values/autokern bool.
        """
        rows = table["rows"]
        rows = [list(i) if i != None else None for i in rows]
        cols = table["cols"]
        cols = [list(i) if i != None else None for i in cols]

        self.font.addLookup("kern", "gpos_pair", 0, [["kern", [["latn", ["dflt"]]]]])

        if table.get("autokern", True):
            self.font.addKerningClass(
                "kern", "kern-1", table.get("seperation", 0), rows, cols, True
            )
        else:
            kerning_table = table.get("table", False)
            if not kerning_table:
                raise ValueError("Kerning offsets not found in the config file.")
            flatten_list = lambda y: (
                [x for a in y for x in flatten_list(a)] if type(y) is list else [y]
            )
            offsets = [0 if x is None else x for x in flatten_list(kerning_table)]
            self.font.addKerningClass("kern", "kern-1", rows, cols, offsets)

    def generate_font_file(self, filename, outdir, config_file):
        """Output TTF file.

        Additionally checks for multiple outputs and duplicates.

        Parameters
        ----------
        filename : str
            Output filename.
        outdir : str
            Path to output directory.
        config_file : str
            Path to config file.
        """
        if filename is None:
            raise NameError("filename not found in config file.")

        outfile = str(
            outdir
            + os.sep
            + (filename + ".ttf" if not filename.endswith(".ttf") else filename)
        )

        while os.path.exists(outfile):
            outfile = os.path.splitext(outfile)[0] + " (1).ttf"

        sys.stderr.write("\nGenerating %s...\n" % outfile)
        self.font.generate(outfile)

    def convert_main(self, config_file, directory, outdir, metadata):
        try:
            self.font = fontforge.font()
        except:
            import fontforge

        with open(config_file) as f:
            self.config = json.load(f)
        self.metadata = json.loads(metadata) or {}

        self.font = fontforge.font()
        self.unicode_mapping = {}
        self.set_properties()
        self.add_glyphs(directory)

        # bearing table
        self.set_bearings(self.config["typography_parameters"].get("bearing_table", {}))

        # kerning table
        self.set_kerning(self.config["typography_parameters"].get("kerning_table", {}))

        # Generate font and save as a .ttf file
        filename = self.metadata.get("filename", None) or self.config["props"].get(
            "filename", None
        )
        self.generate_font_file(str(filename), outdir, config_file)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise ValueError("Incorrect call to SVGtoTTF")
    SVGtoTTF().convert_main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
