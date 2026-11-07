import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ImageChops

SVG_NS = "http://www.w3.org/2000/svg"

# Bilevel classification threshold. The predicate is kept byte-for-byte
# compatible with the historical pipeline: a pixel is considered white iff
# its R, G and A channels are all >= THRESHOLD (the B channel is intentionally
# never consulted). Nothing in the sampling/tracing stage may change it.
THRESHOLD = 200

# Fixed design space shared with svgtottf: 1000 font units per em. The SVG
# stays y-down; FontForge flips the y-axis around ascent=800 when importing.
DESIGN_EM = 1000

# Potrace quantizes its SVG path coordinates to 1/internal-unit pixels, i.e.
# raw path coordinates are in 1/10 px, y-up, origin at the bottom-left.
POTRACE_INTERNAL_UNIT = 10

# potrace 1.16 geometrically corrupts true 1-pixel-wide strokes (it bends
# their long edges and erodes the caps). Strokes of >= 2 traced pixels are
# reproduced exactly, so the sampling stage integer-magnifies bilevel crops
# until this floor is reached; the explicit design transform maps the larger
# trace bitmap back to the same glyph size.
MIN_TRACE_STROKE_PX = 2

# Resource/quality budget for the stage after cropping. Every entry is
# optional in the config file's "vectorization" section; missing entries fall
# back to these defaults, so existing configs keep working unchanged.
DEFAULT_VECTORIZATION = {
    # Hard resource caps for the sampled bitmap.
    "max_dimension": 1000,
    "max_pixels": 1_000_000,
    # Thinnest stroke (font units) that may survive into the final font.
    "min_stroke_design_units": 4,
    # Potrace curve optimization tolerance, in sampled pixels.
    "opttolerance": 0.2,
    # Final-font geometry validation tolerances (used by svgtottf).
    "max_area_error": 0.03,
    "tolerance_band_units": 6,
    "validate_font": True,
}


class PotraceNotFound(Exception):
    pass


class VectorizationBudgetError(Exception):
    """Raised when the resource budget cannot preserve the glyph features.

    The pipeline refuses to emit a degraded result instead of silently
    dropping dots, closing holes or breaking thin strokes.
    """


def threshold_mask(img_rgba, threshold=THRESHOLD):
    """Classify an RGBA image into a boolean ink mask using the fixed rule.

    Parameters
    ----------
    img_rgba : PIL.Image.Image
        Image in RGBA mode.
    threshold : int
        White threshold, identical semantics to the historical converter.

    Returns
    -------
    numpy.ndarray
        2-D boolean array, True where the pixel is black ink.
    """
    a = np.asarray(img_rgba)
    white = (
        (a[..., 0] >= threshold) & (a[..., 1] >= threshold) & (a[..., 3] >= threshold)
    )
    return ~white


def _run_lengths(mask, axis):
    """Length (in pixels) of the contiguous True run containing each pixel.

    Fully vectorized: each True run along ``axis`` gets an id from the
    False->True edges, run sizes from a weighted bincount, then the size is
    broadcast back onto the run's pixels.
    """
    out = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return out
    moved = np.moveaxis(mask, axis, 0)
    flat = moved.astype(bool)
    starts = flat & ~np.roll(flat, 1, axis=0)
    starts[0] = flat[0]
    run_ids = np.cumsum(starts, axis=0)
    # Runs on different lines otherwise share ids 1..n; offset per line so
    # every run has a globally unique id for the bincount size lookup.
    line_count = moved.shape[1]
    run_ids += np.arange(line_count)[None, :] * (moved.shape[0] + 1)
    ids = run_ids[flat]
    sizes = np.bincount(ids)
    lengths = sizes[ids]
    out_moved = np.zeros(moved.shape, dtype=np.int32)
    out_moved[flat] = lengths
    return np.moveaxis(out_moved, 0, axis)


def _min_stroke_width(mask):
    """Smallest steady stroke thickness, in pixels, along the medial axis.

    The medial axis ("ridge") is the set of ink pixels with no strictly
    greater 8-neighbour in the distance transform. Unlike thinning
    algorithms, this ridge never disappears for a compact dot (thinning
    would erode 2x2 cores completely) while still running down the spine of
    every stroke. Thickness at ridge pixels is the smaller of the
    horizontal/vertical ink run, exact for bilevel strokes including true
    1-pixel lines. The 20th percentile - rather than the raw minimum - is
    used because anti-aliased rasterizers turn square caps into long
    pointy nubs; a genuinely thin spine still dominates the distribution.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return None
    ink = mask.astype(np.uint8) * 255
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    padded = np.pad(dist, 1)
    greater = np.zeros(mask.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            greater |= (
                padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
                > dist
            )
    ridge = mask & ~greater
    if not ridge.any():
        return None
    horizontal = _run_lengths(mask, axis=1)
    vertical = _run_lengths(mask, axis=0)
    widths = np.minimum(horizontal[ridge], vertical[ridge])
    return int(np.percentile(widths.astype(np.float64), 20, method="lower"))


def measure_features(mask):
    """Measure topology-relevant features of a black/white glyph mask.

    Counts independent ink components (isolated dots), enclosed holes and the
    thinnest stroke. Mirrors potrace's corner-connectivity model: regions of
    either color that touch only at a pixel corner merge into one contour, so
    both ink components and background holes use 8-connectivity.

    Parameters
    ----------
    mask : numpy.ndarray
        2-D boolean ink mask.

    Returns
    -------
    dict
        Geometry/topology descriptor of the bitmap.
    """
    mask = mask.astype(bool)
    height, width = mask.shape
    ink = (mask.astype(np.uint8)) * 255

    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = sorted(int(a) for a in stats[1:, cv2.CC_STAT_AREA])

    # Holes are background components that never touch the image border.
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

    bbox = None
    if areas:
        ys, xs = np.where(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    return {
        "width": int(width),
        "height": int(height),
        "components": count - 1,
        "holes": holes,
        "areas": areas,
        "min_component_area": areas[0] if areas else None,
        "ink_pixels": int(mask.sum()),
        "min_stroke_width": _min_stroke_width(mask),
        "bbox": bbox,
    }


def _sample_preserves(original, candidate, ratio, budget):
    """Whether a downsampled candidate still carries every source feature."""
    if candidate["components"] != original["components"]:
        return False
    if candidate["holes"] != original["holes"]:
        return False
    expected_ink = original["ink_pixels"] * ratio * ratio
    if expected_ink and candidate["ink_pixels"] < 0.5 * expected_ink:
        return False
    # potrace needs >= 2 px strokes and the fixed design space needs the
    # configured minimum number of font units.
    if (candidate["min_stroke_width"] or 0) < MIN_TRACE_STROKE_PX:
        return False
    design_scale = DESIGN_EM / max(candidate["width"], candidate["height"])
    if (candidate["min_stroke_width"] or 0) * design_scale < budget[
        "min_stroke_design_units"
    ]:
        return False
    return True


def _nearest_upscale(mask, factor):
    """Exact integer replication of a bilevel mask (no invented detail)."""
    if factor == 1:
        return mask
    return np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)


def choose_sample(img, budget):
    """Pick the sampling/tracing resolution for one glyph.

    The native crop resolution is the information ceiling and is never
    downscaled while it fits the resource budget. When the bitmap contains
    sub-2-pixel strokes (which potrace traces incorrectly), the bilevel crop
    is integer-replicated for tracing; the explicit design transform later
    maps it back to an unchanged glyph size. If the crop exceeds the budget,
    the largest budget-allowed resolution preserving every component, hole
    and thin stroke is searched. In every impossible case
    VectorizationBudgetError is raised instead of emitting a degraded glyph.

    Parameters
    ----------
    img : PIL.Image.Image
        RGBA image of one already cropped character.
    budget : dict
        Merged vectorization budget.

    Returns
    -------
    (numpy.ndarray, dict, int)
        Ink mask of the bitmap handed to potrace, its measured features and
        the integer replication factor relative to the native crop.
    """
    native_mask = threshold_mask(img)
    native = measure_features(native_mask)
    width, height = img.size

    if native["components"] == 0:
        return native_mask, native, 1

    # Integer replication needed to lift the thinnest stroke over potrace's
    # faithful-tracing floor.
    thinnest = native["min_stroke_width"] or MIN_TRACE_STROKE_PX
    factor = max(1, int(math.ceil(MIN_TRACE_STROKE_PX / float(thinnest))))

    def fits(f):
        return (
            width * f <= budget["max_dimension"]
            and height * f <= budget["max_dimension"]
            and width * height * f * f <= budget["max_pixels"]
        )

    if factor > 1:
        if fits(factor):
            magnified = _nearest_upscale(native_mask, factor)
            return magnified, measure_features(magnified), factor
        if fits(1) and thinnest < MIN_TRACE_STROKE_PX:
            raise VectorizationBudgetError(
                "Glyph contains a {}-px stroke but {}-fold sampling needed "
                "for faithful tracing does not fit the resource budget "
                "(max {}px / {} pixels). Refusing to trace it at degraded "
                "precision.".format(
                    thinnest,
                    factor,
                    budget["max_dimension"],
                    budget["max_pixels"],
                )
            )

    cap = min(
        budget["max_dimension"] / width,
        budget["max_dimension"] / height,
        math.sqrt(budget["max_pixels"] / float(width * height)),
    )
    if cap >= 1.0:
        return native_mask, native, 1

    ratio = cap
    for _ in range(40):
        size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        candidate_img = img.resize(size, Image.LANCZOS)
        candidate_mask = threshold_mask(candidate_img)
        candidate = measure_features(candidate_mask)
        if _sample_preserves(native, candidate, size[0] / width, budget):
            return candidate_mask, candidate, 1
        ratio -= max(0.05, ratio * 0.07)
        if ratio <= 0.1:
            break

    raise VectorizationBudgetError(
        "Resource budget (max {}px / {} pixels) cannot preserve the glyph "
        "features ({} component(s), {} hole(s), {}-px thinnest stroke). "
        "Raise the sampling budget or supply a smaller scan; refusing to "
        "emit a degraded vectorization.".format(
            budget["max_dimension"],
            budget["max_pixels"],
            native["components"],
            native["holes"],
            native["min_stroke_width"],
        )
    )


def _tracing_options(features, budget):
    """Choose potrace precision from the measured stroke detail."""
    # potrace deletes areas <= turdsize, so it must stay below the smallest
    # real ink component or isolated dots disappear.
    min_area = features["min_component_area"] or 1
    turdsize = min(2, max(0, min_area - 1))
    opttolerance = budget["opttolerance"]
    if (min_area or 0) < 9:
        # Tighten curve fitting while tiny dots are present.
        opttolerance = min(opttolerance, 0.1)
    return turdsize, opttolerance


_MOVETO_RE = re.compile(r"[Mm]")


def count_subpaths(paths):
    """Number of subpaths in parsed potrace SVG <path> elements."""
    return sum(len(_MOVETO_RE.findall(p.get("d", ""))) for p in paths)


def build_design_svg(raw_svg, out_svg, info):
    """Rewrite a potrace SVG into the fixed, y-down 1000-unit design space.

    Raw potrace coordinates live at 1/10 pixel, y-up. They are mapped with an
    explicit, aspect-ratio-preserving transform so that increasing the
    sampling resolution never changes the resulting glyph size:

        design_x = xoff + (raw_x / 10) * scale
        design_y = yoff + height*scale - (raw_y / 10) * scale
    """
    tree = ET.parse(raw_svg)
    root = tree.getroot()
    paths = list(root.iter("{%s}path" % SVG_NS))

    width, height = info["width"], info["height"]
    scale = DESIGN_EM / float(max(width, height))
    k = scale / POTRACE_INTERNAL_UNIT
    xoff = (DESIGN_EM - width * scale) / 2.0
    yoff = (DESIGN_EM - height * scale) / 2.0

    components = info["features"]["components"]
    holes = info["features"]["holes"]
    subpaths = count_subpaths(paths)
    if subpaths != components + holes:
        raise VectorizationBudgetError(
            "Tracing lost topology: expected {} contour(s) ({} component(s) "
            "+ {} hole(s)) but potrace produced {} subpath(s). Refusing to "
            "emit a degraded vectorization.".format(
                components + holes, components, holes, subpaths
            )
        )

    new_root = ET.Element(
        "{%s}svg" % SVG_NS,
        {
            "version": "1.1",
            "width": str(DESIGN_EM),
            "height": str(DESIGN_EM),
            "viewBox": "0 0 {} {}".format(DESIGN_EM, DESIGN_EM),
            "preserveAspectRatio": "xMidYMid meet",
        },
    )
    bbox = info["features"]["bbox"]
    design_bbox = None
    if bbox is not None:
        design_bbox = [
            xoff + bbox[0] * scale,
            yoff + bbox[1] * scale,
            xoff + bbox[2] * scale,
            yoff + bbox[3] * scale,
        ]
    metadata = {
        "handwrite_vectorization": 1,
        "sampled_width": width,
        "sampled_height": height,
        "design_em": DESIGN_EM,
        "threshold": THRESHOLD,
        "transform": {
            "xoff": xoff,
            "yoff": yoff,
            "units_per_pixel": scale,
            "internal_unit": POTRACE_INTERNAL_UNIT,
            "matrix": "translate({},{}) scale({},{})".format(
                xoff, height * scale + yoff, k, -k
            ),
        },
        "native_width": info.get("native_width", width),
        "native_height": info.get("native_height", height),
        "sample_factor": info.get("sample_factor", 1),
        "potrace": {
            "turdsize": info["turdsize"],
            "opttolerance": info["opttolerance"],
        },
        "features": info["features"],
        "design": {
            "ink_bbox": design_bbox,
            "min_stroke_units": (info["features"]["min_stroke_width"] or 0) * scale,
        },
    }
    meta_el = ET.SubElement(new_root, "{%s}metadata" % SVG_NS)
    meta_el.text = json.dumps(metadata, sort_keys=True)

    group = ET.SubElement(
        new_root,
        "{%s}g" % SVG_NS,
        {
            "id": "handwrite-glyph",
            "fill": "#000000",
            "stroke": "none",
            "transform": "translate({:.6f},{:.6f}) scale({:.6f},{:.6f})".format(
                xoff, height * scale + yoff, k, -k
            ),
        },
    )
    for path in paths:
        group.append(path)

    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".{}.".format(os.path.basename(out_svg)),
        suffix=".svg",
        dir=os.path.dirname(out_svg),
    )
    os.close(tmp_fd)
    ET.ElementTree(new_root).write(tmp_path, xml_declaration=True, encoding="utf-8")
    os.replace(tmp_path, out_svg)


class PNGtoSVG:
    """Converter class to convert character PNGs to BMPs and SVGs."""

    def load_budget(self, config=None):
        """Merge an optional config file/dict's vectorization section."""
        section = {}
        if isinstance(config, dict):
            section = config.get("vectorization", {})
        elif config:
            with open(config) as f:
                section = json.load(f).get("vectorization", {})
        budget = dict(DEFAULT_VECTORIZATION)
        budget.update({k: v for k, v in section.items() if k in budget})
        return budget

    def convert(self, directory, config=None):
        """Call converters on each .png in the provided directory.

        Walk through the custom directory containing all .png files from
        sheettopng and convert them to png -> bmp -> svg.

        Parameters
        ----------
        directory : str
            Directory tree containing the character PNGs.
        config : str or dict, optional
            Config file path or dict with an optional "vectorization"
            section controlling sampling/tracing precision and budgets.
        """
        budget = self.load_budget(config)
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".png"):
                    bmp_path, info = self.pngToBmp(os.path.join(root, f), budget)
                    self.bmpToSvg(bmp_path, info, budget)

    def pngToBmp(self, path, budget=None):
        """Convert one character PNG to a bilevel BMP without blind resizing.

        The native crop resolution is kept whenever it fits the resource
        budget; a detail-aware downsample is only used as a last resort and
        is validated before being accepted.

        Parameters
        ----------
        path : str
            Path to the character PNG.
        budget : dict, optional
            Merged vectorization budget.

        Returns
        -------
        (str, dict)
            Path of the written BMP and a sampling descriptor consumed by
            bmpToSvg.

        Raises
        ------
        VectorizationBudgetError
            If no budget-allowed sampling resolution preserves the glyph
            topology.
        """
        if budget is None:
            budget = dict(DEFAULT_VECTORIZATION)

        img = Image.open(path).convert("RGBA")
        native_size = img.size
        mask, features, factor = choose_sample(img, budget)

        bilevel = np.where(mask, 0, 255).astype(np.uint8)
        bmp_path = path[0:-4] + ".bmp"
        Image.fromarray(bilevel, "L").save(bmp_path)

        turdsize, opttolerance = _tracing_options(features, budget)
        info = {
            "width": mask.shape[1],
            "height": mask.shape[0],
            "features": features,
            "turdsize": turdsize,
            "opttolerance": opttolerance,
            "native_width": native_size[0],
            "native_height": native_size[1],
            "sample_factor": factor,
        }
        return bmp_path, info

    def bmpToSvg(self, path, sample=None, budget=None):
        """Convert .bmp image to .svg using potrace.

        When ``sample`` (the descriptor produced by pngToBmp) is given, the
        potrace output is normalized into the fixed 1000-unit design space
        with an explicit transform and validated for topology preservation.
        Without it, plain potrace pass-through behavior is kept.

        Parameters
        ----------
        path : str
            Path to the bmp file to be converted.
        sample : dict, optional
            Sampling descriptor from pngToBmp.
        budget : dict, optional
            Merged vectorization budget (unused without ``sample``).

        Raises
        ------
        PotraceNotFound
            Raised if potrace not found in path by shutil.which()
        VectorizationBudgetError
            If the traced SVG fails topology validation.
        """
        if shutil.which("potrace") is None:
            raise PotraceNotFound("Potrace is either not installed or not in path")

        out_svg = path[0:-4] + ".svg"

        if sample is None:
            subprocess.run(["potrace", path, "-b", "svg", "-o", out_svg], check=True)
            return out_svg

        tmp_fd, tmp_svg = tempfile.mkstemp(
            prefix="." + os.path.basename(path) + ".",
            suffix=".svg",
            dir=os.path.dirname(path),
        )
        os.close(tmp_fd)
        try:
            subprocess.run(
                [
                    "potrace",
                    path,
                    "-b",
                    "svg",
                    "-o",
                    tmp_svg,
                    "-t",
                    str(sample["turdsize"]),
                    "-O",
                    str(sample["opttolerance"]),
                ],
                check=True,
            )
            build_design_svg(tmp_svg, out_svg, sample)
        finally:
            if os.path.exists(tmp_svg):
                os.remove(tmp_svg)
        return out_svg

    def trim(self, im_path):
        im = Image.open(im_path)
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        bbox = list(diff.getbbox())
        bbox[0] -= 1
        bbox[1] -= 1
        bbox[2] += 1
        bbox[3] += 1
        cropped_im = im.crop(bbox)
        cropped_im.save(im_path)
