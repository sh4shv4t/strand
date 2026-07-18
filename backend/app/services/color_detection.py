"""Deterministic garment color detection: read the actual pixel colors out
of a garment's bounding box and map them to the nearest canonical color
name, no VLM call needed for this axis specifically.

This is the concrete version of an idea from the very first design pass
(Working_notes.md Section 2): instead of asking a model to guess a color
in words, run detection (here, Fashionpedia's own ground-truth bounding
boxes, no detector of our own needed) and read the real RGB values off
the detected region. Cheap, fully deterministic, and it reuses the exact
canonical color vocabulary (colors.py's COLOR_HEX) already used to render
swatch chips, so a detected color is guaranteed to be a name the rest of
the app already understands.

Known limitation, measured directly, not just assumed: a bounding box is
a rectangle, not a garment mask, and the concrete way that bites here is
skin. Checked a real failure by hand, a "shirt" box on a bearded man
photographed near a warmly-lit yellow door read as "brown" (127, 117,
107) instead of the shirt's actual light grey, because the box's
neckline edge includes a real strip of visible neck. Cropping in from
every edge by INSET_FRACTION before sampling fixed this specific case,
(158, 150, 139), "khaki", clearly closer to right than "brown" was,
without any slot-specific logic (no different handling for "this is an
upper-body garment, trim the top more"), a plain uniform inset already
buys most of the benefit. Still not exact, a tighter segmentation mask
would beat a rectangle at this outright, Fashionpedia's own ground truth
only exposes boxes through this particular mirror, see
pull_fashionpedia_sample.py, and no attempt is made here to correct for
scene lighting/white balance, a color read under warm ambient light can
still legitimately skew warmer than the same garment would read in
neutral light.
"""

from PIL import Image

from app.services.colors import COLOR_HEX

Bbox = tuple[float, float, float, float]

# Fraction trimmed from each edge before sampling, see the module docstring
# for the measured shirt/neckline case this is directly answering.
INSET_FRACTION = 0.15


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


_CANONICAL_RGB: dict[str, tuple[int, int, int]] = {name: _hex_to_rgb(hex_) for name, hex_ in COLOR_HEX.items()}


def dominant_rgb(crop: Image.Image) -> tuple[int, int, int]:
    """Per-channel median RGB across the crop's pixels. Median rather than
    mean specifically to be more robust to a handful of highlight/shadow
    outlier pixels within one garment than a plain average would be.
    Downsamples first since a median over a few thousand pixels is just as
    representative as one over a few hundred thousand, and much faster.
    """
    rgb_image = crop.convert("RGB")
    rgb_image.thumbnail((64, 64))
    pixels = rgb_image.get_flattened_data()
    if not pixels:
        raise ValueError("crop has no pixels")

    reds = sorted(p[0] for p in pixels)
    greens = sorted(p[1] for p in pixels)
    blues = sorted(p[2] for p in pixels)
    mid = len(pixels) // 2
    return (reds[mid], greens[mid], blues[mid])


def nearest_color_name(rgb: tuple[int, int, int]) -> str:
    """Nearest canonical color name by squared Euclidean distance in RGB
    space. A perceptual space (Lab, via CIEDE2000) would be a more
    faithful match to human color perception; plain RGB distance is the
    simpler, still-reasonable choice for a first version, and easy to
    swap later behind this same function signature.
    """

    def squared_distance(candidate: tuple[int, int, int]) -> int:
        return sum((a - b) ** 2 for a, b in zip(rgb, candidate))

    return min(_CANONICAL_RGB, key=lambda name: squared_distance(_CANONICAL_RGB[name]))


def _inset(x_min: int, y_min: int, x_max: int, y_max: int, fraction: float) -> Bbox:
    """Trims `fraction` of the box's width/height off every edge, biasing
    the sample toward the garment's center and away from its boundary
    with whatever is next to it (skin, background, another garment).
    Clamped so a box too small to survive the trim is returned unchanged
    rather than collapsing to zero area.
    """
    width, height = x_max - x_min, y_max - y_min
    dx, dy = int(width * fraction), int(height * fraction)
    if width - 2 * dx < 1 or height - 2 * dy < 1:
        return (x_min, y_min, x_max, y_max)
    return (x_min + dx, y_min + dy, x_max - dx, y_max - dy)


def detect_color(image: Image.Image, bbox: Bbox) -> str:
    """bbox is (x_min, y_min, x_max, y_max) in absolute pixel coordinates,
    Fashionpedia's own ground-truth format. Insets the box before
    cropping (see INSET_FRACTION), reads the dominant color, and maps it
    to the nearest name the rest of the app recognizes.
    """
    x_min, y_min, x_max, y_max = (int(v) for v in bbox)
    x_min, y_min, x_max, y_max = _inset(x_min, y_min, x_max, y_max, INSET_FRACTION)
    crop = image.crop((x_min, y_min, x_max, y_max))
    rgb = dominant_rgb(crop)
    return nearest_color_name(rgb)
