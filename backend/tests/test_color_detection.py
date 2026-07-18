"""color_detection.py tests. Uses synthetic solid-color images so these
stay deterministic and fast, no real photos or bounding boxes needed to
prove the color math itself is correct."""

from PIL import Image

from app.services.color_detection import _inset, detect_color, dominant_rgb, nearest_color_name


def test_dominant_rgb_of_a_solid_color_image_is_exact():
    image = Image.new("RGB", (40, 40), color=(220, 20, 20))
    assert dominant_rgb(image) == (220, 20, 20)


def test_nearest_color_name_matches_an_exact_canonical_color():
    # #DC2626 is "red" in colors.COLOR_HEX
    assert nearest_color_name((0xDC, 0x26, 0x26)) == "red"


def test_nearest_color_name_matches_a_close_but_inexact_color():
    # a slightly darker red should still resolve to "red", not something else
    assert nearest_color_name((0xD0, 0x20, 0x20)) == "red"


def test_detect_color_crops_the_bbox_before_reading_color():
    # left half red, right half blue; a bbox over just the left half
    # should read red, not some average of both halves
    image = Image.new("RGB", (40, 20), color=(59, 111, 214))  # "blue"
    for x in range(20):
        for y in range(20):
            image.putpixel((x, y), (220, 20, 20))  # "red"

    left_half_color = detect_color(image, (0, 0, 20, 20))
    right_half_color = detect_color(image, (20, 0, 40, 20))

    assert left_half_color == "red"
    assert right_half_color == "blue"


def test_detect_color_excludes_a_border_strip_of_a_neighboring_color():
    """The actual measured failure this inset fixes: a garment box whose
    edge includes a strip of something else (skin, background) skews the
    naive median. A red square with a thin blue border on all sides
    should read red once the border is trimmed away, not a blend."""
    image = Image.new("RGB", (100, 100), color=(59, 111, 214))  # blue border
    for x in range(10, 90):
        for y in range(10, 90):
            image.putpixel((x, y), (220, 20, 20))  # red interior

    assert detect_color(image, (0, 0, 100, 100)) == "red"


def test_inset_shrinks_the_box_symmetrically():
    assert _inset(0, 0, 100, 100, 0.15) == (15, 15, 85, 85)


def test_inset_leaves_a_too_small_box_unchanged_rather_than_collapsing_it():
    # 15% of a 4px box rounds to 0px trimmed either way, but a box this
    # small must never be allowed to shrink to zero area.
    assert _inset(0, 0, 4, 4, 0.5) == (0, 0, 4, 4)
