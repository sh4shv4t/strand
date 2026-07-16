"""Shared color-name -> hex map, used to render swatch chips for both the
hand-written mock catalog and real dataset samples."""

COLOR_HEX = {
    "red": "#DC2626",
    "white": "#F8FAFC",
    "black": "#111318",
    "blue": "#3B6FD6",
    "yellow": "#F5C542",
    "green": "#10B981",
    "grey": "#94A3B8",
    "gray": "#94A3B8",
    "navy": "#1E2A5E",
    "amber": "#F59E0B",
    "beige": "#D9C8A9",
    "brown": "#8B5E3C",
    "khaki": "#C3B091",
    "denim": "#4C6B8A",
    "nude": "#D8B79A",
    "orange": "#F97316",
    "pink": "#EC4899",
    "purple": "#8B5CF6",
    "tan": "#D2B48C",
}


def color_to_hex(color: str | None) -> str | None:
    if not color:
        return None
    return COLOR_HEX.get(color.lower())
