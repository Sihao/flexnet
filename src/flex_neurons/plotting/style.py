"""Shared plotting style constants and helpers."""
from typing import Dict

PALETTE: Dict[str, str] = {
    "vanilla": "#1f77b4",
    "flex": "#d62728",
    "exp2": "#1f77b4",
    "exp4": "#d62728",
    "baseline": "#7f7f7f",
}

FIGSIZE_SINGLE = (5, 3.5)
FIGSIZE_DOUBLE = (10, 3.5)
FIGSIZE_GRID = (12, 8)
FONT_SIZE_BASE = 11
FONT_SIZE_AXIS = 12
FONT_SIZE_TITLE = 14
LINE_WIDTH = 1.8
DPI = 300


def apply_default_style() -> None:
    """Set matplotlib rcParams to project defaults. Call at top of any plot script."""
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = DPI
    mpl.rcParams['font.size'] = FONT_SIZE_BASE
    mpl.rcParams['axes.labelsize'] = FONT_SIZE_AXIS
    mpl.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
    mpl.rcParams['lines.linewidth'] = LINE_WIDTH
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False


def get_color(name: str, default: str = "#666666") -> str:
    return PALETTE.get(name, default)
