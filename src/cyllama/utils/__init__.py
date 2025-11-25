"""
Utility modules for cyllama.

- color: Terminal color utilities
- log: Logging configuration with colored output
"""

from .log import config, CustomFormatter, DEBUG, COLOR
from .color import (
    # 8-bit colors
    black, red, green, yellow, blue, magenta, cyan, white,
    # bright/light colors
    grey, gray, bright_red, bright_green, bright_yellow,
    bright_blue, bright_magenta, bright_cyan, bright_white,
    # backgrounds
    black_bg, red_bg, green_bg, yellow_bg, blue_bg, magenta_bg, cyan_bg, white_bg,
    # highlights
    black_hl, red_hl, green_hl, yellow_hl, blue_hl, magenta_hl, cyan_hl, white_hl,
    # styles
    bold, italic, underline, strike, blink,
    # 256 colors
    fg256, bg256, hl256,
    # grayscale
    grayscale, grayscale_bg, grayscale_hl,
    # utilities
    use_color, use_color_no_tty, esc, END,
)

__all__ = [
    # log module
    "config",
    "CustomFormatter",
    "DEBUG",
    "COLOR",
    # color module - 8-bit colors
    "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
    # color module - bright/light colors
    "grey", "gray", "bright_red", "bright_green", "bright_yellow",
    "bright_blue", "bright_magenta", "bright_cyan", "bright_white",
    # color module - backgrounds
    "black_bg", "red_bg", "green_bg", "yellow_bg", "blue_bg", "magenta_bg", "cyan_bg", "white_bg",
    # color module - highlights
    "black_hl", "red_hl", "green_hl", "yellow_hl", "blue_hl", "magenta_hl", "cyan_hl", "white_hl",
    # color module - styles
    "bold", "italic", "underline", "strike", "blink",
    # color module - 256 colors
    "fg256", "bg256", "hl256",
    # color module - grayscale
    "grayscale", "grayscale_bg", "grayscale_hl",
    # color module - utilities
    "use_color", "use_color_no_tty", "esc", "END",
]
