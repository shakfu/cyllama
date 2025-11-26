"""
color.py
========

This the color.py module from from `https://github.com/reorx/python-terminal-color`
Thanks to reorx for writing it!


Usage
-----

>>> import color
>>>
>>> # 8-bit color
>>> print(red('red') + green('green') + blue('blue'))
>>> print(bold(yellow('bold yellow')) + underline(cyan('underline cyan')))
>>> print(magenta_hl('magenta highlight'))
>>>
>>> # xterm 256 color
>>> print(bg256('A9D5DE', fg256('276F86', 'Info!')))
>>> print(bg256('E0B4B4', fg256('912D2B', 'Warning!')))
>>> print(hl256('10a3a3', 'Teal'))

Note:

1. Every color function receives and returns string, so that the result
   could be used with any other strings, in any string formatting situation.

2. If you pass a str type string, the color function will return a str.
   If you pass a bytes type string, the color function will return a bytes string.

3. Color functions could be composed together, like put ``red`` into ``bold``,
   or put ``bg256`` into ``fg256``. ``xxx_hl`` and ``hl256`` are mostly used
   independently.

API
---

8-bit colors:

========  ============  ===========
 Colors    Background    Highlight
========  ============  ===========
black     black_bg      black_hl
red       red_bg        red_hl
green     green_bg      green_hl
yellow    yellow_bg     yellow_hl
blue      blue_bg       blue_hl
magenta   magenta_bg    magenta_hl
cyan      cyan_bg       cyan_hl
white     white_bg      white_hl
========  ============  ===========

Styles:
- bold
- italic
- underline
- strike
- blink

.. py:function:: <color_function>(s)

   Decorate string with specified color or style.

   A color function with ``_bg`` suffix means it will set color as background.
   A color function with ``_hl`` suffix means it will set color as background,
   and change the foreground as well to make the word standout.

   :param str s: The input string
   :return: The decorated string
   :rtype: string
   :raises ValueError: if the message_body exceeds 160 characters


256 colors:
- fg256
- bg256
- hl256

.. py:function:: <256_color_function>(hexrgb, s)

   Decorate string with specified hex rgb color

   ``fg256`` will set color as foreground.
   ``bg256`` will set color as background.
   ``hg256`` will highlight input with the color.

   :param str hexrgb: The hex rgb color string, accept length 3 and 6. eg: ``555``, ``912D2B``
   :param str s: The input string
   :return: The decorated string
   :rtype: string
   :raises ValueError: If the input string's length not equal to 3 or 6.
"""

from typing import Union, Any, Callable, Optional, Tuple, List, Dict
import sys


# Identity function (Python 2/3 compatibility layer from original)
def t_(s):
    """Identity function for string compatibility."""
    return s


_use_color_no_tty = True


def use_color_no_tty(flag):
    global _use_color_no_tty
    _use_color_no_tty = flag


def use_color():
    if sys.stdout.isatty():
        return True
    if _use_color_no_tty:
        return True
    return False


def esc(*codes: Union[int, str]) -> str:
    """Produces an ANSI escape code from a list of integers
    :rtype: text_type
    """
    return '\x1b[{}m'.format(';'.join(str(c) for c in codes))


###############################################################################
# 8 bit Color
###############################################################################

def make_color(start, end: str) -> Callable[[str], str]:
    def color_func(s: str) -> str:
        if not use_color():
            return s

        # render
        return start + s + end

    return color_func


# According to https://en.wikipedia.org/wiki/ANSI_escape_code#graphics ,
# 39 is reset for foreground, 49 is reset for background, 0 is reset for all
# we can use 0 for convenience, but it will make color combination behaves weird.
END = esc(0)

FG_END = esc(39)
black = make_color(esc(30), FG_END)
red = make_color(esc(31), FG_END)
green = make_color(esc(32), FG_END)
yellow = make_color(esc(33), FG_END)
blue = make_color(esc(34), FG_END)
magenta = make_color(esc(35), FG_END)
cyan = make_color(esc(36), FG_END)
white = make_color(esc(37), FG_END)

# Bright/light colors (codes 90-97)
grey = make_color(esc(90), FG_END)  # bright black = grey
gray = grey  # alias for American spelling
bright_red = make_color(esc(91), FG_END)
bright_green = make_color(esc(92), FG_END)
bright_yellow = make_color(esc(93), FG_END)
bright_blue = make_color(esc(94), FG_END)
bright_magenta = make_color(esc(95), FG_END)
bright_cyan = make_color(esc(96), FG_END)
bright_white = make_color(esc(97), FG_END)

BG_END = esc(49)
black_bg = make_color(esc(40), BG_END)
red_bg = make_color(esc(41), BG_END)
green_bg = make_color(esc(42), BG_END)
yellow_bg = make_color(esc(43), BG_END)
blue_bg = make_color(esc(44), BG_END)
magenta_bg = make_color(esc(45), BG_END)
cyan_bg = make_color(esc(46), BG_END)
white_bg = make_color(esc(47), BG_END)

HL_END = esc(22, 27, 39)
#HL_END = esc(22, 27, 0)

black_hl = make_color(esc(1, 30, 7), HL_END)
red_hl = make_color(esc(1, 31, 7), HL_END)
green_hl = make_color(esc(1, 32, 7), HL_END)
yellow_hl = make_color(esc(1, 33, 7), HL_END)
blue_hl = make_color(esc(1, 34, 7), HL_END)
magenta_hl = make_color(esc(1, 35, 7), HL_END)
cyan_hl = make_color(esc(1, 36, 7), HL_END)
white_hl = make_color(esc(1, 37, 7), HL_END)

bold = make_color(esc(1), esc(22))
italic = make_color(esc(3), esc(23))
underline = make_color(esc(4), esc(24))
strike = make_color(esc(9), esc(29))
blink = make_color(esc(5), esc(25))


###############################################################################
# Xterm 256 Color (delete if you don't need)
###############################################################################
#
# Rewrite from: https://gist.github.com/MicahElliott/719710

import re  # NOQA

# Default color levels for the color cube
CUBELEVELS: List[int] = [0x00, 0x5f, 0x87, 0xaf, 0xd7, 0xff]

# Generate a list of midpoints of the above list
SNAPS: List[int] = [(x + y) // 2 for x, y in list(zip(CUBELEVELS, [0] + CUBELEVELS))[1:]]

# Gray-scale range.
_GRAYSCALE = [
    (0x08, 232),  # 0x08 means 080808 in HEX color
    (0x12, 233),
    (0x1c, 234),
    (0x26, 235),
    (0x30, 236),
    (0x3a, 237),
    (0x44, 238),
    (0x4e, 239),
    (0x58, 240),
    (0x62, 241),
    (0x6c, 242),
    (0x76, 243),
    (0x80, 244),
    (0x8a, 245),
    (0x94, 246),
    (0x9e, 247),
    (0xa8, 248),
    (0xb2, 249),
    (0xbc, 250),
    (0xc6, 251),
    (0xd0, 252),
    (0xda, 253),
    (0xe4, 254),
    (0xee, 255),
]
GRAYSCALE: Dict[int, int] = dict(_GRAYSCALE)

GRAYSCALE_POINTS: List[int] = [i for i, _ in _GRAYSCALE]


def get_closest(v: int, l: list):
    return min(l, key=lambda x: abs(x - v))


class Memorize(dict):
    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result


def memorize(func) -> Callable:
    func._cache = {}

    def wrapper(*args, **kwargs):
        if kwargs:
            return func(*args, **kwargs)
        if args not in func._cache:
            func._cache[args] = func(*args, **kwargs)
        return func._cache[args]

    for i in ('__module__', '__name__', '__doc__'):
        setattr(wrapper, i, getattr(func, i))
    wrapper.__dict__.update(getattr(func, '__dict__', {}))  # type: ignore
    wrapper._origin = func  # type: ignore
    return wrapper


@memorize
def rgb_to_xterm(r: int, g: int, b: int) -> int:
    """ Converts RGB values to the nearest equivalent xterm-256 color.
    """
    if r == g == b:
        # use gray scale
        gs = get_closest(r, GRAYSCALE_POINTS)
        return GRAYSCALE[gs]
    # Using list of snap points, convert RGB value to cube indexes
    r, g, b = map(lambda x: len(tuple(s for s in SNAPS if s < x)), (r, g, b))
    # Simple colorcube transform
    return r * 36 + g * 6 + b + 16


@memorize
def hex_to_rgb(hx: str) -> Tuple[int, int, int]:
    hxlen = len(hx)
    if hxlen != 3 and hxlen != 6:
        raise ValueError('hx color must be of length 3 or 6')
    if hxlen == 3:
        hx = t_('').join(i * 2 for i in hx)
    parts = [int(h, 16) for h in re.split(t_(r'(..)(..)(..)'), hx)[1:4]]
    return tuple(parts)  # type: ignore


def make_256(start: str, end: str) -> Callable[[Union[tuple, str], str, Optional[Tuple[int, int, int]]], str]:
    def rgb_func(rgb: Union[tuple, str], s: str, x: Optional[Tuple[int, int, int]] = None) -> str:
        """
        :param rgb: (R, G, B) tuple, or RRGGBB hex string
        """
        if not use_color():
            return s

        t = t_(s)

        # render
        if not isinstance(rgb, tuple):
            rgb = hex_to_rgb(t_(rgb))
        if x is not None:
            xcolor = x
        else:
            xcolor = rgb_to_xterm(*rgb)

        tpl = start + t_('{s}') + end
        f = tpl.format(
            x=xcolor,
            s=t)

        return f

    return rgb_func


fg256 = make_256(esc(38, 5, t_('{x}')), esc(39))
bg256 = make_256(esc(48, 5, t_('{x}')), esc(49))
hl256 = make_256(esc(1, 38, 5, t_('{x}'), 7), esc(27, 39, 22))

_grayscale_xterm_codes = [i for _, i in _GRAYSCALE]
grayscale = {(i - _grayscale_xterm_codes[0]): make_color(esc(38, 5, i), esc(39)) for i in _grayscale_xterm_codes}
grayscale_bg = {(i - _grayscale_xterm_codes[0]): make_color(esc(48, 5, i), esc(49)) for i in _grayscale_xterm_codes}
grayscale_hl = {(i - _grayscale_xterm_codes[0]): make_color(esc(1, 38, 5, i, 7), esc(27, 39, 22)) for i in _grayscale_xterm_codes}

# ----------------------------------------------------------
# colored test

COLORS = {
    # Standard foreground colors
    'black': black,
    'red': red,
    'green': green,
    'yellow': yellow,
    'blue': blue,
    'magenta': magenta,
    'cyan': cyan,
    'white': white,

    # Bright/light foreground colors
    'grey': grey,
    'gray': gray,
    'bright_red': bright_red,
    'bright_green': bright_green,
    'bright_yellow': bright_yellow,
    'bright_blue': bright_blue,
    'bright_magenta': bright_magenta,
    'bright_cyan': bright_cyan,
    'bright_white': bright_white,

    # Background colors
    'black_bg': black_bg,
    'red_bg': red_bg,
    'green_bg': green_bg,
    'yellow_bg': yellow_bg,
    'blue_bg': blue_bg,
    'magenta_bg': magenta_bg,
    'cyan_bg': cyan_bg,
    'white_bg': white_bg,

    # Highlight colors (bold + reverse)
    'black_hl': black_hl,
    'red_hl': red_hl,
    'green_hl': green_hl,
    'yellow_hl': yellow_hl,
    'blue_hl': blue_hl,
    'magenta_hl': magenta_hl,
    'cyan_hl': cyan_hl,
    'white_hl': white_hl,

    # Styles
    'bold': bold,
    'italic': italic,
    'underline': underline,
    'strike': strike,
    'blink': blink,
}


def cprint(txt: str, color: Optional[str] = None, end: str = '\n') -> None:
    """Print text with optional color.

    Args:
        txt: Text to print
        color: Color name from COLORS dict (e.g., 'red', 'bold', 'green_hl')
        end: String appended after the last value, default newline
    """
    if color and color in COLORS:
        txt = COLORS[color](txt)
    print(txt, end=end)


def section(txt: str, color: str = 'bright_cyan', width: int = 70, char: str = '=') -> None:
    """Print a section header with decorative lines.

    Args:
        txt: Section title
        color: Color for the title text
        width: Width of the decorative lines
        char: Character to use for decorative lines
    """
    print("\n" + char * width)
    cprint(txt, color)
    print(char * width)

def subsection(txt: str, color: str = 'bright_cyan', width: int = 70, char: str = '-') -> None:
    """Print a subsection header with decorative below.

    Args:
        txt: Section title
        color: Color for the title text
        width: Width of the decorative lines
        char: Character to use for decorative lines
    """
    print()
    cprint(txt, color)
    print(char * width)

def header(txt: str, color: str = 'bold', width: int = 70, char: str = '=') -> None:
    """Print a centered header with decorative lines above and below.

    Args:
        txt: Header text
        color: Color for the header text
        width: Total width including padding
        char: Character to use for decorative lines
    """
    print(char * width)
    padding = (width - len(txt)) // 2
    centered = ' ' * padding + txt
    cprint(centered, color)
    print(char * width)


def subheader(txt: str, color: str = 'bright_white', char: str = '-') -> None:
    """Print a subheader with a decorative line below.

    Args:
        txt: Subheader text
        color: Color for the text
        char: Character for the underline
    """
    cprint(txt, color)
    print(char * len(txt))


def success(txt: str, prefix: str = '[OK]') -> None:
    """Print a success message in green.

    Args:
        txt: Success message
        prefix: Prefix symbol/text
    """
    cprint(f"{prefix} {txt}", 'green')


def error(txt: str, prefix: str = '[ERROR]') -> None:
    """Print an error message in red.

    Args:
        txt: Error message
        prefix: Prefix symbol/text
    """
    cprint(f"{prefix} {txt}", 'red')


def warning(txt: str, prefix: str = '[WARN]') -> None:
    """Print a warning message in yellow.

    Args:
        txt: Warning message
        prefix: Prefix symbol/text
    """
    cprint(f"{prefix} {txt}", 'yellow')


def info(txt: str, prefix: str = '[INFO]') -> None:
    """Print an info message in cyan.

    Args:
        txt: Info message
        prefix: Prefix symbol/text
    """
    cprint(f"{prefix} {txt}", 'cyan')


def debug(txt: str, prefix: str = '[DEBUG]') -> None:
    """Print a debug message in grey.

    Args:
        txt: Debug message
        prefix: Prefix symbol/text
    """
    cprint(f"{prefix} {txt}", 'grey')


def bullet(txt: str, color: Optional[str] = None, indent: int = 0, marker: str = '-') -> None:
    """Print a bullet point item.

    Args:
        txt: Bullet item text
        color: Optional color for the text
        indent: Number of spaces to indent
        marker: Bullet marker character
    """
    prefix = ' ' * indent + marker + ' '
    cprint(f"{prefix}{txt}", color)


def numbered(items: list, color: Optional[str] = None, start: int = 1) -> None:
    """Print a numbered list.

    Args:
        items: List of items to print
        color: Optional color for all items
        start: Starting number
    """
    for i, item in enumerate(items, start=start):
        cprint(f"{i}. {item}", color)


def kv(key: str, value: str, key_color: str = 'bright_white',
       value_color: Optional[str] = None, separator: str = ': ') -> None:
    """Print a key-value pair with optional colors.

    Args:
        key: The key/label
        value: The value
        key_color: Color for the key
        value_color: Color for the value (None for default)
        separator: String between key and value
    """
    key_str = COLORS[key_color](key) if key_color and key_color in COLORS else key
    value_str = COLORS[value_color](str(value)) if value_color and value_color in COLORS else str(value)
    print(f"{key_str}{separator}{value_str}")


def progress(current: int, total: int, width: int = 40,
             fill_char: str = '#', empty_char: str = '-',
             color: str = 'green') -> None:
    """Print a progress bar.

    Args:
        current: Current progress value
        total: Total/maximum value
        width: Width of the progress bar
        fill_char: Character for filled portion
        empty_char: Character for empty portion
        color: Color for the filled portion
    """
    if total <= 0:
        pct = 0
    else:
        pct = min(current / total, 1.0)

    filled = int(width * pct)
    empty = width - filled

    bar_filled = COLORS[color](fill_char * filled) if color in COLORS else fill_char * filled
    bar_empty = empty_char * empty

    print(f"[{bar_filled}{bar_empty}] {current}/{total} ({pct*100:.1f}%)")


def divider(char: str = '-', width: int = 70, color: Optional[str] = None) -> None:
    """Print a horizontal divider line.

    Args:
        char: Character to use for the divider
        width: Width of the divider
        color: Optional color for the divider
    """
    line = char * width
    cprint(line, color)


def box(txt: str, color: Optional[str] = None, padding: int = 1,
        border_char: str = '*', width: Optional[int] = None) -> None:
    """Print text inside a box.

    Args:
        txt: Text to display inside the box
        color: Color for the text
        padding: Padding spaces on each side of text
        border_char: Character for the box border
        width: Box width (auto-calculated if None)
    """
    lines = txt.split('\n')
    max_len = max(len(line) for line in lines)
    box_width = width or (max_len + padding * 2 + 2)

    # Top border
    print(border_char * box_width)

    # Content lines
    for line in lines:
        inner_width = box_width - 2
        padded = line.center(inner_width)
        content = COLORS[color](padded) if color and color in COLORS else padded
        print(f"{border_char}{content}{border_char}")

    # Bottom border
    print(border_char * box_width)


def table_row(columns: list, widths: list, colors: Optional[list] = None,
              separator: str = ' | ') -> None:
    """Print a table row with fixed column widths.

    Args:
        columns: List of column values
        widths: List of column widths
        colors: Optional list of colors for each column
        separator: String between columns
    """
    parts = []
    for i, (col, width) in enumerate(zip(columns, widths)):
        text = str(col)[:width].ljust(width)
        if colors and i < len(colors) and colors[i] and colors[i] in COLORS:
            text = COLORS[colors[i]](text)
        parts.append(text)
    print(separator.join(parts))

