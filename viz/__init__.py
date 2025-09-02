"""
Scientific visualization package for publication-quality figures.

This package provides:
- Global figure styling (figstyle)
- Color palettes and accessibility tools (palette)
- Reusable plotting functions (figures)
"""

from . import figstyle
from . import palette
from . import figures

__version__ = "1.0.0"
__all__ = ["figstyle", "palette", "figures"]