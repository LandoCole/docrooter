"""Optional ML layout engines (spec §6): regions that type the lines inside them.

``import docrooter.layout`` loads no model package; ``HeronLayout`` imports ``onnxruntime``
when it is created (extra ``layout``).
"""

from docrooter.layout.base import (
    HERON_FLAGS,
    HERON_TO_KIND,
    MIN_CONFIDENCE,
    LayoutEngine,
    LayoutRegion,
    apply_regions,
    region_kind,
)
from docrooter.layout.heron import HeronLayout
from docrooter.layout.registry import (
    available_layout_engines,
    get_layout_engine,
    register_layout_engine,
)

__all__ = [
    "HERON_FLAGS",
    "HERON_TO_KIND",
    "MIN_CONFIDENCE",
    "HeronLayout",
    "LayoutEngine",
    "LayoutRegion",
    "apply_regions",
    "available_layout_engines",
    "get_layout_engine",
    "region_kind",
    "register_layout_engine",
]
