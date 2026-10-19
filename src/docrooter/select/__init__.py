"""Selection: which elements a downstream LLM should read for a need (spec v0.6)."""

from docrooter.select.crc import (
    DEFAULT_GRID,
    CalibUnit,
    DialTable,
    calibrate_dial,
    crc_threshold,
)
from docrooter.select.fields import (
    Field,
    builtin_fields,
    list_builtin_field_sets,
    load_fields,
    validate_fields,
)
from docrooter.select.relevance import CONTENT_KINDS, FEATURE_NAMES, Combiner, features
from docrooter.select.selection import SELECTOR_FILE, Selection, Selector, select

__all__ = [
    "CONTENT_KINDS",
    "DEFAULT_GRID",
    "FEATURE_NAMES",
    "SELECTOR_FILE",
    "CalibUnit",
    "Combiner",
    "DialTable",
    "Field",
    "Selection",
    "Selector",
    "builtin_fields",
    "calibrate_dial",
    "crc_threshold",
    "features",
    "list_builtin_field_sets",
    "load_fields",
    "select",
    "validate_fields",
]
