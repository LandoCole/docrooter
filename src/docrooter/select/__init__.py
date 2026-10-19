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

__all__ = [
    "CONTENT_KINDS",
    "DEFAULT_GRID",
    "FEATURE_NAMES",
    "CalibUnit",
    "Combiner",
    "DialTable",
    "Field",
    "builtin_fields",
    "calibrate_dial",
    "crc_threshold",
    "features",
    "list_builtin_field_sets",
    "load_fields",
    "validate_fields",
]
