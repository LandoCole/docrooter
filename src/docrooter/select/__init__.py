"""Selection: which elements a downstream LLM should read for a need (spec v0.6).

**Importing.** ``docrooter.select`` on the top-level package is the :func:`select` function,
which shadows this subpackage there (as ``docrooter.embed`` and ``docrooter.summarize`` do).
Use ``from docrooter.select.selection import ...`` (or ``from docrooter.select import ...``);
``import docrooter.select.selection as m`` fails, and so does monkeypatching by a dotted string
such as ``"docrooter.select.selection.name"``. Patch the module object, from ``sys.modules``
or a ``from`` import, instead.
"""

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
    calibration_of,
    field_fingerprint,
    list_builtin_field_sets,
    load_fields,
    validate_fields,
)
from docrooter.select.fit import fit_selector
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
    "calibration_of",
    "crc_threshold",
    "features",
    "field_fingerprint",
    "fit_selector",
    "list_builtin_field_sets",
    "load_fields",
    "select",
    "validate_fields",
]
