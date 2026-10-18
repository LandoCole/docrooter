"""The rooter (spec §7): one configurable pipeline from a document to an :class:`Analysis`.

:func:`process` resolves a profile or :class:`RouterConfig` into engines (text, OCR, layout)
and backends (document type, sections, escalation), runs them, and records each decision in
the route log. :func:`process_many` does the same over many documents with one set of engines.
"""

from docrooter.rooter.analysis import Analysis
from docrooter.rooter.config import RouterConfig
from docrooter.rooter.process import process, process_many
from docrooter.rooter.profiles import DEFAULT_PROFILE, PROFILES, profile_defaults

__all__ = [
    "DEFAULT_PROFILE",
    "PROFILES",
    "Analysis",
    "RouterConfig",
    "process",
    "process_many",
    "profile_defaults",
]
