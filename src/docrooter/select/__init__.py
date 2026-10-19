"""Selection: which elements a downstream LLM should read for a need (spec v0.6)."""

from docrooter.select.fields import (
    Field,
    builtin_fields,
    list_builtin_field_sets,
    load_fields,
    validate_fields,
)

__all__ = ["Field", "builtin_fields", "list_builtin_field_sets", "load_fields", "validate_fields"]
