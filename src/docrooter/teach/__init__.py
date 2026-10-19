"""Teacher LLMs for labelling corpus elements (spec §7).

A teacher chooses among an element's options by multiple-choice log-probability scoring, or
answers a yes/no question about it. Engines:

- ``mlx:<repo>@<rev>``: mlx-lm on Apple silicon (``teach-mlx`` extra), the default:
  :data:`~docrooter.teach.mlx.DEFAULT_TEACHER`;
- ``hf:<repo>@<rev>``: transformers on MPS, CUDA or CPU (``models`` extra);
- ``openai:<model>``: an OpenAI-compatible endpoint returning ``top_logprobs`` (``llm``).

Jev is never a teacher (TypeSafe MCA §2.3(b)): :func:`get_teacher` refuses it with
:class:`~docrooter.errors.TeacherNotAllowedError`.

Importing this package loads no model library; engines import theirs on the first item.
"""

from docrooter.errors import TeacherNotAllowedError
from docrooter.teach.base import (
    JEV_REFUSAL,
    LogitRunner,
    LogprobTeacher,
    TeacherEngine,
    TeachItem,
    get_teacher,
    label_key,
)
from docrooter.teach.prompts import (
    MAX_CONTEXT_TOKENS,
    MAX_TEXT_TOKENS,
    PROMPT_VERSION,
    context_text,
    function_question,
    options_block,
    prompt_hash,
    table_text,
)

__all__ = [
    "JEV_REFUSAL",
    "MAX_CONTEXT_TOKENS",
    "MAX_TEXT_TOKENS",
    "PROMPT_VERSION",
    "LogitRunner",
    "LogprobTeacher",
    "TeachItem",
    "TeacherEngine",
    "TeacherNotAllowedError",
    "context_text",
    "function_question",
    "get_teacher",
    "label_key",
    "options_block",
    "prompt_hash",
    "table_text",
]
