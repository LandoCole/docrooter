"""The element tagger (spec §9, §10): a local student that tags every element by topic and
function, trained from the teacher's labels.

- :class:`ElementTagger`: topic (Beam(3) over the merged ``element_topics`` tree, restricted to
  the document's family on request) and function heads over one shared body, saved as a bundle;
- :func:`train_tagger`: stage A (heads on the frozen body) or B (fine-tuned body);
- :func:`calibrate_tagger`: one temperature per facet on the ``calib`` split;
- :func:`active_learning_round`: the least certain unlabelled elements, labelled by the teacher.

Importing this package loads no model library; the ``models`` extra is needed to train or tag.
"""

from docrooter.tagger.bundle import ElementTagger, student_text
from docrooter.tagger.functions import (
    FunctionRow,
    function_rows,
    function_targets,
    train_function_heads,
)
from docrooter.tagger.train import (
    active_learning_round,
    calibrate_tagger,
    select_uncertain,
    train_tagger,
)

__all__ = [
    "ElementTagger",
    "FunctionRow",
    "active_learning_round",
    "calibrate_tagger",
    "function_rows",
    "function_targets",
    "select_uncertain",
    "student_text",
    "train_function_heads",
    "train_tagger",
]
