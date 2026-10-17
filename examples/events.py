"""Watch the pipeline run: an ``on_event`` callback receives every stage with its timing.

Needs no models (keyword backend). The playground (``docrooter serve``) is built on the same
events.
"""

from docrooter import builtin_taxonomy, classify_document, load, normalize_sections
from docrooter.events import Event, pipeline_run

MARKDOWN = """\
# Sparse Attention for Long Documents

## Abstract
We propose a sparse attention method and report experiments, results and a dataset.

## Results
Results of the experiments against the baselines.
"""


def show(event: Event) -> None:
    print(f"{event.t:7.3f}s  {event.stage:<10} {event.status:<8} {event.data}")


# One call is one run: it reports its stages, then "done" (or "error" before re-raising).
doc = load(MARKDOWN, on_event=show)

# pipeline_run shares one clock and one "done" across several calls.
with pipeline_run(show):
    result = classify_document(doc, builtin_taxonomy("doc_types"), backend="keyword", on_event=show)
    norm = normalize_sections(
        doc, doc_type=result.best.node_ids[-1], backend="keyword", on_event=show
    )
