"""Classify sections with TypeSafe Jev (needs docrooter[jev] and a TYPESAFE_API_KEY).

Keep the key out of your code and out of git. Export it in your shell, or keep it in a
git-ignored .env file and run:  uv run --env-file .env python examples/jev_backend.py
"""

import os
import sys

from docrooter import Classifier, Greedy, Taxonomy, load

if not os.environ.get("TYPESAFE_API_KEY"):
    sys.exit("set TYPESAFE_API_KEY first (see the README)")

MARKDOWN = """\
# Annual report

## Income taxes
The effective tax rate was 21.4 percent, down from 23.1 percent a year earlier.

## Class action
A shareholder filed a lawsuit and the court dismissed it in March.
"""

taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {"id": "financial", "label": "Financial", "description": "Accounting and taxes"},
            {"id": "legal", "label": "Legal", "description": "Contracts and legal proceedings"},
        ]
    }
)

# "jev" pins jev-1.13.0; use "jev:jev-preview" or another model id to choose a different one.
# For an OpenAI-compatible server use "openai:<model>" with OPENAI_API_KEY (or
# DOCROOTER_LLM_BASE_URL and DOCROOTER_LLM_API_KEY) set instead.
classifier = Classifier(taxonomy, "jev", strategy=Greedy())
for section in classifier.classify(load(MARKDOWN)).sections.values():
    print(f"{section.title}: {' > '.join(section.best.labels)} ({section.best.score:.2f})")
