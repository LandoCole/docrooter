"""Classify sections with any OpenAI-compatible chat model (needs docrooter[llm]).

The server and key come from the environment, never from code:

- ``DOCROOTER_LLM_BASE_URL`` (else ``OPENAI_BASE_URL``; unset means the OpenAI API), for
  example a local server such as ``http://localhost:11434/v1``;
- ``DOCROOTER_LLM_API_KEY`` (else ``OPENAI_API_KEY``); a local server may need none.

Run it with a git-ignored .env file:  uv run --env-file .env python examples/openai_backend.py
The model name is required; pass it as the first argument (default gpt-4o-mini).
"""

import os
import sys

from docrooter import Classifier, Greedy, Taxonomy, load

SETTINGS = ("DOCROOTER_LLM_API_KEY", "OPENAI_API_KEY", "DOCROOTER_LLM_BASE_URL", "OPENAI_BASE_URL")
if not any(os.environ.get(name) for name in SETTINGS):
    sys.exit("set OPENAI_API_KEY, or DOCROOTER_LLM_BASE_URL for a local server (see the docs)")

model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o-mini"

taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {"id": "financial", "label": "Financial", "description": "Accounting and taxes"},
            {"id": "legal", "label": "Legal", "description": "Contracts and legal proceedings"},
        ]
    }
)
doc = load("# Report\n\n## Income taxes\nThe tax rate rose.\n\n## Lawsuit\nThe court ruled.\n")

classifier = Classifier(taxonomy, f"openai:{model}", strategy=Greedy())
for section in classifier.classify(doc).sections.values():
    print(f"{section.title}: {' > '.join(section.best.labels)} ({section.best.score:.2f})")
