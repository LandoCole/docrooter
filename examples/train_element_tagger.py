"""The element-tagger training chain on a tiny corpus, with no model and no network.

Real runs use ``docrooter corpus build`` and ``docrooter train-elements`` (see the guide
"Train your own element tagger"). This script runs the same stages in Python, torch-free, so
you can see what each one produces: a corpus store of fund-document elements, a keyword teacher
standing in for the LLM, the labels it writes, and the confident training set that would train
the student. Training itself needs the ``models`` extra, so it stops there.
"""

import re
import tempfile
import zlib

import numpy as np

from docrooter import Taxonomy
from docrooter.backends.base import Scores
from docrooter.corpus import Corpus, CorpusElement, ManifestRecord
from docrooter.teach.base import softmax
from docrooter.teach.dataset import build_training_set, to_dataset
from docrooter.teach.label import SampleSpec, label_corpus

WORDS = re.compile(r"[a-z]+")
STOP = {"the", "and", "of", "a", "an", "to", "in", "for", "is", "are", "on", "or", "as", "by"}
STOP |= {"with", "that", "this", "be", "it", "its", "at", "from", "not", "may", "can", "any"}
DIMS = 256

# Elements of a fund fact sheet, by the topic a reader would give them.
PARAGRAPHS = {
    "fees": [
        "The management fee is 0.45% a year and the expense ratio is 0.62%.",
        "Fees and expenses: no sales charge applies and the total expense ratio is 0.71%.",
        "Annual fund operating expenses include a management fee and other expenses.",
        "Investors pay a redemption fee of 1% on shares sold within 30 days.",
    ],
    "risks": [
        "Principal risks: the fund can lose money, and bond prices fall as interest rates rise.",
        "Market risk and credit risk may reduce the value of your investment.",
        "Liquidity risk: the fund may be unable to sell holdings quickly without a loss.",
        "Currency risk can cause losses when foreign currencies weaken against the dollar.",
    ],
    "performance": [
        "Performance: the fund returned 7.2% over one year and 5.1% over five years.",
        "Past performance does not guarantee future returns; annualised returns are shown.",
        "Cumulative performance against the benchmark index, as of 30 June.",
        "The fund's total return for the year trailed its benchmark by 0.8%.",
    ],
    "objective": [
        "Investment objective: the fund seeks long-term capital growth.",
        "The fund's objective is to provide income while preserving capital.",
        "The investment objective is to track the total return of the index.",
        "The fund aims for capital appreciation over the long term.",
    ],
}


def stems(text: str) -> list[str]:
    """Lower-case words cut to six letters, so "fees" and "fee" meet."""
    return [w.rstrip("s")[:6] for w in WORDS.findall(text.lower()) if w not in STOP]


class KeywordEmbedder:
    """Hashes word stems into a unit vector. Stands in for UltraBERT-Embed: it has what the
    labelling and training stages call (``score``, ``encode``, ``close``)."""

    name = "embedding:keyword"
    temperature = 0.05

    def encode(self, texts, *, prompt=""):
        rows = np.zeros((len(texts), DIMS), dtype=np.float32)
        for i, text in enumerate(texts):
            for stem in stems(text):
                rows[i, zlib.crc32(stem.encode()) % DIMS] += 1.0
            rows[i] /= max(float(np.linalg.norm(rows[i])), 1e-9)
        return rows

    def score(self, texts, candidates, *, instruction=None, multi_label=False):
        queries = self.encode(list(texts))
        options = self.encode([candidate.text for candidate in candidates])
        return Scores(queries @ options.T, "similarity")

    def close(self):
        pass


class KeywordTeacher:
    """Chooses the option whose text shares most word stems with the element. Stands in for
    the LLM teacher; it implements the same ``TeacherEngine`` protocol."""

    name = "keyword:overlap@1"

    def choose(self, items, *, instruction, prefix_key, prefix_options=None):
        out = []
        for item in items:
            words = set(stems(item.text))
            scores = [
                len(words & set(stems(text))) / (1 + len(stems(text))) ** 0.5
                for _, text in item.options
            ]
            probs = softmax([6.0 * score for score in scores])
            out.append({tag: p for (tag, _), p in zip(item.options, probs, strict=True)})
        return out

    def yes_no(self, items, *, question):
        shared = [len(set(stems(item.text)) & set(stems(question))) for item in items]
        return [0.9 if n >= 2 else 0.1 for n in shared]

    def close(self):
        pass


def fund_topics() -> Taxonomy:
    """A cut-down ``fund_document`` family: the four topics this corpus covers, plus ``other``.
    The built-in ``element_topics`` has ten fund leaves and five more families."""
    return Taxonomy.from_dict(
        {
            "nodes": [
                {"id": "fund_document", "label": "Fund document"},
                {
                    "id": "fund_document.investment_objective",
                    "label": "Investment objective",
                    "description": "The fund's objective or aim: growth, income, tracking.",
                    "parent_id": "fund_document",
                },
                {
                    "id": "fund_document.fees_and_expenses",
                    "label": "Fees and expenses",
                    "description": "Fees, charges and expense ratios paid by investors.",
                    "parent_id": "fund_document",
                },
                {
                    "id": "fund_document.principal_risks",
                    "label": "Principal risks",
                    "description": "Risks of losing money: market, credit, liquidity.",
                    "parent_id": "fund_document",
                },
                {
                    "id": "fund_document.performance",
                    "label": "Performance",
                    "description": "Returns of the fund against its benchmark, past performance.",
                    "parent_id": "fund_document",
                },
                {
                    "id": "fund_document.other",
                    "label": "Other",
                    "description": "Anything else.",
                    "parent_id": "fund_document",
                },
            ]
        }
    )


def tiny_corpus(directory: str) -> Corpus:
    """Eight fact-sheet documents of 8 elements each, the topics in turn."""
    corpus = Corpus(directory)
    topics = list(PARAGRAPHS)
    for n in range(8):
        elements = []
        for i in range(8):
            topic = topics[(n + i) % len(topics)]
            text = PARAGRAPHS[topic][(n + i // 2) % 4]
            elements.append(
                CorpusElement(
                    doc_id=f"fact_sheet_{n}",
                    element_id=f"e{i}",
                    kind="paragraph",
                    text=text,
                    heading_path=[topic.title()],
                    section_title=topic.title(),
                    page=1,
                )
            )
        record = ManifestRecord(
            id=f"fact_sheet_{n}",
            sha256=f"{n + 1:064x}",
            split="train",
            doc_type_hint="financial.fact_sheet",
        )
        corpus.add_document(record, elements, [1])
    return corpus


def main() -> None:
    topics = fund_topics()
    with tempfile.TemporaryDirectory() as directory:
        corpus = tiny_corpus(directory)
        embedder, teacher = KeywordEmbedder(), KeywordTeacher()
        report = label_corpus(
            corpus,
            teacher,
            embedder,
            sample=SampleSpec(per_doc=8, target=64, floor=0),
            topics=topics,
        )
        print(
            f"{teacher.name}: {report.topic_labelled} topics, {report.function_labelled} functions"
        )
        # the confident set: term mining, centroids, the similarity gap, the keep rule
        enriched, training_set = build_training_set(
            corpus, corpus.labels("teacher"), topics, embedder, min_per_class=3
        )
        for leaf, counts in training_set.classes.items():
            state = "trained" if counts.trained else "left untrained"
            print(f"{leaf}: {counts.labelled} labelled, {counts.kept} kept, {state}")
        keywords = enriched.nodes["fund_document.fees_and_expenses"].keywords
        print("fees keywords:", ", ".join(keywords[:3]))
        dataset = to_dataset(training_set, corpus)
        print(f"{len(dataset)} training examples; the first:")
        print(dataset.examples[0].path, dataset.examples[0].text.splitlines()[-1][:50])
        print("training itself needs the models extra: docrooter train-elements")


if __name__ == "__main__":
    main()
