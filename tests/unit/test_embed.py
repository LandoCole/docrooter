"""``embed()`` and ``EmbeddingSet`` (spec §9) with a fake embedding backend: no model is loaded."""

import json
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from docrooter import Document, RouterConfig
from docrooter.backends import registry as backend_registry
from docrooter.chunking import chunk
from docrooter.elements.models import Element, TableData
from docrooter.embed import EmbeddingSet, embed
from docrooter.errors import BackendError, DocrooterError
from docrooter.rooter.analysis import Analysis


class FakeEmbedding:
    """Deterministic unit vectors from the text; records every ``encode`` call."""

    name = "embedding:fake/embed"
    model_id = "fake/embed"
    recipe = SimpleNamespace(
        revision="0123abcd", query_prompt="Instruct: {instruction}\nQuery: ", label_prompt="Label: "
    )

    def __init__(self, dim=4):
        self.dim = dim
        self.calls: list[tuple[list[str], str]] = []
        self.closed = 0

    def encode(self, texts, *, prompt):
        self.calls.append((list(texts), prompt))
        rows = []
        for text in texts:
            vec = np.random.default_rng(zlib.crc32(text.encode())).normal(size=self.dim)
            rows.append((vec / np.linalg.norm(vec)).astype(np.float32))
        return np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)

    def close(self):
        self.closed += 1


def _doc() -> Document:
    doc = Document.from_sections(
        [{"title": "A", "level": 1}, {"title": "B", "level": 2}, {"title": "C", "level": 1}]
    )
    table = TableData(rows=[["Region", "Revenue"], ["North", "10"]], header_rows=1)
    items = [
        ("heading", "A", "s1", {"level": 1}),
        ("paragraph", "Intro of A.", "s1", {}),
        ("heading", "B", "s1.1", {"level": 2}),
        ("paragraph", "First of B.", "s1.1", {}),
        ("table", "Region Revenue North 10", "s1.1", {"table": table}),
        ("caption", "Table 1. Revenue", "s1.1", {"caption_for": "e4"}),
        ("page_footer", "Page 3", "s1.1", {}),
        ("contents", "Item 1 .... 3", "s2", {}),
        ("paragraph", "Body of C.", "s2", {}),
    ]
    doc.elements = [
        Element(id=f"e{i}", kind=kind, text=text, boxes=[], order=i, section_id=sid, **extra)
        for i, (kind, text, sid, extra) in enumerate(items)
    ]
    return doc


def test_embed_records_model_and_revision():
    backend = FakeEmbedding(dim=6)
    result = embed(_doc(), backend)
    assert isinstance(result, EmbeddingSet)
    assert (result.model, result.revision, result.dim) == ("fake/embed", "0123abcd", 6)
    assert result.vectors.shape == (len(result.ids), 6)
    # one call, with the document-side prompt: documents are not instructed like queries
    ((texts, prompt),) = backend.calls
    assert prompt == "" and len(texts) == len(result.ids)


def test_embed_element_unit_texts_are_path_plus_text():
    backend = FakeEmbedding()
    result = embed(_doc(), backend)
    # furniture and contents are skipped by default; the others are in reading order
    assert result.ids == ["e0", "e1", "e2", "e3", "e4", "e5", "e8"]
    texts = backend.calls[0][0]
    assert texts[0] == "A"  # a heading is its own path
    assert texts[1] == "A\n\nIntro of A."
    assert texts[2] == "A > B"
    assert texts[3] == "A > B\n\nFirst of B."
    assert texts[4].startswith("A > B\n\nTable: Table 1. Revenue; 2×2\n| Region | Revenue |")
    assert texts[5] == "A > B\n\nTable 1. Revenue"
    assert texts[6] == "C\n\nBody of C."


def test_embed_the_chunk_unit_embeds_the_chunk_texts():
    backend = FakeEmbedding()
    doc = _doc()
    result = embed(doc, backend, unit="chunk")
    chunks = chunk(doc)
    assert result.ids == [c.id for c in chunks]
    assert backend.calls[0][0] == [c.text for c in chunks]


def test_float16_default():
    result = embed(_doc(), FakeEmbedding())
    assert result.dtype == "float16" and result.vectors.dtype == np.float16
    wide = embed(_doc(), FakeEmbedding(), dtype="float32")
    assert wide.dtype == "float32" and wide.vectors.dtype == np.float32
    assert np.allclose(result.vectors.astype(np.float32), wide.vectors, atol=1e-3)
    with pytest.raises(ValueError, match="dtype"):
        embed(_doc(), FakeEmbedding(), dtype="float64")


def test_npz_round_trip(tmp_path):
    result = embed(_doc(), FakeEmbedding(dim=5))
    path = tmp_path / "vectors.npz"
    result.to_npz(path)
    loaded = EmbeddingSet.from_npz(path)
    assert loaded == result
    assert loaded.vectors.dtype == np.float16 and loaded.ids == result.ids
    assert (loaded.model, loaded.revision, loaded.dim) == ("fake/embed", "0123abcd", 5)
    # no pickle is needed to read it, and a missing revision survives
    with np.load(path, allow_pickle=False) as data:
        assert {"vectors", "ids", "meta"} <= set(data.files)
    bare = EmbeddingSet(model="m", revision=None, dim=2, dtype="float32", ids=["x"],
                        vectors=np.ones((1, 2), dtype=np.float32))  # fmt: skip
    bare.to_npz(str(tmp_path / "bare.npz"))
    assert EmbeddingSet.from_npz(tmp_path / "bare.npz") == bare


def test_npz_round_trip_of_an_empty_set(tmp_path):
    empty = embed(Document.from_sections([{"title": "A", "level": 1}]), FakeEmbedding())
    assert empty.ids == [] and empty.vectors.shape[0] == 0
    empty.to_npz(tmp_path / "e.npz")
    assert EmbeddingSet.from_npz(tmp_path / "e.npz") == empty


def test_from_npz_rejects_other_files(tmp_path):
    np.savez(tmp_path / "other.npz", x=np.zeros(3))
    with pytest.raises(DocrooterError, match="EmbeddingSet"):
        EmbeddingSet.from_npz(tmp_path / "other.npz")


def test_embed_kinds_filter():
    backend = FakeEmbedding()
    result = embed(_doc(), backend, kinds=["paragraph", "caption"])
    assert result.ids == ["e1", "e3", "e5", "e8"]
    assert backend.calls[0][0][0] == "A\n\nIntro of A."
    # naming a kind overrides the default skip of furniture
    footer = embed(_doc(), FakeEmbedding(), kinds={"page_footer"})
    assert footer.ids == ["e6"]
    assert embed(_doc(), FakeEmbedding(), kinds=[]).ids == []
    with pytest.raises(ValueError, match="unknown element kind.*'paragraf'"):
        embed(_doc(), FakeEmbedding(), kinds=["paragraf"])
    with pytest.raises(ValueError, match="unit='element'"):
        embed(_doc(), FakeEmbedding(), unit="chunk", kinds=["paragraph"])


def test_embed_bad_arguments():
    with pytest.raises(ValueError, match="unit"):
        embed(_doc(), FakeEmbedding(), unit="sentence")
    doc = Document.from_sections([{"title": "A", "level": 1, "text": "body text"}])
    with pytest.raises(DocrooterError, match="elements=True"):
        embed(doc, FakeEmbedding())


def test_embed_accepts_an_analysis():
    doc = _doc()
    analysis = Analysis(document=doc, config=RouterConfig(profile="fast"))
    assert embed(analysis, FakeEmbedding()) == embed(doc, FakeEmbedding())


def test_embed_closes_only_a_backend_it_created(monkeypatch):
    made = []

    def factory(model="", *, revision=None, **kwargs):
        made.append(FakeEmbedding())
        return made[-1]

    monkeypatch.setattr(
        backend_registry, "_KINDS", {**backend_registry._KINDS, "fake-embed": factory}
    )
    embed(_doc(), "fake-embed")
    assert len(made) == 1 and made[0].closed == 1
    mine = FakeEmbedding()
    embed(_doc(), mine)
    assert mine.closed == 0


def test_embed_rejects_a_backend_that_cannot_encode():
    with pytest.raises(BackendError, match="cannot embed"):
        embed(_doc(), "keyword")
    with pytest.raises(BackendError, match="cannot embed"):
        embed(_doc(), SimpleNamespace(name="no-encode"))


def test_embed_rejects_a_backend_with_the_wrong_row_count():
    class Short(FakeEmbedding):
        def encode(self, texts, *, prompt):
            return super().encode(texts[:-1], prompt=prompt)

    with pytest.raises(BackendError, match="one row per text"):
        embed(_doc(), Short())


def test_a_backend_without_a_recipe_falls_back_to_its_name():
    class Plain(FakeEmbedding):
        model_id = None
        recipe = None

    result = embed(_doc(), Plain())
    assert (result.model, result.revision) == ("embedding:fake/embed", None)


def test_embedding_set_json_round_trip():
    result = embed(_doc(), FakeEmbedding())
    again = EmbeddingSet.model_validate_json(result.model_dump_json())
    assert again == result and again.vectors.dtype == np.float16
    data = json.loads(result.model_dump_json())
    assert data["model"] == "fake/embed" and len(data["vectors"]) == len(result.ids)


def test_analysis_json_has_vectors_only_when_asked():
    doc = _doc()
    analysis = Analysis(
        document=doc, config=RouterConfig(profile="fast"), embeddings=embed(doc, FakeEmbedding())
    )
    assert "embeddings" not in json.loads(analysis.to_json())
    asked = json.loads(analysis.to_json(include_embeddings=True))
    assert asked["embeddings"]["ids"] == analysis.embeddings.ids
    back = Analysis.model_validate_json(analysis.to_json(include_embeddings=True))
    assert back.embeddings == analysis.embeddings


def test_embedding_set_validates_its_shape():
    ok = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="ids"):
        EmbeddingSet(model="m", revision=None, dim=3, dtype="float32", ids=["a"], vectors=ok)
    with pytest.raises(ValueError, match="dim"):
        EmbeddingSet(model="m", revision=None, dim=4, dtype="float32", ids=["a", "b"], vectors=ok)
