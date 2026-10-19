"""DocLayNet sample for the element benchmark (v0.4 task E11a; spec §10.1): fetch and sample.

Source: the Hugging Face dataset ``docling-project/DocLayNet-v1.2`` at a pinned revision. It is
DocLayNet (Pfitzmann et al., KDD 2022; https://github.com/DS4SD/DocLayNet) re-packed as parquet
with each page's born-digital single-page PDF in a binary column, the gold boxes (COCO
``[x, y, w, h]`` in 1025 x 1025 page-image pixels), the class ids and the PDF text cells inside
each box. Licence: CDLA-Permissive-1.0 (https://cdla.io/permissive-1-0/), as for DocLayNet; the
pages stay in the git-ignored cache and are not redistributed.

The parquet shards are sorted by document category, so the sample needs every shard of a
split: each is downloaded, checked against its pinned SHA-256 (the LFS object id at the pinned
revision), read for metadata, and deleted once its sampled pages are written.

``.venv/bin/python -I benchmarks/doclaynet.py prepare --split test`` (then ``--split train``)

writes, under ``benchmarks/.cache/doclaynet/`` (git-ignored):
- ``pages/<split>/<page_hash>.pdf``: the sampled single-page PDFs;
- ``<split>.jsonl``: one record per sampled page: ids, category, the PDF's SHA-256 and the gold
  boxes with their class and text cells.

Sample: ``SAMPLE_SIZE`` pages per split, stratified over the six categories with an equal share
each (``SAMPLE_SIZE`` / 6, the first categories in name order taking the remainder), drawn with
``random.Random(f"{SEED}:{split}")`` from the rows of each category in shard order. Train and
test are DocLayNet's own splits, which share no documents.

Downloads are untrusted data: this module only reads them with pyarrow (run it with ``-I``), and
it imports nothing from the repository, so it runs as a plain script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = "docling-project/DocLayNet-v1.2"
REVISION = "0daf93102e2efce76c3e11a274a5e0d0969391d3"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{{name}}"
LICENCE = "CDLA-Permissive-1.0"
LICENCE_URL = "https://cdla.io/permissive-1-0/"
CITATION = "Pfitzmann et al., DocLayNet, KDD 2022, doi:10.1145/3534678.3539043"

CACHE = Path(__file__).resolve().parent / ".cache" / "doclaynet"
SEED = 20261018
SAMPLE_SIZE = 400
CATEGORIES = (
    "financial_reports", "government_tenders", "laws_and_regulations", "manuals", "patents",
    "scientific_articles",
)  # fmt: skip
CLASS_NAMES = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item", 5: "Page-footer",
    6: "Page-header", 7: "Picture", 8: "Section-header", 9: "Table", 10: "Text", 11: "Title",
}  # fmt: skip

# Shard name -> (bytes, SHA-256), from the dataset tree at REVISION.
SHARDS: dict[str, tuple[int, str]] = {
    "test-00000-of-00006.parquet": (
        515258598,
        "b1b9a9d137dd7bcf95a01d8aca69bdba83e4221a8f72a5b0ee24e053c484b0c6",
    ),
    "test-00001-of-00006.parquet": (
        433755114,
        "0e95ce9fe6ce0475cbce89b4857ee907739e4cd36a70da0a3d758b1636cd14bf",
    ),
    "test-00002-of-00006.parquet": (
        391008800,
        "2a22ed140abc9ef0e0b5c6acb6ed10d1a8420a98a30f1ca99de54df073c90329",
    ),
    "test-00003-of-00006.parquet": (
        458137785,
        "a94b2131b35176324cd7020bebb92eeb03b000ce93f5783ce85c94895600ae93",
    ),
    "test-00004-of-00006.parquet": (
        266503202,
        "509e91dd74cb3d04fb7600834ac64fd1c52f450770dbd5dc26e1a44e828e5ba9",
    ),
    "test-00005-of-00006.parquet": (
        356987377,
        "267dcc7dc20056b44cce7389825d64569a10d7ebec366c72de91b45474924c60",
    ),
    "train-00000-of-00072.parquet": (
        519150134,
        "5aa0eb1e70b586e70c47cc9116c9c6a233a51bac142b47d738a92b507b83a675",
    ),
    "train-00001-of-00072.parquet": (
        519668889,
        "48b5829ea6b5a561c45aa5252e63f45534fc15b4a477d75682c2d7b03e949656",
    ),
    "train-00002-of-00072.parquet": (
        524973937,
        "8d52e5a9ea65cff12e5f6ba79f76a3877059f77760da241d400590654ab52938",
    ),
    "train-00003-of-00072.parquet": (
        510024491,
        "fa1cd9cd57b0e918a0947c81e5296b4bfe1a6fa962441af2272ebfe2435e95da",
    ),
    "train-00004-of-00072.parquet": (
        516594943,
        "7426226b2bcf4aee0c98c47b5f38d983f8962265edda9bbbab6cf57ca7701d39",
    ),
    "train-00005-of-00072.parquet": (
        517375241,
        "6510a04ea5f693aa6f35e8d9429e948de2e2e4ac1fe9db41377ef4f12f494dac",
    ),
    "train-00006-of-00072.parquet": (
        533123629,
        "243499d9c332e93afd9bdb26852bc838c29664499c6bffa42570c5c210fbb946",
    ),
    "train-00007-of-00072.parquet": (
        543781256,
        "479fa7d1397df865f53fdba33beaadc8b70d4f3b7face5fe573047801fc34a65",
    ),
    "train-00008-of-00072.parquet": (
        522971737,
        "cbe654761cda01c539dec36ddd01e6ca34ef68bd43d38fd97a52d6e0b3bbbf06",
    ),
    "train-00009-of-00072.parquet": (
        515439307,
        "a659b2d50d14072c954928fe3e8e346ce06f7ce24ac79bb859ceb078a0bd182a",
    ),
    "train-00010-of-00072.parquet": (
        524740165,
        "fdbe352064b81bf5195a0c20e534a7efba4372f6f93d9434a29036bbfc863a3b",
    ),
    "train-00011-of-00072.parquet": (
        565502188,
        "bc99d3e9a1ff1fe84e5af0f2cb399fd283c189f98d27893c2bc199f2fca7a801",
    ),
    "train-00012-of-00072.parquet": (
        565902409,
        "69ed838c15f8006909d46781eea89a0cc03bd4711b956f930f3657d8346da4dc",
    ),
    "train-00013-of-00072.parquet": (
        532857914,
        "fe71861d96dff788f1d079bade6ce71ea6a5c2a8b4ae05872f3e1dd527f967bc",
    ),
    "train-00014-of-00072.parquet": (
        537986923,
        "838cfcc9623628a95c8bd2cc0da191c91591206382218ebcf94af5aae9a79ac2",
    ),
    "train-00015-of-00072.parquet": (
        532562671,
        "14563d740f3bbeb549ff25a423a97296719c57b763a9e7af54bf2177ce17ab04",
    ),
    "train-00016-of-00072.parquet": (
        552988739,
        "ea12b542a5a980ac42983b56e9f2e5ee00b2d352489dedad37f38d73ddbd3e9f",
    ),
    "train-00017-of-00072.parquet": (
        545152422,
        "1e6a11931e2084f11eae7b4b980e40cd92ef81e454399d9724a356831d7b4096",
    ),
    "train-00018-of-00072.parquet": (
        544892085,
        "1cedecb3d53ced31f074ccc2555c65592ddc6a78e2f3d36697a6ff78c1d0e023",
    ),
    "train-00019-of-00072.parquet": (
        561338697,
        "196f4293cac1ce87cc617a379ab2a319ed5e7eedece8157aa76d74898828d9a0",
    ),
    "train-00020-of-00072.parquet": (
        366553286,
        "7a9b80727c192a14338a8af08f7b9589e51c5f53f87c3842473d593b1ce47027",
    ),
    "train-00021-of-00072.parquet": (
        324789986,
        "d9ffd351cc59b3cdc4e6de955599ae46053c151d13b7e7736c7bb3cbe27e6c0e",
    ),
    "train-00022-of-00072.parquet": (
        321602817,
        "47e337e9c278cba9d3c2ef9f99d1e06b5f12b82de4ad78d7da4dae5109103942",
    ),
    "train-00023-of-00072.parquet": (
        331101135,
        "dff635882b5a384d8ab0a13b85e4719ae4c04087e46c367aa68e575109008ed3",
    ),
    "train-00024-of-00072.parquet": (
        367918557,
        "5983b6f81f81e98a309c0dd1e9af05058dac657aa9d6585f48057df92814c61e",
    ),
    "train-00025-of-00072.parquet": (
        364553767,
        "376391ab02f47282ff352c8b66c935043f9f9b310d3d1a6f4071f9d565c4952e",
    ),
    "train-00026-of-00072.parquet": (
        363228470,
        "d339c6db1bfd96538fcabaa31ea54bfffb0f7ee4ff9b66ad9db9064f68857506",
    ),
    "train-00027-of-00072.parquet": (
        363339909,
        "cfefe1458c3b57f3f5278dc96183ca5e3ac7764fbd54098bfa7d811c76694883",
    ),
    "train-00028-of-00072.parquet": (
        359916438,
        "5d0eb41c1f3edc95a6a2e23190c560c25647de031cf2a812516d34e52f54c9c0",
    ),
    "train-00029-of-00072.parquet": (
        596662612,
        "7a4d829dcaf50b454fa7a4ce8a681f7c452fd5c587fc07d1dc0bf5bb07f2000c",
    ),
    "train-00030-of-00072.parquet": (
        637583095,
        "1e2bd1e65d0bb0c820a2a3f7ec323313d9aa217787293dd1494b5e449184e733",
    ),
    "train-00031-of-00072.parquet": (
        625453987,
        "efc51aa673553bad889253623b36fd1021e776ffbbc6e7f568cf8210205cceed",
    ),
    "train-00032-of-00072.parquet": (
        597227930,
        "f9e172d4bb39e191e11e9a4f58b66f6ecdcc03cbd2dc01b838685885fc1b95b4",
    ),
    "train-00033-of-00072.parquet": (
        475358872,
        "ccd31a41eef3069359e79237c8b1b8f88d37a2b46faeb23a31f50cccbef8df5b",
    ),
    "train-00034-of-00072.parquet": (
        451913698,
        "43ed20e4ee09b5d9010bf178a115324b232ffa897a2b88743b23056081c04afd",
    ),
    "train-00035-of-00072.parquet": (
        451660876,
        "e88ca4744a2250ba45bf7a30eb215b20e0bc5e0dca10524275ccb54525e53c3a",
    ),
    "train-00036-of-00072.parquet": (
        454378313,
        "d927abc6bfb1165036bf4a8fa06bb95f8796e8fddabb55eeaf17dbf7dfb21da5",
    ),
    "train-00037-of-00072.parquet": (
        702999847,
        "2e4704cecc037542d058fe1338550c90e65aea0a5d381baa210f86fd943c860a",
    ),
    "train-00038-of-00072.parquet": (
        771767272,
        "1b33a4803dd749a3b792f116e09419614ca33c06a9a7cd5463f856f05998d3cd",
    ),
    "train-00039-of-00072.parquet": (
        739443576,
        "f6b01cd233244d4e220fa0aa58ade1b218dbb3062a6cb337181ad22bddd435f0",
    ),
    "train-00040-of-00072.parquet": (
        770478076,
        "829eba53ce43d876887a64cbf9bb59ce597b9939d75ebd9e533d5c8b4e030270",
    ),
    "train-00041-of-00072.parquet": (
        754410373,
        "9cb8e782f165adb9df17eec57e775d601823158243f19131740043cd147e36ee",
    ),
    "train-00042-of-00072.parquet": (
        756620836,
        "cce306facb7e011330a2adfe0026b4d048b717386f1c214ee45eabf2b5ccd4ef",
    ),
    "train-00043-of-00072.parquet": (
        583345127,
        "e72e821388828d7ce1d4edccc9d641f66fc4d86699a8684b084495bea6dc31ad",
    ),
    "train-00044-of-00072.parquet": (
        338342817,
        "537eb4a4a49f35104510379755c32b48fc1906119881a400f9fae2026ca5eb9d",
    ),
    "train-00045-of-00072.parquet": (
        550047343,
        "d52b2a2a1ce6ee44b5a9d4c22bd3a3d99047954b1c7600e8b36b79c165e5fb43",
    ),
    "train-00046-of-00072.parquet": (
        510289579,
        "bfb47227005b8006c271116fcd445e19e96a6cf44ca90b78d82b2fc922e4e867",
    ),
    "train-00047-of-00072.parquet": (
        381558072,
        "e2fb74b4c5e3df4f07f2b431671dc6d3ce575307c9baaab7e1da3f2d6165f65e",
    ),
    "train-00048-of-00072.parquet": (
        390191412,
        "2d1b0aea98391b0a09101b6f6eb5af4aca033cbf0c6975721027092112e239f5",
    ),
    "train-00049-of-00072.parquet": (
        383044405,
        "bed978eba0ad70c2603b318915d9b4cb887f3ca7d23f0cd20bdbdeff56556b76",
    ),
    "train-00050-of-00072.parquet": (
        381722807,
        "b0b4b6fbf8ae264485f46ed774cbb96a98cfe2c22f757868bf611aa890f9ce05",
    ),
    "train-00051-of-00072.parquet": (
        380067932,
        "0c5b4261e9e5a526fc6fad84a678cf9e3ced23e2c0359eccc2a7359a5be99655",
    ),
    "train-00052-of-00072.parquet": (
        390248068,
        "b99210ca09979e9693416ceea484ca934ea12a1d55216078caf408cfde08e0fd",
    ),
    "train-00053-of-00072.parquet": (
        382764074,
        "2fa3c13c8b545047487e35b2fa628fade1121e7e45842924c270411bad18210d",
    ),
    "train-00054-of-00072.parquet": (
        391057697,
        "bb74daef9ccbcddafa816018e6c657d281b8bf3f8e3c7aa3c7071fca6e81c52c",
    ),
    "train-00055-of-00072.parquet": (
        380092363,
        "694d4104b6dab8c6122b53728090a47751c3d03e50ea74fea7ea93621fbb64d1",
    ),
    "train-00056-of-00072.parquet": (
        388209915,
        "0b101d926bfe6c01a007fe9800fede5df6a72ea76f70c665b0fbd8c5f54002ac",
    ),
    "train-00057-of-00072.parquet": (
        399426177,
        "9e3dad6ceeb52a53c9c969c6f5d4f599cb2ecf91d534846c7def00ef66d7b89d",
    ),
    "train-00058-of-00072.parquet": (
        403641047,
        "2a21f163dce6218382e84df31c8e7a0a729d609c2de05aa9174112d7020bdc2d",
    ),
    "train-00059-of-00072.parquet": (
        296794996,
        "61655496dc74846fd38a3e1a40219556b2ad40bf5e8ddbc286d1cd867348f1c0",
    ),
    "train-00060-of-00072.parquet": (
        289954194,
        "788fc9297237716dbd24ce7051e303a4f99fd14d11c17b84fdf930661630887e",
    ),
    "train-00061-of-00072.parquet": (
        478549244,
        "15c60b6accd5cbb16a0c02eae672f25b469f3c5b726f2630210cf135356ab157",
    ),
    "train-00062-of-00072.parquet": (
        458085959,
        "72e61c78690e43510d7d8091634cc2fc961b4572307e6c81135836bd660cad7c",
    ),
    "train-00063-of-00072.parquet": (
        367359937,
        "5da19e2311fb51b431fbe3ea7510276d70f1003559ee2973baaf140a2edcaa41",
    ),
    "train-00064-of-00072.parquet": (
        392029788,
        "bfb8c72e543cee7baf033d050b4c16c405f1076903d2d53fff99b2b84dca55b8",
    ),
    "train-00065-of-00072.parquet": (
        415663158,
        "579d72bcd89a0de515c3485862166cae5c8e7d0cb2534aaf843bbc70e9913197",
    ),
    "train-00066-of-00072.parquet": (
        403874293,
        "a8daeb37008b5ac0c23e8baccb9c9f5ee41ad6d44c7d9efb5954642a54d889ff",
    ),
    "train-00067-of-00072.parquet": (
        413201491,
        "aa0aae32a728589297694feed9d582cde39a0e877d9f88928314674721885722",
    ),
    "train-00068-of-00072.parquet": (
        420924736,
        "3fb060c7e9470815252ce28d632d047e47658105c9ce752d14779ff4279d3fe0",
    ),
    "train-00069-of-00072.parquet": (
        388975931,
        "ff51fab8669c17c13732eb475f6dd9805562a8348338a345f26b150ed182243b",
    ),
    "train-00070-of-00072.parquet": (
        388657260,
        "d71ecf62c982247edd61b4eb3de5fb8e6225f2375781978c7591ad416d48e7ae",
    ),
    "train-00071-of-00072.parquet": (
        396358736,
        "8cdd4d03915f8d4768babc2efb5b0b6816430a10d5668dad42104f1d16ba7b2a",
    ),
}


def shards(split: str) -> list[str]:
    return sorted(name for name in SHARDS if name.startswith(f"{split}-"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, raw: Path) -> Path:
    """Download one shard into ``raw`` (resuming nothing) and check its size and SHA-256."""
    size, expected = SHARDS[name]
    path = raw / name
    if not (path.exists() and path.stat().st_size == size and sha256(path) == expected):
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(URL.format(name=name), timeout=120) as response:  # noqa: S310
            with tmp.open("wb") as f:
                while block := response.read(1 << 20):
                    f.write(block)
        tmp.replace(path)
    got = sha256(path)
    if path.stat().st_size != size or got != expected:
        path.unlink()
        raise SystemExit(f"{name}: checksum mismatch (got {got}, expected {expected})")
    return path


def allocation(total: int) -> dict[str, int]:
    """Pages per category: an equal share, the first categories taking the remainder."""
    base, extra = divmod(total, len(CATEGORIES))
    return {c: base + (1 if k < extra else 0) for k, c in enumerate(CATEGORIES)}


def choose(rows: list[tuple[str, int, str]], split: str, total: int) -> set[tuple[str, int]]:
    """The sampled ``(shard, row)`` pairs from ``(shard, row, category)`` rows."""
    by_category: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, row, category in sorted(rows):
        by_category[category].append((name, row))
    rng = random.Random(f"{SEED}:{split}")
    chosen: set[tuple[str, int]] = set()
    for category, k in allocation(total).items():
        pool = by_category.get(category, [])
        chosen.update(rng.sample(pool, min(k, len(pool))))
    return chosen


def record(name: str, row: int, data: dict[str, Any], pdf_sha: str) -> dict[str, Any]:
    meta = data["metadata"]
    gold = []
    for bbox, cls, cells in zip(data["bboxes"], data["category_id"], data["pdf_cells"],
                                strict=True):  # fmt: skip
        gold.append({
            "cls": CLASS_NAMES[cls], "bbox": [float(v) for v in bbox],
            "cells": [{"bbox": [float(v) for v in c["bbox"]], "text": c["text"]}
                      for c in cells or []],
        })  # fmt: skip
    return {
        "id": meta["page_hash"], "category": meta["doc_category"],
        "collection": meta["collection"], "original_filename": meta["original_filename"],
        "page_no": meta["page_no"], "image_id": meta["image_id"],
        "coco": [meta["coco_width"], meta["coco_height"]],
        "original": [meta["original_width"], meta["original_height"]],
        "shard": name, "row": row, "pdf_sha256": pdf_sha, "gold": gold,
    }  # fmt: skip


def prepare(split: str, total: int = SAMPLE_SIZE, keep: bool = False) -> Path:
    """Download every shard of ``split``, sample, write the pages and the manifest, and delete
    the shards (unless ``keep``)."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    raw = CACHE / "raw"
    out = CACHE / "pages" / split
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    names = shards(split)
    rows: list[tuple[str, int, str]] = []
    for name in names:
        path = download(name, raw)
        meta = pq.read_table(path, columns=["metadata"]).column("metadata").to_pylist()
        rows.extend((name, k, m["doc_category"]) for k, m in enumerate(meta))
        print(f"{name}: {len(meta)} pages", flush=True)
    chosen = choose(rows, split, total)
    records = []
    for name in names:
        wanted = sorted(row for shard, row in chosen if shard == name)
        if wanted:
            table = pq.read_table(
                raw / name, columns=["bboxes", "category_id", "pdf_cells", "metadata", "pdf"]
            ).take(wanted)
            for row, data in zip(wanted, table.to_pylist(), strict=True):
                pdf = data["pdf"]
                pdf_sha = hashlib.sha256(pdf).hexdigest()
                (out / f"{data['metadata']['page_hash']}.pdf").write_bytes(pdf)
                records.append(record(name, row, data, pdf_sha))
        if not keep:
            (raw / name).unlink()
    records.sort(key=lambda r: (r["category"], r["shard"], r["row"]))
    manifest = CACHE / f"{split}.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    print(f"{split}: {len(records)} pages -> {manifest} (sha256 {sha256(manifest)})")
    return manifest


def load_manifest(split: str) -> list[dict[str, Any]]:
    """The sampled pages of ``split`` (see ``prepare``), in category order."""
    path = CACHE / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found: run benchmarks/doclaynet.py prepare --split {split}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pdf_path(split: str, page_id: str) -> Path:
    return CACHE / "pages" / split / f"{page_id}.pdf"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch and sample DocLayNet pages")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="download, sample, extract, delete the shards")
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--size", type=int, default=SAMPLE_SIZE)
    p.add_argument("--keep", action="store_true", help="keep the downloaded shards")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.split, args.size, args.keep)


if __name__ == "__main__":
    main(sys.argv[1:])
