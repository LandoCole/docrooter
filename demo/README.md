# doc-rooter live demo

The interactive page behind [landonthornton.com/docrooter/](https://landonthornton.com/docrooter/):
pick a rung of the context ladder and an approach, and see exactly what Jev (TypeSafe's decision
model) saw, the type it named with its probabilities, the tag it gave every section, and the
tokens and cost.

- `index.html` is the whole page (no build step). Samples load instantly from `data/`.
- `data/` holds the samples and the benchmark summary, made by `build_samples.py`: each sample
  is classified live with Jev, both approaches at all four rungs (a few cents in all). Samples
  are synthetic financial filings from
  [document-classifier](https://github.com/LandoCole/document-classifier) (`samples/`, MIT;
  invented companies, gold tags in `samples/gold.json`) and public documents from the benchmark
  whose licence allows redistribution (a GAO report, a CC BY 4.0 arXiv paper and CUAD contract,
  a U.S. federal court opinion).
- `api/docrooter_demo.py` is the live upload and the code the samples are built with: a Vercel
  Python function that reads the uploaded file and classifies it with Jev at all four rungs
  (`TYPESAFE_API_KEY` from the environment), limited per visitor to 10 runs a minute and 100 a
  day, PDFs read to 30 pages. It needs `docrooter[jev,pdf]` with PyMuPDF and PyMuPDF4LLM pinned
  to 1.28.2: about 230 MB installed, close to Vercel's 250 MB function limit (PyMuPDF4LLM brings
  onnxruntime and its layout model).

Run it locally (live uploads need `TYPESAFE_API_KEY`; the samples work without it):

```bash
PYTHONPATH=src python demo/serve.py
```

Rebuild the samples after a benchmark run or a change to the catalogue (needs
`TYPESAFE_API_KEY` and, for the public samples, the benchmark PDFs and readings in the cache):

```bash
PYTHONPATH=src python demo/build_samples.py
```

The site copies this folder with its own `tools/sync_docrooter.py`, which swaps in the site's
licensed font and pins the function's `docrooter` dependency to a commit of this repository.
Edit the demo here, not in the site.
