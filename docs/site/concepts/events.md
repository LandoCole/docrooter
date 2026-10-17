# Events

Every public entry point (`load`, `load_pdf`, `classify_document`,
`Classifier.classify_documents` and `normalize_sections`) takes an optional `on_event` callback.
It receives an `Event` for each stage as it starts, progresses and ends, with the seconds since
the run began. Use it for progress bars, logs, timing breakdowns, or a live UI: the
[playground](../guides/playground.md) is built on it.

```python
def show(event: Event) -> None:
    print(f"{event.t:7.3f}s  {event.stage:<10} {event.status:<8} {event.data}")


with pipeline_run(show):
    result = classify_document(doc, builtin_taxonomy("doc_types"), backend="keyword", on_event=show)
    norm = normalize_sections(
        doc, doc_type=result.best.node_ids[-1], backend="keyword", on_event=show
    )
```

```text
  0.002s  view       start    {'mode': 'view'}
  0.002s  view       end      {'tokens': 68}
  0.002s  classify   start    {'mode': 'view'}
  0.002s  classify   end      {'documents': 1, 'type': 'government.government_report', 'confidence': 0.07393347477496882}
  0.003s  normalize  start    {'taxonomy': 'section_types/generic'}
  0.003s  normalize  end      {'sections': 3}
  0.003s  done       end      {}
```

(The keyword backend mistypes this two-section paper; the events are the point here.)

## The event

`Event(stage, status, t, data)`: `status` is `"start"`, `"progress"`, `"end"` or `"error"`;
`t` is seconds since the run started; `data` is a small JSON-safe dict.

| Stage | Reported by | `data` includes |
| --- | --- | --- |
| `extract` | PDF loading; progress every 10 pages | `engine`; `page` and `pages` on progress; `lines` and `pages` at the end |
| `signals` | PDF loading | `bookmarks`, `printed_toc_entries` |
| `headings` | PDF loading | `headings`, the count per `levels` |
| `assemble` | `load` (PDF and Markdown) | `sections` |
| `view` | document-type classification | `tokens` |
| `model_load` | the first use of a model backend | the device chosen on `start`, the device used on `end` (they differ after an MPS-to-CPU fallback) |
| `classify` | document-type classification | `type`, `confidence` |
| `normalize` | `normalize_sections` | `taxonomy`, `sections` |
| `done` | the end of every run | |
| `error` | a run that raised | `message`, the open `stage` |
| `upload` | callers that receive files, such as the playground | |

## Runs

Each call is a **run**: it starts a clock, reports its stages and ends with `done`, or with
`error` followed by the re-raised exception. To report several calls as one run with one clock
and one `done`, wrap them in `docrooter.events.pipeline_run(sink)` as above.

- A sink that raises never breaks processing: the exception is logged once and the sink is
  disabled for the rest of the run.
- Without a sink, nothing is built or emitted, so events cost nothing when unused.
- Sinks are called synchronously on the thread that runs the pipeline. To feed an event loop or
  another thread, hand each event over with `loop.call_soon_threadsafe` or a queue. The run does
  not follow work handed to a `ThreadPoolExecutor`; use `asyncio.to_thread` or
  `contextvars.copy_context().run`, which carry it along.

The script is [`examples/events.py`](../../../examples/events.py).
