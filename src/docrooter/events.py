"""Pipeline events: an optional ``on_event`` callback that reports each processing stage.

The public entry points (:func:`~docrooter.load`, :func:`~docrooter.load_pdf`,
:func:`~docrooter.classify_document`, :meth:`Classifier.classify_documents
<docrooter.Classifier.classify_documents>` and :func:`~docrooter.normalize_sections`) take
``on_event``. Each call is a *run*: it starts a clock, reports its stages as :class:`Event`
objects, ends with a ``done`` event, or with an ``error`` event followed by the re-raised
exception.

Stages (``status`` is ``"start"``, ``"progress"`` or ``"end"``): ``extract`` (progress every
10 pages), ``signals``, ``headings``, ``assemble``, ``view``, ``model_load``, ``classify`` and
``normalize``. ``upload`` is for a caller that receives files, such as the playground.

A caller that chains several calls (load, classify, normalize) wraps them in :func:`pipeline_run`
so that they share one clock and one ``done``. Internal code reports through a context variable,
so no function below the public ones takes a sink.

A sink that raises never breaks processing: the exception is logged once and the sink is
disabled for the rest of the run. Without a sink nothing is built or emitted.

Threads and async code. Sinks are called synchronously on the thread that runs the pipeline,
so a sink that feeds an event loop or another thread must hand each event over with
``loop.call_soon_threadsafe`` or a queue. The run lives in a :class:`~contextvars.ContextVar`:
a new thread starts without it, and it does not cross ``loop.run_in_executor`` or
``ThreadPoolExecutor.submit``. Enter :func:`pipeline_run` (or pass ``on_event``) inside the
worker, or use ``asyncio.to_thread`` / ``contextvars.copy_context().run``, which carry the
context along. Runs in different threads are independent.

``model_load`` reports the device chosen when loading began on ``start`` and the device
actually used on ``end``; they differ after an out-of-memory fallback to the CPU. A stage still
open when a run finishes normally is closed with ``{"aborted": True}`` just before ``done``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["Event", "EventSink", "finish", "pipeline_run", "start"]

logger = logging.getLogger(__name__)

Status = Literal["start", "progress", "end", "error"]
ERROR_MESSAGE_CHARS = 500


class Event(BaseModel):
    """One pipeline event: ``stage``, ``status``, ``t`` (seconds since the run started) and a
    small JSON-safe ``data`` dict."""

    stage: str
    status: Status
    t: float
    data: dict[str, Any] = Field(default_factory=dict)


EventSink = Callable[[Event], None]


class _Run:
    """The sink and clock of the running pipeline call."""

    def __init__(self, sink: EventSink, started: float) -> None:
        self.origin = sink  # the sink the caller passed, kept to recognise nested calls
        self.sink: EventSink | None = sink  # None once it has raised
        self.started = started
        self.open: list[str] = []  # started stages that have not ended

    def emit(self, stage: str, status: Status, data: dict[str, Any] | None = None) -> None:
        sink = self.sink
        if sink is None:
            return
        event = Event(
            stage=stage, status=status, t=time.perf_counter() - self.started, data=data or {}
        )
        try:
            sink(event)
        except Exception:
            self.sink = None
            logger.warning(
                "on_event sink raised; it is disabled for the rest of this run", exc_info=True
            )


_current: ContextVar[_Run | None] = ContextVar("docrooter_events_run", default=None)


class _Stage:
    """A started stage; ``progress`` and ``end`` report on it."""

    __slots__ = ("_name", "_run")
    active = True

    def __init__(self, run: _Run, name: str) -> None:
        self._run = run
        self._name = name

    def progress(self, **data: Any) -> None:
        self._run.emit(self._name, "progress", data)

    def end(self, **data: Any) -> None:
        if self._name in self._run.open:
            self._run.open.remove(self._name)
            self._run.emit(self._name, "end", data)


class _NoStage:
    """The stage handle when nothing listens. ``active`` lets a caller skip building data."""

    __slots__ = ()
    active = False

    def progress(self, **data: Any) -> None:
        pass

    def end(self, **data: Any) -> None:
        pass


_NO_STAGE = _NoStage()
Stage = _Stage | _NoStage


def start(stage: str, **data: Any) -> Stage:
    """Report that ``stage`` started and return its handle (``progress``/``end``).

    Without a sink in the current run this does nothing and returns an inactive handle.
    """
    run = _current.get()
    if run is None or run.sink is None:
        return _NO_STAGE
    run.open.append(stage)
    run.emit(stage, "start", data)
    return _Stage(run, stage)


def finish(stage: str, **data: Any) -> None:
    """Report the end of ``stage`` if the current run started it (for a stage that starts and
    ends in different functions); otherwise do nothing."""
    run = _current.get()
    if run is not None and stage in run.open:
        _Stage(run, stage).end(**data)


def _is_cancellation(exc: BaseException) -> bool:
    import asyncio

    return isinstance(exc, asyncio.CancelledError)


@contextmanager
def pipeline_run(on_event: EventSink | None) -> Iterator[None]:
    """Run a block as one pipeline run reporting to ``on_event``.

    The run starts a clock (reusing the clock of an enclosing run), reports ``done`` when the
    block finishes and ``error`` (``{"message": ...}``, plus the open ``stage`` if any) when it
    raises (any ``BaseException``, so an interrupt or ``asyncio.CancelledError`` too; the latter
    adds ``"cancelled": True``); the exception propagates. Stages still open at a normal finish
    are ended with ``{"aborted": True}`` before ``done``. Inside an enclosing run with an equal
    sink (``==``, so a bound method such as ``job.push`` matches itself), or with
    ``on_event=None``, the block joins that run and reports nothing extra. The public functions
    do this themselves; call it to share one clock and one ``done`` across several of them.

    Sinks run on the calling thread, and the run does not follow work handed to another thread
    or executor; see the module docstring.
    """
    parent = _current.get()
    if on_event is None or (parent is not None and parent.origin == on_event):
        yield
        return
    run = _Run(on_event, parent.started if parent is not None else time.perf_counter())
    token = _current.set(run)
    try:
        yield
    except BaseException as exc:
        data: dict[str, Any] = {"message": str(exc)[:ERROR_MESSAGE_CHARS]}
        if run.open:
            data["stage"] = run.open[-1]
        if _is_cancellation(exc):
            data["cancelled"] = True
        run.emit("error", "error", data)
        raise
    else:
        while run.open:
            run.emit(run.open.pop(), "end", {"aborted": True})
        run.emit("done", "end")
    finally:
        _current.reset(token)
