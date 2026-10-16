// docrooter playground: upload a document, follow its pipeline events over SSE, show results.
// No framework and no build step. Every string from the server goes into the DOM as text.

const $ = (id) => document.getElementById(id);

const STAGES = {
  upload: { name: "Upload", fam: "io" },
  extract: { name: "Read pages", fam: "doc" },
  signals: { name: "Page signals", fam: "doc" },
  headings: { name: "Find headings", fam: "doc" },
  assemble: { name: "Build tree", fam: "doc" },
  view: { name: "Render view", fam: "model" },
  model_load: { name: "Load model", fam: "model" },
  classify: { name: "Classify type", fam: "model" },
  normalize: { name: "Label sections", fam: "model" },
};
const FLAG_NAMES = {
  level_from_bookmark: "level from bookmark",
  low_heading_confidence: "low confidence",
};
const PREFS_KEY = "docrooter.playground.options";

const state = {
  backends: null,
  lastFile: null,
  runId: null,
  source: null, // EventSource of the live run
  rows: [], // every stage row, in order
  latest: new Map(), // stage name -> its latest row (engine "both" reads pages twice)
  t0: null, // performance.now() seconds at pipeline t = 0
  endT: null, // set when the run is over
  frame: 0,
  busy: false,
};

// --- small helpers -----------------------------------------------------------------------

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style") for (const [p, v] of Object.entries(value)) el.style.setProperty(p, v);
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

const num = (n) => Number(n).toLocaleString();
const pct = (p) => `${(p * 100).toFixed(p >= 0.995 || p < 0.1 ? 1 : 0)}%`;

function secs(s) {
  if (s === null || s === undefined) return "";
  if (s < 0.001) return "<1 ms";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(2)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function setStatus(text, isError = false) {
  const el = $("run-status");
  el.textContent = text;
  el.classList.toggle("is-error", isError);
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(await errorText(response));
  return response.json();
}

async function errorText(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* not JSON */
  }
  return `The server answered ${response.status}.`;
}

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem(PREFS_KEY) || "{}");
  } catch {
    return {};
  }
}

function savePrefs() {
  try {
    localStorage.setItem(
      PREFS_KEY,
      JSON.stringify({
        backend: $("backend").value,
        engine: $("engine").value,
        mode: $("mode").value,
        normalize: $("normalize").checked,
      }),
    );
  } catch {
    /* storage unavailable: preferences just are not remembered */
  }
}

// --- options -------------------------------------------------------------------------------

async function loadOptions() {
  const info = await getJSON("/api/backends");
  state.backends = info;
  const prefs = loadPrefs();
  const backend = $("backend");
  backend.replaceChildren(
    ...info.backends.map((b) =>
      h("option", { value: b.spec }, b.installed ? b.label : `${b.label} (not installed)`),
    ),
  );
  const specs = info.backends.map((b) => b.spec);
  backend.value = specs.includes(prefs.backend) ? prefs.backend : info.default;
  const engine = $("engine");
  engine.replaceChildren(...info.engines.map((e) => h("option", { value: e }, e)));
  engine.value = info.engines.includes(prefs.engine) ? prefs.engine : "auto";
  if (info.modes.includes(prefs.mode)) $("mode").value = prefs.mode;
  if (typeof prefs.normalize === "boolean") $("normalize").checked = prefs.normalize;
  describeBackend();
}

function describeBackend() {
  const chosen = state.backends?.backends.find((b) => b.spec === $("backend").value);
  const note = $("backend-note");
  note.replaceChildren();
  if (!chosen) return;
  note.append(chosen.description);
  if (!chosen.installed) {
    const extra = chosen.spec.startsWith("jev:") ? "jev" : "models";
    note.append(" ", h("strong", {}, `Needs pip install 'docrooter[${extra}]'.`));
  }
  if (state.backends.loaded === chosen.spec) note.append(" Already loaded.");
}

// --- upload --------------------------------------------------------------------------------

function setBusy(busy) {
  state.busy = busy;
  $("drop").setAttribute("aria-disabled", String(busy));
  $("rerun").disabled = busy || !state.lastFile;
}

async function submit(file) {
  if (state.busy) return;
  $("form-error").textContent = "";
  const limit = state.backends?.max_upload_bytes ?? 50 * 1024 * 1024;
  if (file.size > limit) {
    $("form-error").textContent = `${file.name} is ${bytes(file.size)}; the limit is ${bytes(limit)}.`;
    return;
  }
  state.lastFile = file;
  savePrefs();
  const form = new FormData();
  form.append("file", file);
  form.append("backend", $("backend").value);
  form.append("engine", $("engine").value);
  form.append("mode", $("mode").value);
  form.append("normalize", $("normalize").checked ? "true" : "false");
  resetRun(file.name);
  setBusy(true);
  setStatus(`Uploading ${bytes(file.size)}…`);
  try {
    const response = await fetch("/api/runs", { method: "POST", body: form });
    if (!response.ok) throw new Error(await errorText(response));
    const { run_id: runId } = await response.json();
    follow(runId);
    refreshHistory();
  } catch (error) {
    setBusy(false);
    setStatus("The upload was refused.", true);
    $("form-error").textContent = error.message;
  }
}

function setupDrop() {
  const drop = $("drop");
  const input = $("file");
  const choose = () => {
    if (!state.busy) input.click();
  };
  drop.addEventListener("click", choose);
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      choose();
    }
  });
  input.addEventListener("change", () => {
    if (input.files.length) submit(input.files[0]);
    input.value = "";
  });
  for (const type of ["dragenter", "dragover"]) {
    drop.addEventListener(type, (event) => {
      event.preventDefault();
      drop.classList.add("is-over");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    drop.addEventListener(type, () => drop.classList.remove("is-over"));
  }
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) submit(file);
  });
  // A file dropped beside the zone should not navigate away from the page.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  $("rerun").addEventListener("click", () => state.lastFile && submit(state.lastFile));
  for (const id of ["backend", "engine", "mode", "normalize"]) $(id).addEventListener("change", savePrefs);
  $("backend").addEventListener("change", describeBackend);
}

// --- timeline ------------------------------------------------------------------------------

function resetRun(title) {
  if (state.source) state.source.close();
  state.source = null;
  state.rows = [];
  state.latest.clear();
  state.t0 = null;
  state.endT = null;
  cancelAnimationFrame(state.frame);
  setBusy(false); // a run left behind keeps going on the server; new uploads queue after it
  $("stages").replaceChildren();
  $("axis").replaceChildren();
  $("timeline-empty").hidden = false;
  $("clock").textContent = "0.00";
  $("run-title").textContent = title;
  $("results").hidden = true;
  document.querySelector(".timeline").classList.remove("is-live");
}

function now() {
  if (state.endT !== null) return state.endT;
  if (state.t0 === null) return 0;
  return performance.now() / 1000 - state.t0;
}

function addRow(event) {
  const meta = STAGES[event.stage] ?? { name: event.stage.replace(/_/g, " "), fam: "io" };
  const bar = h("div", { class: "stage-bar" });
  const dur = h("span", { class: "stage-dur" });
  const detail = h("div", { class: "stage-detail" });
  const el = h(
    "li",
    { class: `stage fam-${meta.fam} is-running` },
    h("span", { class: "stage-name", title: meta.name }, meta.name),
    h("div", { class: "stage-track" }, bar),
    dur,
    detail,
  );
  const row = { stage: event.stage, start: event.t, end: null, el, bar, dur, detail, data: event.data };
  state.rows.push(row);
  state.latest.set(event.stage, row);
  $("stages").append(el);
  $("timeline-empty").hidden = true;
  describe(row, event);
  return row;
}

function endRow(row, event) {
  row.end = event.t;
  row.el.classList.remove("is-running");
  if (event.data?.aborted) row.el.classList.add("is-aborted");
  describe(row, event);
}

// The details shown under a stage's bar, from its latest event.
function describe(row, event) {
  const d = { ...row.data, ...event.data };
  row.data = d;
  const parts = [];
  const status = event.status;
  switch (row.stage) {
    case "upload":
      parts.push(`${d.filename ?? ""} ${d.bytes !== undefined ? bytes(d.bytes) : ""}`.trim());
      if (status === "end" && d.queued_s > 0.05) parts.push(`waited ${secs(d.queued_s)} in the queue`);
      break;
    case "extract":
      if (d.engine) parts.push(d.engine);
      if (status === "progress" && d.pages) {
        const done = Math.min(d.page / d.pages, 1);
        parts.push(
          h("span", {}, `page ${num(d.page)} of ${num(d.pages)} `),
          h("span", { class: "progress", "aria-hidden": "true" }, h("i", { style: { "--p": `${done * 100}%` } })),
        );
      }
      if (status === "end") parts.push(`${num(d.pages)} pages`, `${num(d.lines)} lines`);
      break;
    case "signals":
      if (status === "end") {
        parts.push(`${num(d.bookmarks ?? 0)} bookmarks`, `${num(d.printed_toc_entries ?? 0)} printed TOC entries`);
      }
      break;
    case "headings":
      if (status === "end") {
        const levels = Object.entries(d.levels ?? {}).map(([level, n]) => `level ${level}: ${num(n)}`);
        parts.push(`${num(d.headings)} headings`, ...levels);
      }
      break;
    case "assemble":
      if (status === "end" && d.sections !== undefined) parts.push(`${num(d.sections)} sections`);
      break;
    case "view":
      if (d.mode) parts.push(`from ${d.mode === "view" ? "the whole document" : d.mode === "sections" ? "its sections" : "both"}`);
      if (status === "end") parts.push(`${num(d.tokens)} tokens`);
      break;
    case "model_load":
      if (d.backend) parts.push(d.backend);
      if (d.device) parts.push(`on ${d.device}`);
      break;
    case "classify":
      if (status === "end") {
        if (d.type) parts.push(d.type, `confidence ${pct(d.confidence ?? 0)}`);
        else if ("type" in d) parts.push("no type chosen");
      }
      break;
    case "normalize":
      if (d.taxonomy) parts.push(d.taxonomy);
      if (status === "end" && d.sections !== undefined) parts.push(`${num(d.sections)} sections labelled`);
      break;
    default:
      break;
  }
  if (d.aborted) parts.push("stopped before it finished");
  row.detail.replaceChildren(...parts.filter(Boolean).map((p) => (p instanceof Node ? p : h("span", {}, p))));
}

function errorRow(event) {
  const el = h(
    "li",
    { class: "stage fam-bad is-error" },
    h("span", { class: "stage-name" }, "Error"),
    h("div", { class: "stage-track" }),
    h("span", { class: "stage-dur" }, secs(event.t)),
    h("div", { class: "stage-detail" }, h("span", {}, event.data?.message || "The run failed.")),
  );
  $("stages").append(el);
}

// A round tick step giving at most `count` ticks over `span` seconds.
function niceStep(span, count) {
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * power >= raw) return m * power;
  return 10 * power;
}

// Lay every bar out on the current time scale, and the axis ticks and cursor with them.
function draw() {
  const t = now();
  const span = Math.max(t, 0.05);
  for (const row of state.rows) {
    const end = row.end ?? t;
    row.bar.style.setProperty("left", `${(row.start / span) * 100}%`);
    row.bar.style.setProperty("width", `${(Math.max(end - row.start, 0) / span) * 100}%`);
    row.dur.textContent = secs(Math.max(end - row.start, 0));
  }
  const axis = $("axis");
  const step = niceStep(span, Math.max(2, Math.floor(axis.clientWidth / 72)));
  const ticks = [];
  for (let x = 0; x <= span + 1e-9; x += step) {
    ticks.push(h("div", { class: "tick", style: { left: `${(x / span) * 100}%` } }, h("span", {}, x === 0 ? "0" : secs(x))));
  }
  axis.replaceChildren(...ticks);
  $("cursor").style.setProperty("--x", `${axis.clientWidth}px`);
  $("clock").textContent = t.toFixed(2);
}

function tick() {
  draw();
  if (state.endT === null) state.frame = requestAnimationFrame(tick);
}

function handle(event) {
  if (state.t0 === null) {
    state.t0 = performance.now() / 1000 - event.t;
    document.querySelector(".timeline").classList.add("is-live");
    state.frame = requestAnimationFrame(tick);
  }
  const row = state.latest.get(event.stage);
  if (event.status === "start") {
    addRow(event);
    setStatus(`${STAGES[event.stage]?.name ?? event.stage}…`);
  } else if (event.status === "progress" && row) {
    describe(row, event);
  } else if (event.status === "end" && event.stage === "job") {
    finishRun(event);
  } else if (event.status === "end" && event.stage === "done") {
    setStatus("Finishing…");
  } else if (event.status === "end" && row && row.end === null) {
    endRow(row, event);
  } else if (event.status === "error") {
    for (const open of state.rows) if (open.end === null) endRow(open, { ...event, data: { aborted: true } });
    errorRow(event);
  }
}

function finishRun(event) {
  state.endT = event.t;
  for (const row of state.rows) if (row.end === null) endRow(row, { t: event.t, status: "end", data: { aborted: true } });
  cancelAnimationFrame(state.frame);
  draw();
  document.querySelector(".timeline").classList.remove("is-live");
  if (state.source) state.source.close();
  state.source = null;
  setBusy(false);
  const ok = event.data?.status === "done";
  setStatus(ok ? `Done in ${secs(event.t)}.` : event.data?.error || "The run failed.", !ok);
  if (ok) showResult(state.runId);
  refreshHistory();
}

function follow(runId) {
  state.runId = runId;
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  state.source = source;
  source.onmessage = (message) => {
    if (state.source !== source) return;
    handle(JSON.parse(message.data));
  };
  source.onerror = () => {
    if (state.source === source && source.readyState === EventSource.CONNECTING) {
      setStatus("Lost the connection to the server; reconnecting…", true);
    }
  };
  markHistory();
}

// --- results -------------------------------------------------------------------------------

async function showResult(runId) {
  let run;
  try {
    run = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  } catch (error) {
    setStatus(error.message, true);
    return;
  }
  if (state.runId !== runId) return;
  renderFacts(run);
  renderToc(run);
  renderType(run);
  renderSections(run);
  renderJson(run);
  $("results").hidden = false;
}

function renderFacts(run) {
  const meta = run.meta ?? {};
  const facts = [
    ["Pages", meta.n_pages !== undefined ? num(meta.n_pages) : null],
    ["Headings", run.toc ? num(run.toc.length) : null],
    ["Engine", meta.engine ?? (run.kind === "pdf" ? null : "Markdown")],
    ["Model", run.options?.backend],
    ["Wall time", secs(run.seconds)],
  ];
  if (meta.warnings?.length) facts.push(["Warnings", meta.warnings.join(", ").replace(/_/g, " ")]);
  $("facts").replaceChildren(
    ...facts.filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, v))),
  );
}

function scoreChip(score) {
  const level = score >= 0.8 ? "hi" : score >= 0.55 ? "mid" : "lo";
  return h("span", { class: `chip ${level}`, title: `Heading score ${score.toFixed(2)}` }, score.toFixed(2));
}

function tocRow(entry, hasChildren) {
  const flags = entry.flags.length
    ? h("span", { class: "toc-flags" }, entry.flags.map((f) => h("span", { class: "badge" }, FLAG_NAMES[f] ?? f.replace(/_/g, " "))))
    : null;
  return h(
    "span",
    { class: "toc-row" },
    h("span", { class: "toc-twisty", "aria-hidden": "true" }, hasChildren ? "▶" : ""),
    h("span", { class: `toc-title level-${Math.min(entry.level, 3)}` }, entry.title || "(untitled)", flags),
    h("span", { class: "toc-page" }, entry.page ? `p. ${entry.page}` : ""),
    scoreChip(entry.heading_score),
  );
}

// Build the nested TOC from the flat, depth-annotated entries, with a stack (no recursion).
function buildTree(entries) {
  const root = h("ul", { class: "toc" });
  const stack = [{ depth: -1, list: root }];
  entries.forEach((entry, i) => {
    while (stack.length > 1 && stack[stack.length - 1].depth >= entry.depth) stack.pop();
    const parent = stack[stack.length - 1].list;
    const hasChildren = i + 1 < entries.length && entries[i + 1].depth > entry.depth;
    if (hasChildren) {
      const list = h("ul");
      parent.append(h("li", {}, h("details", { open: true }, h("summary", {}, tocRow(entry, true)), list)));
      stack.push({ depth: entry.depth, list });
    } else {
      parent.append(h("li", {}, tocRow(entry, false)));
    }
  });
  return root;
}

function renderToc(run) {
  const panel = $("panel-toc");
  const entries = run.toc ?? [];
  if (!entries.length) {
    panel.replaceChildren(h("p", { class: "empty-note" }, "No headings were found. A scanned PDF without a text layer looks like this; try another engine."));
    return;
  }
  const tree = buildTree(entries);
  const setAll = (open) => tree.querySelectorAll("details").forEach((d) => (d.open = open));
  panel.replaceChildren(
    h(
      "div",
      { class: "toolbar" },
      h("button", { type: "button", class: "secondary", onclick: () => setAll(true) }, "Expand all"),
      h("button", { type: "button", class: "secondary", onclick: () => setAll(false) }, "Collapse all"),
    ),
    tree,
  );
}

function renderType(run) {
  const panel = $("panel-type");
  const type = run.doc_type;
  if (!type) {
    panel.replaceChildren(h("p", { class: "empty-note" }, "No document type for this run."));
    return;
  }
  const head = h(
    "div",
    { class: "type-head" },
    type.path.length
      ? h("ol", { class: "crumbs", "aria-label": "Type" }, type.path.map((p) => h("li", {}, p)))
      : h("p", { class: "crumbs" }, `No type chosen (${(type.stopped_reason || "nothing to classify").replace(/_/g, " ")})`),
    h("p", { class: "confidence" }, "Confidence ", h("strong", {}, pct(type.confidence))),
  );
  const bars = h(
    "ol",
    { class: "bars", "aria-label": "Most probable types" },
    type.leaf_probs.map((item) =>
      h(
        "li",
        { class: `bar-row${item.id === type.id ? " is-best" : ""}` },
        h("span", { class: "bar-label" }, item.label, h("small", {}, item.id)),
        h("span", { class: "bar-track", "aria-hidden": "true" }, h("span", { class: "bar-fill", style: { "--p": `${item.prob * 100}%`, display: "block" } })),
        h("span", { class: "bar-value" }, pct(item.prob)),
      ),
    ),
  );
  panel.replaceChildren(head, bars);
}

function renderSections(run) {
  const panel = $("panel-sections");
  if (!run.sections) {
    panel.replaceChildren(h("p", { class: "empty-note" }, "Section labels were off for this run. Tick “Label sections with canonical names” and run the file again."));
    return;
  }
  const rows = run.sections.entries.map((e) =>
    h(
      "tr",
      {},
      h("td", { style: { "padding-left": `${e.depth * 1.1}rem` } }, e.title || "(untitled)"),
      e.canonical
        ? h("td", {}, e.label, h("span", { class: "canon" }, e.canonical))
        : h("td", { class: "none" }, e.level > 2 ? "below the labelled levels" : "no label"),
      h("td", { class: "num" }, e.score === null ? "" : e.score.toFixed(2)),
    ),
  );
  panel.replaceChildren(
    h("p", { class: "note", style: { "margin-bottom": "0.75rem" } }, `Labels from ${run.sections.taxonomy}.`),
    h(
      "div",
      { class: "table-wrap" },
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Section"), h("th", {}, "Canonical label"), h("th", { class: "num" }, "Score"))), h("tbody", {}, rows)),
    ),
  );
}

function renderJson(run) {
  const text = JSON.stringify(run, null, 2);
  const stem = (run.filename || "run").replace(/\.[^.]+$/, "");
  const download = () => {
    const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    const link = h("a", { href: url, download: `${stem}.docrooter.json` });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  $("panel-json").replaceChildren(
    h("div", { class: "toolbar" }, h("button", { type: "button", class: "secondary", onclick: download }, "Download JSON")),
    h("pre", { class: "json" }, h("code", {}, text)),
  );
}

function setupTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const select = (tab) => {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      $(t.getAttribute("aria-controls")).hidden = !on;
    }
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      const moves = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 };
      if (!(event.key in moves)) return;
      event.preventDefault();
      const next = tabs[(moves[event.key] + tabs.length) % tabs.length];
      next.focus();
      select(next);
    });
  });
}

// --- history -------------------------------------------------------------------------------

async function refreshHistory() {
  let runs;
  try {
    runs = await getJSON("/api/runs");
  } catch {
    return;
  }
  $("history-empty").hidden = runs.length > 0;
  $("history").replaceChildren(
    ...runs.map((run) => {
      const sub = run.status === "done"
        ? [run.doc_type ?? "no type", run.options.backend.split(":").pop()].join(", ")
        : run.status === "error"
          ? run.error ?? "failed"
          : run.status;
      return h(
        "li",
        {},
        h(
          "button",
          { type: "button", "data-run": run.id, onclick: () => openRun(run.id) },
          h("span", { class: `history-dot ${run.status}`, "aria-hidden": "true" }),
          h("span", { class: "history-name" }, run.filename),
          h("span", { class: "history-time" }, run.seconds !== null ? secs(run.seconds) : ""),
          h("span", { class: "history-sub" }, sub),
        ),
      );
    }),
  );
  markHistory();
}

function markHistory() {
  for (const button of $("history").querySelectorAll("button")) {
    button.setAttribute("aria-current", String(button.dataset.run === state.runId));
  }
}

async function openRun(runId) {
  if (runId === state.runId && state.source) return;
  let run;
  try {
    run = await getJSON(`/api/runs/${encodeURIComponent(runId)}`);
  } catch (error) {
    setStatus(error.message, true);
    refreshHistory();
    return;
  }
  resetRun(run.filename);
  if (!["done", "error"].includes(run.status)) {
    // Still queued or running: follow it live (the server replays what has happened so far).
    setBusy(true);
    follow(runId);
    return;
  }
  state.runId = runId;
  markHistory();
  state.t0 = 0;
  for (const event of run.events) handle(event);
  $("main").focus();
}

// --- start ---------------------------------------------------------------------------------

setupDrop();
setupTabs();
loadOptions().catch((error) => setStatus(`Could not load the options: ${error.message}`, true));
refreshHistory();
