// docrooter playground: upload a document, follow its pipeline events over SSE, show results.
// No framework and no build step. Every string from the server goes into the DOM as text.

const $ = (id) => document.getElementById(id);

const STAGES = {
  upload: { name: "Upload", fam: "io" },
  extract: { name: "Read pages", fam: "doc" },
  signals: { name: "Page signals", fam: "doc" },
  headings: { name: "Find headings", fam: "doc" },
  assemble: { name: "Build tree", fam: "doc" },
  route: { name: "Route pages", fam: "doc" },
  ocr: { name: "OCR", fam: "model" },
  layout: { name: "Find layout", fam: "doc" },
  elements: { name: "Build elements", fam: "doc" },
  summarize: { name: "Summarise", fam: "doc" },
  view: { name: "Render view", fam: "model" },
  model_load: { name: "Load model", fam: "model" },
  classify: { name: "Classify type", fam: "model" },
  normalize: { name: "Label sections", fam: "model" },
  tag: { name: "Tag elements", fam: "model" },
};
const FLAG_NAMES = {
  level_from_bookmark: "level from bookmark",
  low_heading_confidence: "low confidence",
};
const ENGINE_LABELS = {
  pymupdf4llm: "pymupdf4llm (sections only)",
  pdfplumber: "pdfplumber (elements only)",
};
const PREFS_KEY = "docrooter.playground.options";
const KIND_NAMES = { page_header: "page header", page_footer: "page footer", page_number: "page number", list_item: "list item" };
const ELEMENT_KIND_FAMILY = {
  title: "head", heading: "head", contents: "head",
  table: "data", figure: "data", formula: "data", caption: "data",
  page_header: "furniture", page_footer: "furniture", page_number: "furniture", footnote: "furniture",
};

const PRESETS = [
  { name: "Lean", value: 0.8 },
  { name: "Balanced", value: 0.9 },
  { name: "Thorough", value: 0.98 },
  { name: "Whole document", value: 1.0 },
];
const SELECT_DEBOUNCE_MS = 250;
const NO_EMBEDDER_NOTE = "Free-text queries need an embedder; this selector was fitted without one. Pick a field instead.";
const NO_SELECTOR_NOTE = "No selector: fit one with docrooter fit-selector, then start the playground with --selector. Only the whole document can be selected until then.";

const state = {
  backends: null,
  fields: null, // family -> [{id, label, description}], loaded once
  select: null, // the dial of the shown run, see renderSelect
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
const TEXT_ANSWER = "no probability (text answer)";
// A confidence, or "–" with a tooltip when the answer was a bare letter (no logprobs).
const confidence = (p) => (p === null || p === undefined ? h("span", { title: TEXT_ANSWER }, "–") : pct(p));
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
        profile: $("profile").value,
        backend: $("backend").value,
        engine: $("engine").value,
        mode: $("mode").value,
        normalize: $("normalize").checked,
        flow: $("flow").value,
        level: $("level").value,
        calls: $("calls").value,
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
  const profile = $("profile");
  profile.replaceChildren(
    ...info.profiles.map((p) =>
      h("option", { value: p.name }, p.installed ? p.label : `${p.label} (not installed)`),
    ),
  );
  const names = info.profiles.map((p) => p.name);
  profile.value = names.includes(prefs.profile) ? prefs.profile : info.default_profile;
  const backend = $("backend");
  backend.replaceChildren(
    ...info.backends.map((b) =>
      h("option", { value: b.spec }, b.installed ? b.label : `${b.label} (not installed)`),
    ),
  );
  const specs = info.backends.map((b) => b.spec);
  backend.value = specs.includes(prefs.backend) ? prefs.backend : info.default;
  const engine = $("engine");
  engine.replaceChildren(...info.engines.map((e) => h("option", { value: e }, ENGINE_LABELS[e] ?? e)));
  engine.value = info.engines.includes(prefs.engine) ? prefs.engine : "auto";
  if (info.modes.includes(prefs.mode)) $("mode").value = prefs.mode;
  if (typeof prefs.normalize === "boolean") $("normalize").checked = prefs.normalize;
  if (prefs.flow === "" || info.flows?.includes(prefs.flow)) $("flow").value = prefs.flow;
  if (info.levels?.includes(prefs.level)) $("level").value = prefs.level;
  if (prefs.calls === "" || info.calls?.includes(prefs.calls)) $("calls").value = prefs.calls;
  describeBackend();
  describeProfile();
  describeFlow();
}

// The flow decides which other choices apply: the context level only with a flow, the
// "type from" mode and the section-label switch only without one.
function describeFlow() {
  const flow = $("flow").value;
  $("level").disabled = !flow;
  $("calls").disabled = !flow;
  $("mode").disabled = Boolean(flow);
  $("normalize").disabled = Boolean(flow);
  const notes = {
    "": "The classic run: the type from the chosen view, then section labels.",
    sections_first: "Tags the sections first; their tags decide the type or narrow the choices sent to the model.",
    doc_first: "Classifies the type first, then labels the sections with that type's taxonomy.",
  };
  $("flow-note").textContent = notes[flow] ?? "";
}

function describeProfile() {
  const chosen = state.backends?.profiles.find((p) => p.name === $("profile").value);
  const note = $("profile-note");
  note.replaceChildren();
  if (!chosen) return;
  note.append(chosen.description);
  if (!chosen.installed) note.append(" ", h("strong", {}, `Needs ${chosen.install}.`));
}

function describeBackend() {
  const chosen = state.backends?.backends.find((b) => b.spec === $("backend").value);
  const note = $("backend-note");
  note.replaceChildren();
  if (!chosen) return;
  note.append(chosen.description);
  if (!chosen.installed) note.append(" ", h("strong", {}, `Needs ${chosen.install}.`));
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
  const isText = /\.(md|txt)$/i.test(file.name);
  const limit = isText
    ? state.backends?.max_text_upload_bytes ?? 1 * 1024 * 1024
    : state.backends?.max_upload_bytes ?? 50 * 1024 * 1024;
  if (file.size > limit) {
    const kind = isText ? "for .md and .txt files " : "";
    $("form-error").textContent = `${file.name} is ${bytes(file.size)}; the limit ${kind}is ${bytes(limit)}.`;
    return;
  }
  state.lastFile = file;
  savePrefs();
  const form = new FormData();
  form.append("file", file);
  form.append("profile", $("profile").value);
  form.append("backend", $("backend").value);
  form.append("engine", $("engine").value);
  form.append("mode", $("mode").value);
  form.append("normalize", $("normalize").checked ? "true" : "false");
  if ($("flow").value) {
    form.append("flow", $("flow").value);
    form.append("level", $("level").value);
    if ($("calls").value) form.append("calls", $("calls").value);
  }
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
  for (const id of ["profile", "backend", "engine", "mode", "normalize", "flow", "level", "calls"]) $(id).addEventListener("change", savePrefs);
  $("flow").addEventListener("change", describeFlow);
  $("profile").addEventListener("change", describeProfile);
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
  stopSelect();
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

// The name and colour family of a stage's row. The rooter's engine choices come first, as their
// own "route" stage (scope "document"); the page decisions follow in a second one.
function stageMeta(event) {
  if (event.stage === "route" && event.data?.scope === "document") return { name: "Choose engines", fam: "doc" };
  return STAGES[event.stage] ?? { name: event.stage.replace(/_/g, " "), fam: "io" };
}

function addRow(event) {
  const meta = stageMeta(event);
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
    case "route":
      if (status === "end") {
        if (d.pages !== undefined) {
          parts.push(`${num(d.pages)} pages`);
          if (d.ocr) parts.push(`${num(d.ocr)} need OCR`);
          if (d.merge) parts.push(`${num(d.merge)} scans with text`);
          if (d.regions) parts.push(`${num(d.regions)} image regions`);
        } else if (d.decisions !== undefined) {
          parts.push(`${num(d.decisions)} engine decisions`);
        }
      }
      break;
    case "ocr":
      if (d.engine) parts.push(d.engine);
      if (status === "end") parts.push(`${num(d.pages ?? 0)} pages and ${num(d.regions ?? 0)} regions read`, `${num(d.lines ?? 0)} lines`);
      else if (d.pages || d.regions) parts.push(`${num(d.pages ?? 0)} pages and ${num(d.regions ?? 0)} regions to read`);
      if (status === "end" && d.skipped) parts.push("needed OCR but no engine ran");
      break;
    case "layout":
      if (d.engine) parts.push(d.engine);
      if (status === "end") parts.push(`${num(d.regions ?? 0)} regions`);
      break;
    case "elements":
      if (status === "end") {
        parts.push(`${num(d.elements)} elements`);
        parts.push(...Object.entries(d.kinds ?? {}).slice(0, 4).map(([kind, n]) => `${num(n)} ${kindName(kind)}`));
      }
      break;
    case "summarize":
      if (d.method) parts.push(d.method);
      if (status === "end") parts.push(`${num(d.summaries)} summaries`);
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
    setStatus(`${stageMeta(event).name}…`);
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
    if (state.source !== source) return;
    if (source.readyState === EventSource.CONNECTING) {
      setStatus("Lost the connection to the server; reconnecting…", true);
    } else if (source.readyState === EventSource.CLOSED) {
      // The reconnect was refused (a restarted server no longer knows the run): stop waiting.
      state.source = null;
      cancelAnimationFrame(state.frame);
      document.querySelector(".timeline").classList.remove("is-live");
      setBusy(false);
      setStatus("This run is no longer on the server (it may have restarted).", true);
      refreshHistory();
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
  renderElements(run);
  renderSelect(run);
  renderType(run);
  renderSections(run);
  renderJson(run);
  $("results").hidden = false;
}

function renderFacts(run) {
  const meta = run.meta ?? {};
  const facts = [
    ["Pages", meta.n_pages !== undefined ? num(meta.n_pages) : null],
    ["Headings", run.toc ? num(run.toc_total ?? run.toc.length) : null],
    ["Elements", run.elements_total ? num(run.elements_total) : null],
    ["Profile", run.options?.profile],
    ["Engine", meta.engine ?? (run.kind === "pdf" ? null : "Markdown")],
    ["Model", run.options?.backend],
    ["Flow", run.options?.flow ? `${run.options.flow.replace(/_/g, " ")}, ${run.options.level ?? "measured level"}` : null],
    ["Calls", run.doc_type?.calls ? run.doc_type.calls.replace(/_/g, " ") : null],
    ["Tokens", run.doc_type?.tokens ? tokenText(run.doc_type.tokens) : null],
    ["Wall time", secs(run.seconds)],
  ];
  const ocrPages = (run.routes?.tasks?.ocr ?? 0) + (run.routes?.tasks?.["text+ocr"] ?? 0);
  if (ocrPages) facts.push(["OCR pages", num(ocrPages)]);
  if (meta.warnings?.length) facts.push(["Warnings", meta.warnings.join(", ").replace(/_/g, " ")]);
  $("facts").replaceChildren(
    ...facts.filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, v))),
  );
}

// "1,234 prompt tokens over 5 calls (exact)": exact only when every call reported its usage.
function tokenText(tokens) {
  const kind = tokens.exact ? "exact" : "estimated";
  return `${num(tokens.prompt)} prompt tokens over ${num(tokens.calls)} calls (${kind})`;
}

function scoreChip(score) {
  const level = score >= 0.8 ? "hi" : score >= 0.55 ? "mid" : "lo";
  return h("span", { class: `chip ${level}`, title: `Heading score ${score.toFixed(2)}` }, score.toFixed(2));
}

// The key sentences of a section, as a quiet block under its TOC row.
function summaryBlock(summary) {
  if (!summary?.key_sentences?.length) return null;
  return h(
    "div",
    { class: "toc-summary" },
    summary.key_sentences.map((sentence) => h("p", {}, sentence)),
    summary.keyphrases?.length ? h("p", { class: "toc-phrases" }, summary.keyphrases.join(" · ")) : null,
  );
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
function buildTree(entries, summaries = {}) {
  const root = h("ul", { class: "toc" });
  const stack = [{ depth: -1, list: root }];
  entries.forEach((entry, i) => {
    while (stack.length > 1 && stack[stack.length - 1].depth >= entry.depth) stack.pop();
    const parent = stack[stack.length - 1].list;
    const hasChildren = i + 1 < entries.length && entries[i + 1].depth > entry.depth;
    if (hasChildren) {
      const list = h("ul");
      parent.append(h("li", {}, h("details", { open: true }, h("summary", {}, tocRow(entry, true)), summaryBlock(summaries[entry.id]), list)));
      stack.push({ depth: entry.depth, list });
    } else {
      parent.append(h("li", {}, tocRow(entry, false), summaryBlock(summaries[entry.id])));
    }
  });
  return root;
}

// The playground keeps at most a fixed number of headings per run; say so when it cut some.
function truncationNote(run) {
  if (!run.truncated) return null;
  return h(
    "p",
    { class: "note", style: { "margin-bottom": "0.75rem" } },
    h("strong", {}, `Showing the first ${num(run.toc.length)} of ${num(run.toc_total)} headings.`),
    " The playground keeps that many per run; use the library or the CLI for the full list.",
  );
}

function renderToc(run) {
  const panel = $("panel-toc");
  const entries = run.toc ?? [];
  if (!entries.length) {
    panel.replaceChildren(h("p", { class: "empty-note" }, "No headings were found. A scanned PDF without a text layer looks like this; try another engine."));
    return;
  }
  const tree = buildTree(entries, run.summaries ?? {});
  const setAll = (open) => tree.querySelectorAll("details").forEach((d) => (d.open = open));
  panel.replaceChildren(
    ...[
      truncationNote(run),
      h(
        "div",
        { class: "toolbar" },
        h("button", { type: "button", class: "secondary", onclick: () => setAll(true) }, "Expand all"),
        h("button", { type: "button", class: "secondary", onclick: () => setAll(false) }, "Collapse all"),
      ),
      tree,
    ].filter(Boolean),
  );
}

const kindName = (kind) => KIND_NAMES[kind] ?? kind;

// A tag id as a short name: a topic leaf without its family ("annual_report.risk_factors").
const tagName = (id) => id.split(".").pop().replace(/_/g, " ");
const tagProb = (result, id) => result.probs.find((p) => p.id === id)?.prob ?? result.confidence;

function tagTitle(facet, result) {
  return `${facet}: ${result.probs.map((p) => `${tagName(p.id)} ${pct(p.prob)}`).join(", ")}`;
}

// An element's tags as chips: its topic, then each function label, with their probabilities.
function tagChips(tags) {
  if (!tags) return null;
  const chips = [];
  for (const [facet, title] of [["topic", "Topic"], ["function", "Function"]]) {
    const result = tags[facet];
    if (!result) continue;
    for (const id of result.labels) {
      chips.push(
        h("span", { class: `chip tag-chip tag-${facet}`, title: tagTitle(title, result) }, `${tagName(id)} ${pct(tagProb(result, id))}`),
      );
    }
  }
  return chips.length ? h("span", { class: "tag-chips" }, chips) : null;
}

function elementRow(el) {
  const family = ELEMENT_KIND_FAMILY[el.kind] ?? "body";
  const head = h(
    "span",
    { class: "el-row" },
    h("span", { class: `kind kind-${family}`, title: `Element ${el.id}` }, kindName(el.kind)),
    el.source === "ocr" ? h("span", { class: "badge ocr", title: "Read by OCR" }, "OCR") : null,
    el.flags.map((f) => h("span", { class: "badge" }, f.replace(/_/g, " "))),
    h("span", { class: "el-text" }, el.text || "(no text)"), // the server marks a cut with "…"
    tagChips(el.tags),
  );
  if (!el.summary.length) return h("li", { class: "el", "data-id": el.id }, head);
  return h(
    "li",
    { class: "el", "data-id": el.id },
    h(
      "details",
      {},
      h("summary", {}, head, h("span", { class: "el-more" }, "summary")),
      h("div", { class: "toc-summary" }, el.summary.map((sentence) => h("p", {}, sentence))),
    ),
  );
}

// Elements grouped by page, in reading order; a long document's pages start collapsed.
function renderElements(run) {
  const panel = $("panel-elements");
  const items = run.elements ?? [];
  if (!items.length) {
    const fast = run.options?.profile === "fast";
    panel.replaceChildren(
      h("p", { class: "empty-note" }, fast
        ? "The fast profile builds headings only. Pick Balanced or Accurate and run the file again to see elements."
        : "No elements were found in this document."),
    );
    return;
  }
  const pages = new Map();
  for (const el of items) {
    const key = el.page ?? 0;
    if (!pages.has(key)) pages.set(key, []);
    pages.get(key).push(el);
  }
  const groups = [...pages].map(([page, list], index) =>
    h(
      "details",
      { class: "el-page", open: pages.size <= 3 || index === 0 },
      h("summary", {}, page ? `Page ${page}` : "Whole document", h("small", {}, `${num(list.length)} elements`)),
      h("ol", { class: "el-list" }, list.map(elementRow)),
    ),
  );
  const counts = Object.entries(run.element_counts ?? {}).map(([kind, n]) => `${num(n)} ${kindName(kind)}`);
  if (run.element_tags) counts.push(`tags by ${run.element_tags.source} (${tagName(run.element_tags.family)} topics)`);
  const setAll = (open) => groups.forEach((g) => (g.open = open));
  const selectedOnly = h("input", { type: "checkbox", id: "el-selected-only", disabled: true, onchange: applySelection });
  const note = run.elements_truncated
    ? h(
        "p",
        { class: "note", style: { "margin-bottom": "0.75rem" } },
        h("strong", {}, `Showing the first ${num(items.length)} of ${num(run.elements_total)} elements.`),
        " The playground keeps that many per run; use the library or the CLI for the full list.",
      )
    : null;
  panel.replaceChildren(
    ...[
      note,
      h("p", { class: "note", style: { "margin-bottom": "0.75rem" } }, counts.join(", ")),
      h(
        "div",
        { class: "toolbar" },
        h("label", { class: "check selected-only" }, selectedOnly, h("span", {}, "Show selected only")),
        h("button", { type: "button", class: "secondary", onclick: () => setAll(true) }, "Expand all"),
        h("button", { type: "button", class: "secondary", onclick: () => setAll(false) }, "Collapse all"),
      ),
      ...groups,
    ].filter(Boolean),
  );
}

// --- the completeness dial ------------------------------------------------------------------

// The state of the dial for the shown run. `selected` is a Set of element ids once a selection
// has been read, else null; `seq` numbers the requests so a slow answer never overwrites a newer.
function newSelectState(run) {
  return { runId: run.id, family: run.field_family, selected: null, timer: 0, controller: null, seq: 0 };
}

function stopSelect() {
  const dial = state.select;
  state.select = null;
  if (!dial) return;
  clearTimeout(dial.timer);
  if (dial.controller) dial.controller.abort();
}

function selectBadges(selection) {
  const found = [];
  if (!selection.guarantee && selection.expected_recall !== null) found.push(["estimate", "The recall is the model's own estimate, not a guarantee"]);
  if (selection.capped) found.push(["capped", "Too few calibration documents reach this completeness, so the whole document is kept"]);
  if (selection.doc_type_mismatch) found.push(["mismatch", "The field belongs to another document type than this document's"]);
  if (selection.uncalibrated_field) found.push(["uncalibrated", "The dial was not calibrated on this field, so the recall is an estimate"]);
  if (selection.multi_field) found.push(["multi", "The dial was calibrated on single fields, so several fields together are an estimate: select once per field for a guarantee"]);
  if (selection.fit_mismatch.length) found.push(["fit", `This run was not read and tagged as the selector was fitted (${selection.fit_mismatch.join(", ")} differ), so the recall is an estimate`]);
  if (selection.over_budget) found.push(["over-budget", "The most relevant element alone is over the token budget"]);
  if (selection.calibration_n !== null) found.push(["calibration", `The dial was calibrated on n = ${selection.calibration_n} units; its finite-sample term 1/(n+1) is ${(1 / (selection.calibration_n + 1)).toFixed(3)}`]);
  const labels = { estimate: "estimate", capped: "dial capped by calibration size", mismatch: "doc type mismatch", uncalibrated: "field not calibrated", multi: "several fields", fit: "reading differs from calibration", "over-budget": "over budget", calibration: `n = ${selection.calibration_n}` };
  return found.map(([kind, title]) => h("span", { class: "badge select-badge", "data-kind": kind, title }, labels[kind]));
}

// Mark the selected elements in the Elements tab, and hide the rest when asked to.
function applySelection() {
  const dial = state.select;
  const selected = dial?.selected ?? null;
  const toggle = $("el-selected-only");
  if (toggle) toggle.disabled = selected === null;
  const only = Boolean(toggle?.checked) && selected !== null;
  const panel = $("panel-elements");
  for (const item of panel.querySelectorAll("li.el")) {
    const on = selected !== null && selected.has(item.dataset.id);
    item.classList.toggle("is-selected", on);
    item.classList.toggle("is-unselected", only && !on);
  }
  for (const group of panel.querySelectorAll("details.el-page")) {
    group.hidden = only && !group.querySelector("li.el.is-selected");
  }
}

function currentNeed() {
  const query = $("select-query").value.trim();
  return query || $("select-field").value || null;
}

function showReadout(text, isError = false) {
  $("select-readout").textContent = text;
  $("select-readout").classList.toggle("is-error", isError);
}

async function runSelect() {
  const dial = state.select;
  if (!dial) return;
  clearTimeout(dial.timer);
  const need = currentNeed();
  if (dial.controller) dial.controller.abort();
  if (!need) {
    dial.selected = null;
    $("select-badges").replaceChildren();
    $("select-note").replaceChildren();
    showReadout("Pick a field or type a query to see what would be sent.");
    applySelection();
    return;
  }
  const controller = new AbortController();
  dial.controller = controller;
  const seq = ++dial.seq;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(dial.runId)}/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ need, completeness: Number($("select-slider").value) }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await errorText(response));
    const selection = await response.json();
    if (state.select !== dial || dial.seq !== seq) return;
    dial.selected = new Set(selection.element_ids);
    $("select-readout").textContent = selection.readout;
    $("select-readout").classList.remove("is-error");
    $("select-badges").replaceChildren(...selectBadges(selection));
    const shown = selection.truncated ? ` Highlighting the first ${num(selection.element_ids.length)}.` : "";
    $("select-note").textContent = `${num(selection.selected_total)} elements selected, with their headings and neighbours.${shown}`;
    applySelection();
  } catch (error) {
    if (error.name === "AbortError" || state.select !== dial || dial.seq !== seq) return;
    dial.selected = null;
    $("select-badges").replaceChildren();
    $("select-note").replaceChildren();
    showReadout(error.message, true);
    applySelection();
  }
}

function scheduleSelect() {
  const dial = state.select;
  if (!dial) return;
  clearTimeout(dial.timer);
  dial.timer = setTimeout(runSelect, SELECT_DEBOUNCE_MS);
}

function syncDial() {
  const value = Number($("select-slider").value);
  $("select-value").textContent = value.toFixed(2);
  for (const button of $("select-presets").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.value) === value));
  }
}

async function fieldGroups(family) {
  if (!state.fields) {
    try {
      state.fields = await getJSON("/api/fields");
    } catch {
      state.fields = {};
    }
  }
  const families = Object.keys(state.fields).sort((a, b) => (a === family ? -1 : b === family ? 1 : a.localeCompare(b)));
  return families.map((name) =>
    h(
      "optgroup",
      { label: name === family ? `${name.replace(/_/g, " ")} (this document)` : name.replace(/_/g, " ") },
      state.fields[name].map((f) => h("option", { value: f.id, title: f.description }, f.label)),
    ),
  );
}

async function renderSelect(run) {
  stopSelect();
  const panel = $("select-panel");
  panel.hidden = !run.elements_total;
  if (!run.elements_total) return;
  const dial = newSelectState(run);
  state.select = dial;
  const hasSelector = Boolean(state.backends?.selector?.loaded);
  const hasEmbedder = Boolean(state.backends?.selector?.embedder);
  const slider = $("select-slider");
  slider.value = hasSelector ? "0.9" : "1";
  slider.disabled = !hasSelector;
  const query = $("select-query");
  query.value = "";
  query.disabled = hasSelector && !hasEmbedder; // without sim a query matches nothing
  query.title = query.disabled ? NO_EMBEDDER_NOTE : "";
  $("select-presets").replaceChildren(
    ...PRESETS.map((preset) =>
      h(
        "button",
        {
          type: "button",
          class: "secondary preset",
          "data-value": preset.value,
          "aria-pressed": "false",
          disabled: !hasSelector && preset.value !== 1.0,
          onclick: () => {
            slider.value = String(preset.value);
            syncDial();
            runSelect();
          },
        },
        preset.name,
      ),
    ),
  );
  $("select-note").textContent = !hasSelector ? NO_SELECTOR_NOTE : hasEmbedder ? "" : NO_EMBEDDER_NOTE;
  showReadout("Pick a field or type a query to see what would be sent.");
  $("select-badges").replaceChildren();
  syncDial();
  const groups = await fieldGroups(run.field_family);
  if (state.select !== dial) return;
  $("select-field").replaceChildren(h("option", { value: "" }, "Choose a field…"), ...groups);
  applySelection();
}

function setupSelect() {
  $("select-slider").addEventListener("input", () => {
    syncDial();
    scheduleSelect();
  });
  $("select-field").addEventListener("change", () => {
    if ($("select-field").value) $("select-query").value = "";
    runSelect();
  });
  $("select-query").addEventListener("input", () => {
    if ($("select-query").value.trim()) $("select-field").value = "";
    scheduleSelect();
  });
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
    h("p", { class: "confidence" }, "Confidence ", h("strong", {}, confidence(type.confidence))),
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
  panel.replaceChildren(head, ...flowBlock(type), bars);
}

// How a flow run decided its type: by rules, by the model among the candidates, or by
// fallback, with the flags and the tokens each stage spent.
function flowBlock(type) {
  if (!type.decided_by) return [];
  const how = {
    rules: "Decided by the section rules, with no document-level call.",
    model: "Chosen by the model among the candidates.",
    fallback: "Chosen by the model among every type (fallback).",
  };
  const stages = Object.entries(type.stage_tokens ?? {}).map(([stage, t]) => `${stage}: ${num(t.prompt)}`);
  const levels = Object.entries(type.levels ?? {}).filter(([, v]) => v).map(([k, v]) => `${k} ${v}`);
  const block = h(
    "div",
    { class: "flow-facts" },
    h("p", {}, h("strong", {}, `decided_by: ${type.decided_by}`), ` · ${how[type.decided_by] ?? ""}`),
    type.flags.includes("thin_evidence")
      ? h("p", { class: "note" }, "Too few sections were tagged with confidence for the rules, so every type was offered.")
      : type.flags.includes("low_support")
        ? h("p", { class: "note" }, "The rules left one weakly supported type, so every type was offered.")
        : null,
    h(
      "p",
      {},
      "Candidates: ",
      type.candidates.length
        ? type.candidates.flatMap((c, i) => [i ? ", " : "", h("span", { title: c.id }, c.label)])
        : "none (every type was offered)",
    ),
    type.flags.length ? h("p", {}, `Flags: ${type.flags.join(", ").replace(/_/g, " ")}`) : null,
    h("p", {}, `Tokens: ${tokenText(type.tokens)}${stages.length ? ` · ${stages.join(", ")}` : ""}`),
    levels.length ? h("p", { class: "note" }, `Levels: ${levels.join(", ")}`) : null,
  );
  return [block];
}

function renderSections(run) {
  const panel = $("panel-sections");
  if (!run.sections) {
    panel.replaceChildren(h("p", { class: "empty-note" }, "Section labels were off for this run. Tick “Label sections with canonical names” and run the file again."));
    return;
  }
  const hasLevels = run.sections.entries.some((e) => "level_used" in e);
  const rows = run.sections.entries.map((e) =>
    h(
      "tr",
      {},
      h("td", { style: { "padding-left": `${e.depth * 1.1}rem` } }, e.title || "(untitled)"),
      e.canonical
        ? h("td", {}, e.label, h("span", { class: "canon" }, e.canonical))
        : h("td", { class: "none" }, e.level > 2 ? "below the labelled levels" : "no label"),
      h("td", { class: "num" }, e.confidence_source === "text" && e.score === null
        ? h("span", { title: TEXT_ANSWER }, "–")
        : e.score === null ? "" : e.score.toFixed(2)),
      hasLevels ? h("td", {}, e.level_used ?? "") : null,
    ),
  );
  panel.replaceChildren(
    ...[truncationNote(run)].filter(Boolean),
    h("p", { class: "note", style: { "margin-bottom": "0.75rem" } }, `Labels from ${run.sections.taxonomy}.`),
    h(
      "div",
      { class: "table-wrap" },
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Section"), h("th", {}, "Canonical label"), h("th", { class: "num" }, "Score"), hasLevels ? h("th", {}, "Level") : null)), h("tbody", {}, rows)),
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
setupSelect();
loadOptions().catch((error) => setStatus(`Could not load the options: ${error.message}`, true));
refreshHistory();
