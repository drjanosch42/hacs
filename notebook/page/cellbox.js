// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// cellbox.js -- the notebook cell as two parts (slot-24 NB-OUTBOX): a CODE box (the editable source,
// its own "Copy code") and, below it, a separate read-only OUTPUT box (its own "Copy output", the
// plain text behind every display whatever the LaTeX switch says). The output box is hidden until
// the cell has run; a re-run clears it when the run starts, so it is replaced, never appended to.
// Also here, as plain functions testable under node (notebook/gateway/test_render.mjs, with a
// minimal fake DOM): the run queue, the Colab-style state label ([ ] idle, [..] queued, [*] running,
// [k] ran as execution k) and the three exports (text, code only, JSON with code and outputs as
// separate fields). No session talk and no TeX in this file: app.js drives it; render.js renders.
"use strict";

const OCCells = (() => {
  const R = () => (typeof OCRender !== "undefined" ? OCRender : null);
  const el = (tag, cls, text) => R().el(tag, cls, text);

  // ---- the state label (it lives in the gutter of the code box) ----
  function labelText(state, count) {
    switch (state) {
      case "queued": return "[…]";
      case "running": return "[*]";
      case "done": case "error": return `[${count}]`;
      default: return "[ ]";
    }
  }

  // ---- building a cell: code box + output box inside one wrapper ----
  // opts: { kind: "code" | "md", title }. A "md" cell is a text cell (slot-24 NB-FILE): shown as text, never run.
  function createCell(n, src, opts) {
    const kind = opts && opts.kind === "md" ? "md" : "code";
    const root = el("div", kind === "md" ? "cell-wrap md-wrap" : "cell-wrap");
    const codeBox = el("div", "cell");
    const bar = el("div", "cell-bar");
    const label = el("span", "label", kind === "md" ? "text" : labelText("idle"));
    label.title = kind === "md" ? "a text cell: not executed" : "[ ] not run, […] queued, [*] running, [k] ran as execution k";
    const titleEl = el("input", "cell-title");
    titleEl.type = "text";
    titleEl.value = (opts && opts.title) || "";
    titleEl.placeholder = "title";
    titleEl.title = "the cell's title (saved as `# %% title` in a .oc notebook file)";
    titleEl.setAttribute("aria-label", `title of cell ${n}`);
    const bRun = kind === "md" ? null : el("button", "", "Run");
    if (bRun) bRun.title = "Run this cell (Shift+Enter: run and go to the next cell)";
    const bEdit = kind === "md" ? el("button", "", "Edit") : null;
    const bCopyCode = el("button", "", "Copy code");
    bCopyCode.title = "copy this cell's code (never its output)";
    const bCopyTeX = kind === "md" ? el("button", "", "Copy as LaTeX") : null;
    if (bCopyTeX) bCopyTeX.title = "copy this text cell's source (its $...$ / $$...$$ formulas are the TeX the page renders)";
    const bDel = el("button", "", "Delete");
    for (const b of [bRun, bEdit, bCopyCode, bCopyTeX, bDel]) if (b) b.type = "button";
    bar.append(label, titleEl);
    if (bRun) bar.append(bRun);
    if (bEdit) bar.append(bEdit);
    bar.append(el("span", "spacer"), bCopyCode);
    if (bCopyTeX) bar.append(bCopyTeX);
    bar.append(bDel);
    const ta = el("textarea", "src");
    ta.spellcheck = false;
    ta.value = src || "";
    ta.rows = Math.max(2, (src || "").split("\n").length);
    ta.setAttribute("aria-label", `code of cell ${n}`);
    codeBox.append(bar, ta);
    // A text cell shows its text (text nodes only, never HTML); the textarea appears on Edit.
    const view = kind === "md" ? el("div", "mdtext") : null;
    if (view) codeBox.append(view);

    // The output box: no card, no heavy border; a thin left rule (style.css). Its copy button shows on
    // hover / focus (always on touch devices) but stays in the tab order.
    const outbox = el("div", "outbox");
    outbox.hidden = true;
    outbox.setAttribute("role", "region");
    outbox.setAttribute("aria-label", `output of cell ${n}`);
    const outLabel = el("span", "outlabel muted", "");
    const meta = el("span", "meta muted");
    const bCopyOut = el("button", "outcopy", "Copy output");
    bCopyOut.type = "button";
    bCopyOut.title = "copy this cell's output as plain text (the canonical text behind every display, LaTeX on or off)";
    const obar = el("div", "outbar");
    const bCopyOutTeX = el("button", "outcopy", "Copy as LaTeX");
    bCopyOutTeX.type = "button";
    bCopyOutTeX.title = "copy the TeX of this cell's values (display formulas), as the page renders them";
    obar.append(outLabel, meta, el("span", "spacer"), bCopyOutTeX, bCopyOut);
    const out = el("div", "output");
    out.setAttribute("aria-live", "polite");
    outbox.append(obar, out);

    if (kind === "md") root.append(codeBox); else root.append(codeBox, outbox);
    const cell = { n, kind, root, codeBox, ta, titleEl, view, bEdit, label, bRun, bCopyCode, bDel, outbox, outLabel, meta, out, bCopyOut,
                   state: "idle", settled: "idle", count: 0, editing: false, foldOpen: !!(opts && opts.open) };
    if (kind === "md") {
      if (bEdit) bEdit.addEventListener("click", () => setMdEditing(cell, !cell.editing));
      setMdEditing(cell, false);
    }
    bCopyCode.addEventListener("click", () => R().copyText(ta.value, bCopyCode));
    bCopyOut.addEventListener("click", () => R().copyText(outputPlain(cell), bCopyOut));
    bCopyOutTeX.addEventListener("click", () => R().copyText(R().outputTeX(out), bCopyOutTeX));
    if (bCopyTeX) bCopyTeX.addEventListener("click", () => R().copyText(textCellTeX(ta.value), bCopyTeX));
    return cell;
  }

  // A text cell: view mode shows the text, edit mode the textarea (an empty cell starts in edit mode).
  function setMdEditing(cell, on) {
    if (cell.kind !== "md") return;
    if (!on) renderMdView(cell);
    cell.editing = !!on || cell.ta.value === "";
    cell.ta.hidden = !cell.editing;
    cell.view.hidden = cell.editing;
    cell.bEdit.textContent = cell.editing ? "Done" : "Edit";
  }

  // (slot-26 NBMATH A1) The view of a text cell: its text with $...$ / $$...$$ through KaTeX and a minimal markdown
  // (render.js renderMarkdown; the source verbatim with LaTeX off). A text cell titled "Reading the output..." is a
  // reading panel (mathematics <-> what HACS computed): it folds, closed by default, open in a showcase notebook
  // (opts.open; app.js opens a file named showcase_*.oc that way). The fold state is kept across re-renders.
  const isReading = (cell) => /^Reading\b/.test(cell.titleEl.value.trim());
  function renderMdView(cell) {
    const text = cell.ta.value || "(empty text cell)";
    cell.view.textContent = "";
    const r = R();
    if (!r || typeof r.renderMarkdown !== "function") { cell.view.textContent = text; return; }
    if (!isReading(cell)) { r.renderMarkdown(cell.view, text); return; }
    const d = el("details", "reading");
    d.open = !!cell.foldOpen;
    d.addEventListener("toggle", () => { cell.foldOpen = d.open; });
    const sum = el("summary", "", "Reading the output: the mathematics, and what HACS computed");
    const body = el("div", "mdbody");
    r.renderMarkdown(body, text);
    d.append(sum, body);
    cell.view.appendChild(d);
  }
  // Re-render every text cell in view mode (the LaTeX switch).
  function rerenderText(cells) { for (const c of cells || []) if (c.kind === "md" && !c.editing) renderMdView(c); }

  // The plain text of the output box: exactly what render.js says (canonical text per value).
  // (slot-27 NB-A' C1) A text cell's TeX: its source verbatim (what the page parses and renders; the formulas are its $...$ / $$...$$).
  function textCellTeX(src) { return src; }
  function outputPlain(cell) { return R().outputText(cell.out); }

  // ---- states ----
  function setState(cell, state, count) {
    cell.state = state;
    if (count !== undefined) cell.count = count;
    if (state === "idle" || state === "done" || state === "error") cell.settled = state;
    cell.label.textContent = labelText(state, cell.count);
    const cl = cell.codeBox.classList;
    cl.remove("queued"); cl.remove("running");
    if (state === "queued" || state === "running") cl.add(state);
    cell.codeBox.setAttribute("aria-busy", state === "running" ? "true" : "false");
    cell.outbox.classList.remove("error");
    if (cell.settled === "error") cell.outbox.classList.add("error");
  }
  // A queued cell that never ran returns to its last settled state (a cancelled run).
  function cancelQueued(cell) { setState(cell, cell.settled); }

  // The run starts: the old output goes away now (replaced, not appended to); the box shows with the first output.
  function beginRun(cell, count) {
    cell.out.textContent = "";
    cell.meta.textContent = "";
    cell.outLabel.textContent = "";
    cell.outbox.hidden = true;
    setState(cell, "running", count);
  }
  function reveal(cell) { cell.outbox.hidden = false; }
  function finishRun(cell, ok, metaText) {
    cell.meta.textContent = metaText || "";
    cell.outLabel.textContent = `Out ${labelText(ok ? "done" : "error", cell.count)}`;
    reveal(cell);
    setState(cell, ok ? "done" : "error");
  }

  // ---- the run queue: one cell at a time, in submission order ----
  function makeQueue(exec) {
    const q = [];
    let busy = false;
    function pump() {
      if (busy || !q.length) return;
      busy = true;
      const job = q.shift();
      let p;
      try { p = Promise.resolve(exec(job)); } catch (e) { p = Promise.resolve(); }
      p.catch(() => {}).then(() => { busy = false; pump(); });
    }
    return { push(job) { q.push(job); pump(); }, clear() { return q.splice(0); }, pending: () => q.length, busy: () => busy };
  }

  // ---- exports; cells: [{n, count, status, code, outText}] ----
  const trimEnd = (s) => s.replace(/\s+$/, "");
  function exportText(cells) {
    const parts = [];
    for (const c of cells) {
      if (c.kind === "md") { parts.push(trimEnd(c.code).split("\n").map((l) => (l ? "# " + l : "#")).join("\n") + "\n\n"); continue; }
      parts.push(`# [${c.count || c.n}]\n${trimEnd(c.code)}\n`);
      if (c.outText) parts.push(c.outText.split("\n").map((l) => (l ? "#> " + l : l)).join("\n"));
      parts.push("\n");
    }
    return parts.join("");
  }
  // Code only: every cell's source, separated by a blank line, nothing else (replayable with --script).
  function exportCode(cells) {
    return cells.filter((c) => c.kind !== "md").map((c) => trimEnd(c.code)).filter((s) => s !== "").join("\n\n") + "\n";
  }
  // Code and outputs as separate fields.
  function exportJSON(cells) {
    return JSON.stringify({
      schema: "oc-notebook-export/1",
      cells: cells.map((c, i) => ({ index: i + 1, kind: c.kind === "md" ? "md" : "code", title: c.title || "",
                                    execution_count: c.count || null, status: c.status || "idle",
                                    code: c.code, output: c.outText || "" })),
    }, null, 2) + "\n";
  }
  function snapshot(cells) {
    return cells.map((c) => ({ n: c.n, kind: c.kind || "code", title: c.titleEl ? c.titleEl.value.trim() : "",
                              count: c.settled === "idle" ? 0 : c.count, status: c.settled, code: c.ta.value, outText: outputPlain(c) }));
  }

  // ---- the notebook: the cell list, the preamble of a loaded file, the unsaved mark (slot-24 NB-FILE) ----
  // host: { mount(cell, afterCell | null), wire(cell), unmount(cell) } -- the DOM side (app.js). `cells` is one array,
  // mutated in place, so the page and this object share it. Every mutation that changes the notebook sets `dirty`.
  function makeNotebook(host) {
    const nb = { cells: [], preamble: "", dirty: false, counter: 0, execCount: 0, fileName: "" };
    nb.add = (src, after, opts) => {
      const cell = createCell(++nb.counter, src, opts);
      host.wire(cell);
      host.mount(cell, after || null);
      if (after && nb.cells.includes(after)) nb.cells.splice(nb.cells.indexOf(after) + 1, 0, cell); else nb.cells.push(cell);
      nb.dirty = true;
      return cell;
    };
    nb.remove = (cell) => {
      const i = nb.cells.indexOf(cell);
      if (i < 0) return;
      host.unmount(cell); nb.cells.splice(i, 1); nb.dirty = true;
    };
    nb.clear = () => { for (const c of nb.cells.slice()) { host.unmount(c); } nb.cells.length = 0; };
    // Insert a parsed file (OCFile.parse) below `after` (null: at the end); returns the new cells.
    nb.insertParsed = (parsed, after) => {
      const specs = OCFile.insertCells([], -1, parsed);
      const made = [];
      let at = after || null;
      for (const sp of specs) { at = nb.add(sp.code, at, sp); made.push(at); }
      nb.dirty = true;
      return made;
    };
    // Open a parsed file as the whole notebook: the old cells go, the session restarts (the caller does that),
    // the execution count starts again, nothing is unsaved.
    nb.openParsed = (parsed, name) => {
      nb.clear();
      const nbk = OCFile.replaceAll(parsed);
      nb.preamble = nbk.preamble;
      nb.execCount = 0;
      let last = null;
      const open = /^showcase_/.test(String(name || ""));   // (slot-26 NBMATH) a showcase opens its reading panels
      for (const sp of nbk.cells) last = nb.add(sp.code, last, open ? Object.assign({}, sp, { open: true }) : sp);
      nb.fileName = name || "";
      nb.dirty = false;
      return nb.cells;
    };
    nb.toFile = () => OCFile.serialise({ preamble: nb.preamble, cells: snapshot(nb.cells) });
    // Restart session: the execution count starts again (as Colab does); the cells and their outputs stay.
    nb.restart = () => { nb.execCount = 0; };
    return nb;
  }

  return { createCell, setState, cancelQueued, beginRun, reveal, finishRun, outputPlain, makeQueue, labelText, setMdEditing, rerenderText, makeNotebook,
           exportText, exportCode, exportJSON, snapshot };
})();

if (typeof module !== "undefined" && module.exports) module.exports = OCCells;
