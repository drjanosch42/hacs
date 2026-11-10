// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// test_render.mjs -- the page renderer's fixture test (slot-24 NB-RENDER R2; plain node, no dependency
// beyond the vendored KaTeX). It loads notebook/page/render.js and notebook/vendor/katex/katex.min.js as
// CommonJS modules and walks notebook/gateway/fixtures/envelopes.ndjson, one frozen oc-serve/1 envelope
// per line, captured from a real `--serve` session ({"name", "expect", "env"[, "contains"]}):
//   expect = "tex"  : toTeX(env) is non-null, KaTeX accepts it with throwOnError: true, and (if given)
//                     it contains every string of `contains`;
//   expect = "text" : toTeX(env) is null (the page shows the canonical text / tables).
// Then the LaTeX switch: with LaTeX off no envelope produces TeX, and the text behind every value
// (plainText) is its canonical text, unchanged, in both states.
// Every envelope carrying a `hint`, at any depth, must typeset too (hinted=<h>).
// Last line: RENDER_TEST fixtures=<k> tex=<t> text=<x> hinted=<h> failures=<f> verdict=OK|FAIL. Exit 0 iff OK.
//
//     node notebook/gateway/test_render.mjs
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const katex = require(join(here, "..", "vendor", "katex", "katex.min.js"));
globalThis.katex = katex;
const R = require(join(here, "..", "page", "render.js"));

const lines = readFileSync(join(here, "fixtures", "envelopes.ndjson"), "utf8").split("\n").filter((l) => l.trim() && !l.startsWith("#"));
let failures = 0, tex = 0, text = 0;
const fail = (msg) => { ++failures; console.log("FAIL " + msg); };
const fixtures = lines.map((l) => JSON.parse(l));

R.setTeX(true);
for (const f of fixtures) {
  const t = R.toTeX(f.env, 0);
  if (f.expect === "tex") {
    ++tex;
    if (t === null) { fail(`${f.name}: no TeX`); continue; }
    try { katex.renderToString(t, { throwOnError: true, displayMode: f.env.t === "Matrix", trust: false, strict: "ignore" }); }
    catch (e) { fail(`${f.name}: KaTeX refuses ${JSON.stringify(t)}: ${e.message}`); continue; }
    for (const c of f.contains || []) if (!t.includes(c)) fail(`${f.name}: TeX lacks ${JSON.stringify(c)}: ${t}`);
    console.log(`ok   ${f.name}: ${t.length > 160 ? t.slice(0, 160) + "..." : t}`);
  } else {
    ++text;
    if (t !== null) fail(`${f.name}: expected text, got TeX ${t}`);
    else console.log(`ok   ${f.name}: canonical text`);
  }
  if (R.plainText(f.env) !== f.env.canonical) fail(`${f.name}: plainText != canonical (LaTeX on)`);
}

// Every hinted envelope, at any depth (Record values, List elements, members, calls), typesets.
let hinted = 0;
function walk(env, where) {
  if (!env || typeof env !== "object") return;
  if (env.hint) {
    ++hinted;
    const t = R.toTeX(env, 0);
    if (t === null) fail(`${where}: hint ${JSON.stringify(env.hint)} gave no TeX (canonical ${env.canonical})`);
    else {
      try { katex.renderToString(t, { throwOnError: true, trust: false, strict: "ignore" }); }
      catch (e) { fail(`${where}: KaTeX refuses hinted ${JSON.stringify(t)}: ${e.message}`); }
    }
  }
  const v = env.v;
  if (env.t === "List" && Array.isArray(v)) v.forEach((x, i) => walk(x, `${where}[${i}]`));
  else if (env.t === "Record" && Array.isArray(v)) for (const pr of v) walk(pr[1], `${where}.${pr[0]}`);
  else if (v && typeof v === "object" && v.members) {
    for (const k of Object.keys(v.members)) walk(v.members[k], `${where}.${k}`);
    for (const c of v.calls || []) walk(c[2], `${where}.${c[0]}(${c[1]})`);
  }
}
for (const f of fixtures) walk(f.env, f.name);

// The switch: presentation only.
R.setTeX(false);
for (const f of fixtures) {
  if (R.toTeX(f.env, 0) !== null) fail(`${f.name}: TeX with LaTeX off`);
  if (R.plainText(f.env) !== f.env.canonical) fail(`${f.name}: plainText != canonical (LaTeX off)`);
}
R.setTeX(true);
if (!R.texEnabled()) fail("the switch does not come back on");

// Pure pieces: signs over Z, the high -> low order label, groups, partitions.
const P = R._pure;
const eq = (got, want, what) => { if (got !== want) fail(`${what}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`); };
eq(P.polyTeX(["1", "-3", "0", "2"], "high_to_low", "X"), "X^{3} - 3X^{2} + 2", "polyTeX high_to_low signed");
eq(P.polyTeX([4, 346, 72, 1], "low_to_high", "X"), "X^{3} + 72X^{2} + 346X + 4", "polyTeX low_to_high");
eq(P.polyTeX([], "low_to_high", "X"), "0", "polyTeX zero");
eq(P.polyTeX([1, 2], "sideways", "X"), null, "polyTeX refuses an unknown order");
eq(P.qexpTeX([0, 1, 5, 0]), "q + 5q^{2} + O(q^{4})", "qexpTeX");
eq(P.partitionTeX(["3", "1", "1"]), "(3,\\ 1^{2})", "partitionTeX");
eq(P.cyclicSumTeX(5, ["2", "1", "1"]), "\\mathbb{Z}/5^{2} \\oplus (\\mathbb{Z}/5)^{2}", "cyclicSumTeX");
let threw = false;
try { P.polyTeX(["1\\over 2"], "low_to_high", "X"); } catch (e) { threw = true; }
if (!threw) fail("polyTeX accepted a non-integer string");

// (slot-24 NB-RENDER R3) the hint kinds cyclic / eigen / lesmap, and the statement text behind a frame.
eq(P.hintTeX({ t: "List", v: [{ t: "Int", v: 2 }, { t: "Int", v: 1 }], hint: { kind: "cyclic", ell: 7 } }),
   "\\mathbb{Z}/7^{2} \\oplus \\mathbb{Z}/7", "hint cyclic");
eq(P.hintTeX({ t: "List", v: [], hint: { kind: "cyclic", ell: 7 } }), "0", "hint cyclic, no torsion");
eq(P.eigenTeX({ t: "Poly", v: { ell: 691, deg: 0, order: "low_to_high", coeffs: [252] } }, { kind: "eigen", f: 1 }),
   "252 \\in \\mathbb{F}_{691}", "eigen f = 1");
eq(P.eigenTeX({ t: "Poly", v: { ell: 5, deg: 1, order: "low_to_high", coeffs: [1, 3] } }, { kind: "eigen", f: 2 }),
   "3w + 1 \\in \\mathbb{F}_{5}[w]/(\\pi_w) \\cong \\mathbb{F}_{5^{2}}", "eigen f = 2");
eq(P.lesmapTeX({ t: "Matrix", v: { rows: 1, cols: 1, p: 7, entries: [[3]] } }, { kind: "lesmap", map: "delta", n: 16, n_up: 24 }),
   "\\delta\\colon H^{0}(Q_{16}) \\to H^{1}(\\mathrm{Sym}^{16}\\otimes\\det),\\quad \\begin{pmatrix} 3 \\end{pmatrix} \\in M_{1\\times 1}(\\mathbb{F}_{7})",
   "lesmap delta");
eq(P.lesmapTeX({ t: "Matrix", v: { rows: 1, cols: 1, p: 7, entries: [[3]] } }, { kind: "lesmap", map: "evil}" }), null,
   "lesmap refuses an unknown map name");
eq(P.stmtSource("M = sl2z_model(); H = cohomology(L); H\nx", 1, 38), "H", "stmtSource: a bare name before a newline");
eq(P.stmtSource("a = 1; f(x, \"; #\", [1; 2]); b", 1, 8), "f(x, \"; #\", [1; 2])", "stmtSource: strings and brackets");
eq(P.stmtSource("  T.A(1)  # comment", 1, 3), "T.A(1)", "stmtSource: a trailing comment");
eq(P.stmtSource("x", 3, 1), null, "stmtSource: no such line");

// (slot-24 NB-OUTBOX) The cell as two boxes, under a minimal fake DOM (no jsdom): the output box is separate
// from the code box and hidden until the cell has run, a re-run REPLACES the output, errors are marked, the
// copy of the output is the plain text with LaTeX on (not the TeX), the code copy is code only, the queue runs
// one cell at a time in order, and the exports keep code and output apart.
{
  class FNode {
    constructor(tag) { this.tagName = tag; this.className = ""; this.children = []; this.parentNode = null; this.dataset = {}; this.attrs = {}; this.hidden = false; this.handlers = {}; this._text = ""; this.value = ""; }
    get classList() { const n = this; const set = () => new Set(n.className.split(/\s+/).filter(Boolean));
      return { add: (c) => { const x = set(); x.add(c); n.className = [...x].join(" "); }, remove: (c) => { const x = set(); x.delete(c); n.className = [...x].join(" "); }, contains: (c) => set().has(c) }; }
    get textContent() { return this.children.length ? this.children.map((c) => c.textContent).join("") : this._text; }
    set textContent(v) { for (const c of this.children) c.parentNode = null; this.children = []; this._text = String(v); }
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
    append(...cs) { for (const c of cs) this.appendChild(c); }
    insertBefore(c, ref) { c.parentNode = this; this.children.splice(this.children.indexOf(ref), 0, c); }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    addEventListener(t, f) { (this.handlers[t] = this.handlers[t] || []).push(f); }
    click() { for (const f of this.handlers.click || []) f(); }
    remove() { if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; }
  }
  globalThis.document = { createElement: (t) => new FNode(t), createTextNode: (t) => { const n = new FNode("#text"); n._text = String(t); return n; }, body: new FNode("body") };
  globalThis.window = { isSecureContext: true };
  const copied = [];
  Object.defineProperty(globalThis, "navigator", { value: { clipboard: { writeText: (t) => { copied.push(t); return Promise.resolve(); } } }, configurable: true });
  globalThis.katex = { render: (tex, node) => { node.textContent = "<<TEX " + tex + ">>"; } };   // LaTeX "on" without a real DOM
  globalThis.OCRender = R;
  const C = require(join(here, "..", "page", "cellbox.js"));
  const mat = fixtures.find((f) => f.env.t === "Matrix" && f.expect === "tex");
  if (!mat) fail("outbox: no Matrix fixture");
  else {
    R.setTeX(true);
    const cell = C.createCell(1, "M = matrix(7, [[1,2],[3,4]]);\nM");
    const codeText = () => cell.codeBox.textContent;
    if (cell.outbox.parentNode !== cell.root || cell.codeBox.parentNode !== cell.root || cell.outbox === cell.codeBox) fail("outbox: not siblings");
    if (cell.codeBox.children.includes(cell.out) || cell.codeBox.children.includes(cell.outbox)) fail("outbox: output inside the code box");
    if (!cell.outbox.hidden) fail("outbox: visible before the cell has run");
    if (cell.label.textContent !== "[ ]") fail("outbox: idle label " + cell.label.textContent);
    C.setState(cell, "queued");
    if (cell.label.textContent !== "[\u2026]" || !cell.codeBox.classList.contains("queued")) fail("outbox: queued state");
    C.beginRun(cell, 1);
    if (cell.label.textContent !== "[*]" || !cell.codeBox.classList.contains("running")) fail("outbox: running state");
    cell.out.appendChild(el0("pre", "out", "first run\n"));
    cell.out.appendChild(R.renderValue(mat.env));
    C.finishRun(cell, true, "ok (exit 0)");
    if (cell.outbox.hidden) fail("outbox: hidden after the run");
    if (cell.label.textContent !== "[1]") fail("outbox: count label " + cell.label.textContent);
    if (!cell.out.textContent.includes("<<TEX")) fail("outbox: LaTeX not on in the output");
    if (codeText().includes("first run") || codeText().includes("<<TEX")) fail("outbox: output leaked into the code box");
    cell.bCopyOut.click();
    const want = "first run\n" + mat.env.canonical + "\n";
    if (copied[copied.length - 1] !== want) fail(`outbox: copy output with LaTeX on: ${JSON.stringify(copied[copied.length - 1])} != ${JSON.stringify(want)}`);
    if (copied[copied.length - 1].includes("TEX")) fail("outbox: copy contains TeX");
    cell.ta.value = "M = matrix(7, [[1,2],[3,4]]);\nM";
    cell.bCopyCode.click();
    if (copied[copied.length - 1] !== cell.ta.value) fail("outbox: copy code is not the code only");
    // re-run replaces
    C.setState(cell, "queued");
    C.beginRun(cell, 2);
    if (cell.out.children.length !== 0 || !cell.outbox.hidden) fail("outbox: re-run did not clear the old output at its start");
    cell.out.appendChild(el0("pre", "out err", "parse_error: boom"));
    C.finishRun(cell, false, "error (exit 2)");
    if (cell.out.textContent.includes("first run") || cell.out.children.length !== 1) fail("outbox: re-run appended instead of replacing");
    if (!cell.outbox.classList.contains("error") || cell.label.textContent !== "[2]") fail("outbox: error state / count");
    // a queued cell that is cancelled goes back to its settled state
    C.setState(cell, "queued"); C.cancelQueued(cell);
    if (cell.label.textContent !== "[2]" || cell.codeBox.classList.contains("queued")) fail("outbox: cancelQueued");
    // exports: separate fields, code only
    const snap = C.snapshot([cell]);
    const j = JSON.parse(C.exportJSON(snap));
    if (j.schema !== "oc-notebook-export/1" || j.cells[0].code !== cell.ta.value || j.cells[0].output !== "parse_error: boom\n" ||
        j.cells[0].execution_count !== 2 || j.cells[0].status !== "error") fail("outbox: JSON export " + JSON.stringify(j));
    if (C.exportCode(snap) !== cell.ta.value + "\n") fail("outbox: code export carries more than the code");
    if (!C.exportText(snap).includes("#> parse_error: boom")) fail("outbox: text export");
    // the queue: one at a time, in order
    const order = [];
    let active = 0, maxActive = 0;
    const q = C.makeQueue((job) => new Promise((res) => { ++active; maxActive = Math.max(maxActive, active); order.push(job.id); setTimeout(() => { --active; res(); }, 5); }));
    for (const id of [1, 2, 3]) q.push({ id });
    const dropped = q.clear();
    await new Promise((r) => setTimeout(r, 60));
    if (order.join() !== "1" || dropped.length !== 2 || maxActive !== 1) fail(`outbox: queue order=${order} dropped=${dropped.length} max=${maxActive}`);
    const q2order = [];
    const q2 = C.makeQueue((job) => new Promise((res) => { q2order.push(job.id); setTimeout(res, 3); }));
    for (const id of [1, 2, 3]) q2.push({ id });
    await new Promise((r) => setTimeout(r, 60));
    if (q2order.join() !== "1,2,3") fail("outbox: queue order " + q2order);
    console.log("ok   outbox: separate output box, state labels, re-run replaces, copy = plain text, exports, queue");
  }
  function el0(tag, cls, text) { return R.el(tag, cls, text); }
}
// (slot-24 NB-FILE) The notebook file format ("# %%" cells; notebook/page/ocfile.js) and the notebook object of
// cellbox.js (insert a file as cells, open a file as the whole session), under the same minimal fake DOM.
{
  const F = require(join(here, "..", "page", "ocfile.js"));
  globalThis.OCFile = F;
  const C = require(join(here, "..", "page", "cellbox.js"));
  const tourDir = join(here, "..", "examples", "notebook");
  const E = (got, want, what) => { if (got !== want) fail(`file: ${what}: ${JSON.stringify(got)} != ${JSON.stringify(want)}`); };
  const throwsMsg = (fn) => { try { fn(); } catch (e) { return e.message; } return null; };

  // the format: a canonical file is a fixed point; the canonical form of messy input
  const canon = "# preamble\n# SPDX\n\n# %% Setup\nx = 1;\ny = 2;\n\n# %% [md] Notes\n# a line\n#\n#  indented\n\n# %%\nz = 3;\n";
  const p = F.parse(canon);
  E(p.preamble, "# preamble\n# SPDX", "preamble kept");
  E(p.cells.length, 3, "three cells");
  E(JSON.stringify(p.cells.map((c) => [c.kind, c.title])), JSON.stringify([["code", "Setup"], ["md", "Notes"], ["code", ""]]), "kinds and titles");
  E(p.cells[0].code, "x = 1;\ny = 2;", "code body");
  E(p.cells[1].code, "a line\n\n indented", "text body without its '#' and one space");
  E(F.serialise(p), canon, "round trip of a canonical file");
  E(F.canonical(F.canonical(canon)), F.canonical(canon), "canonicalisation is idempotent");
  E(F.canonical("﻿# %% A \r\nx = 1;\r\n\r\n\r\n\r\n# %%\r\n\r\ny\r\n\r\n"), "# %% A\nx = 1;\n\n# %%\ny\n", "BOM, CRLF, blank lines, trailing newline");
  E(F.canonical("# %% A\nx = 1;"), "# %% A\nx = 1;\n", "a missing final newline is added");
  E(F.canonical("# %% A\r\nx;\rz;\r"), "# %% A\nx;\nz;\n", "a lone CR is a line break");
  // edge cases
  const one = F.parse("print(1);\n# a comment\nprint(2);\n");
  E(one.cells.length, 1, "no separator: one cell"); E(one.preamble, "", "no separator: no preamble");
  E(one.cells[0].code, "print(1);\n# a comment\nprint(2);", "no separator: the whole text, comments included");
  E(F.parse("").cells.length, 1, "empty file: one cell"); E(F.parse("\n\n").cells[0].code, "", "empty file: empty cell");
  E(F.parse("# %%\n# %%\n").cells.length, 2, "empty bodies are cells");
  E(F.serialise(F.parse("# %%\n# %%\n")), "# %%\n\n# %%\n", "empty bodies round trip");
  E(F.parse("# %%%\nx\n").cells.length, 1, "'# %%%' is not a separator"); E(F.parse("#%%\nx\n").cells[0].code, "#%%\nx", "'#%%' is not a separator");
  E(F.parse("x\n  # %% not at the margin\n").cells.length, 1, "an indented '# %%' is not a separator");
  E(F.parse("# %%\tTitle  \nx\n").cells[0].title, "Title", "tab after the marker, title trimmed");
  E(F.parse("# preamble only\n\n# %% A\n").preamble, "# preamble only", "preamble before the first cell");
  E(F.parse("# %% A\nx\n").preamble, "", "no preamble when the file starts with a cell");
  E(F.serialise({ preamble: "", cells: [{ kind: "code", title: "[md] sneaky", code: "x" }] }), "# %% sneaky\nx\n", "a code title cannot read as [md]");
  E(F.serialise({ preamble: "", cells: [{ kind: "code", title: "a\nb", code: "\n\nx\n\n" }] }), "# %% a b\nx\n", "a title is one line; blank ends of the code dropped");
  E(F.serialise({ preamble: "", cells: [] }), "", "no cells, no preamble: empty file");
  E(F.serialise({ preamble: "# p", cells: [] }), "# p\n", "preamble only");
  E(F.serialise({ preamble: "", cells: [{ kind: "md", title: "T", code: "a\n\nb" }] }), "# %% [md] T\n# a\n#\n# b\n", "a text cell is all comments");
  E(F.parse("# %% [md] T\n# a\n\n# b\n").cells[0].code, "a\n\nb", "a blank line inside a text cell is allowed");
  E(F.parse("# %% [md]\n#x\n").cells[0].code, "x", "'#x' is text 'x'");
  const bad = throwsMsg(() => F.parse("# %% A\nx\n\n# %% [md] B\n\n# fine\nnot a comment\n"));
  if (!bad || !bad.startsWith("line 7:")) fail("file: a text cell with a non-comment line must be refused with its line number: " + bad);
  E(F.parse("# %% [mdx] T\nx\n").cells[0].kind, "code", "[mdx] is not [md]");
  // every body line of a text cell is a comment, so a notebook with text cells is still a script
  for (const l of F.serialise(F.parse(canon)).split("\n")) if (l && !l.startsWith("#") && !/^(x|y|z) = /.test(l)) fail("file: stray line " + l);

  // the shipped tour: every file is canonical (byte-identical round trip), has cells, and tour_all.oc is their concatenation
  const { readdirSync } = require("node:fs");
  const tour = readdirSync(tourDir).filter((n) => /^\d\d_.*\.oc$/.test(n)).sort();
  E(tour.length, 10, "ten tour files");
  const allCells = [];
  for (const n of tour.concat(["tour_all.oc"])) {
    const t = readFileSync(join(tourDir, n), "utf8");
    const q = F.parse(t);
    if (!(q.cells.length >= 3)) fail(`file: ${n}: fewer than three cells`);
    if (!q.preamble.startsWith("# SPDX-FileCopyrightText")) fail(`file: ${n}: the SPDX header is not the preamble`);
    if (q.cells.some((c) => c.kind === "code" && !c.title)) fail(`file: ${n}: an untitled code cell`);
    E(F.serialise(q), t, `${n}: byte-identical round trip`);
    if (n !== "tour_all.oc") allCells.push(...q.cells);
    else E(JSON.stringify(q.cells), JSON.stringify(allCells), "tour_all.oc is the ten files' cells in order");
  }

  // the notebook object: insert a file as cells; open a file as the whole session
  const host = (log) => ({ order: [], mount(c, after) { const i = after ? this.order.indexOf(after) + 1 : this.order.length; this.order.splice(i, 0, c); log.push("mount " + c.n); },
                           unmount(c) { this.order.splice(this.order.indexOf(c), 1); log.push("unmount " + c.n); }, wire() {} });
  const log = [];
  const h = host(log);
  const nb = C.makeNotebook(h);
  const [a, b, c3] = ["a = 1;", "b = 2;", "c = 3;"].map((s) => nb.add(s, nb.cells[nb.cells.length - 1] || null));
  E(nb.cells.length, 3, "three cells added"); nb.dirty = false; nb.execCount = 7;
  const names = () => nb.cells.map((c) => c.ta.value).join("|");
  const made = nb.insertParsed(F.parse("# head\n\n# %% T1\nx1;\n\n# %% [md] T2\n# note\n"), a);
  E(names(), "a = 1;|# head|x1;|note|b = 2;|c = 3;", "insert below the current cell: the preamble is a first cell, order kept");
  E(JSON.stringify(made.map((c) => c.kind + ":" + c.titleEl.value)), JSON.stringify(["code:", "code:T1", "md:T2"]), "inserted kinds and titles");
  E(h.order.length === nb.cells.length && h.order.every((c, i) => c === nb.cells[i]), true, "the DOM order is the cell order");
  E(nb.execCount, 7, "insert keeps the session (execution count)"); E(nb.dirty, true, "insert marks the notebook unsaved");
  nb.insertParsed(F.parse("tail;\n"), null);
  E(nb.cells[nb.cells.length - 1].ta.value, "tail;", "insert with no current cell: at the end; a file without a separator is one cell");
  E(nb.cells[3].kind, "md", "the text cell is a text cell");
  // title edits and a text cell reach the saved file; outputs never do
  nb.cells[2].titleEl.value = "  Edited title ";
  nb.cells[2].state = "done"; nb.cells[2].settled = "done"; nb.cells[2].count = 4;
  nb.cells[2].out.textContent = "OUTPUT-MARKER";
  const saved = nb.toFile();
  if (!saved.includes("# %% Edited title\nx1;\n") || saved.includes("OUTPUT-MARKER")) fail("file: save carries the edited title, never the output: " + saved);
  if (!saved.includes("# %% [md] T2\n# note\n")) fail("file: a text cell is saved as '# %% [md]' with '#' lines: " + saved);
  E(C.exportCode(C.snapshot(nb.cells)).includes("note"), false, "Export code leaves the text cells out");
  const jj = JSON.parse(C.exportJSON(C.snapshot(nb.cells)));
  E(jj.cells[3].kind + "/" + jj.cells[2].title, "md/Edited title", "JSON export carries kind and title");
  // open as a session: cells replaced, execution count restarts, nothing unsaved, the preamble kept and saved again
  const oldCells = nb.cells.slice();
  const text = "# SPDX x\n\n# %% One\nm = 1;\n\n# %% Two\nm\n";
  nb.openParsed(F.parse(text), "nb.oc");
  E(names(), "m = 1;|m", "open replaces every cell");
  E(oldCells.some((c) => nb.cells.includes(c)), false, "none of the old cells survive");
  E(h.order.length, 2, "the old cells left the DOM");
  E(nb.execCount, 0, "open restarts the execution count"); E(nb.dirty, false, "open leaves nothing unsaved"); E(nb.fileName, "nb.oc", "the file name is kept");
  E(nb.cells.every((c) => c.label.textContent === "[ ]"), true, "opened cells are not run");
  E(nb.toFile(), text, "open then save is byte-identical for a canonical file");
  nb.execCount = 3; nb.restart(); E(nb.execCount, 0, "Restart session resets the execution count");
  nb.cells[0].ta.value = "m = 2;"; E(nb.toFile().includes("m = 2;"), true, "save reads the current source");
  console.log("ok   file: '# %%' format (round trip, canonical form, edge cases, text cells), the tour files, insert as cells, open as session");
}

// (slot-24 NB-TOURFIX) the hint kind factored: char_Z(T_2) at (1, 10), and one factor of multiplicity 1 alone.
const fac = (cs, m) => ({ t: "Record", v: [["coefficients", { t: "List", v: cs.map((c) => ({ t: "String", v: c })) }],
                                          ["multiplicity", { t: "Int", v: m }]] });
eq(P.hintTeX({ t: "List", v: [fac(["1", "24"], 2), fac(["1", "-2049"], 1)], hint: { kind: "factored", order: "high_to_low", var: "X", ring: "Z" } }),
   "(X + 24)^{2}\\,(X - 2049) \\in \\mathbb{Z}[X]", "hint factored");
eq(P.factoredTeX({ t: "List", v: [fac(["1", "0", "0", "0", "1"], 1)] }, { order: "high_to_low", var: "X" }), "X^{4} + 1", "factoredTeX one factor");
eq(P.factoredTeX({ t: "List", v: [] }, { order: "high_to_low" }), null, "factoredTeX refuses an empty product");

const ok = failures === 0;
console.log(`RENDER_TEST fixtures=${fixtures.length} tex=${tex} text=${text} hinted=${hinted} failures=${failures} verdict=${ok ? "OK" : "FAIL"}`);
process.exit(ok ? 0 : 1);
