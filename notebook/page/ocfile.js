// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// ocfile.js -- the notebook FILE format (slot-24 NB-FILE; [P] Sec. 18.18): a plain OC script whose cells are
// separated by "percent" lines, the convention of Jupytext and VS Code. Pure functions, no DOM, no session talk:
// parse(text), serialise(nb), canonical(text) and the two list operations insertCells / replaceAll.
//
//   <preamble: optional comment block, kept with the notebook, not a cell>
//   # %% [title]            a code cell (the title is optional; it is shown in the cell bar)
//   <code lines>
//   # %% [md] [title]       a text cell (not executed): every body line starts with '#'
//   # <text line>
//
// `#` is an OC comment, so the whole file is a valid script and `hacs --script nb.oc` runs every cell
// in order. Outputs are never stored (oc-notebook-export/1 is the way to keep them).
"use strict";

const OCFile = (() => {
  // A separator: "# %%" followed by end of line or white space (then the optional title).
  const SEP = /^# %%(?:[ \t]+(.*))?$/;
  const MD = /^\[md\](?:[ \t]+(.*))?$/;
  const isBlank = (l) => /^[ \t]*$/.test(l);

  const normalise = (text) => String(text).replace(/^﻿/, "").replace(/\r\n?/g, "\n");
  // Drop blank lines at both ends of a list of lines.
  function trimLines(lines) {
    let a = 0, b = lines.length;
    while (a < b && isBlank(lines[a])) ++a;
    while (b > a && isBlank(lines[b - 1])) --b;
    return lines.slice(a, b);
  }
  const cleanTitle = (t) => String(t || "").replace(/[\r\n]+/g, " ").trim();

  // parse(text) -> { preamble: string, cells: [{ kind: "code" | "md", title, code }] }. Throws Error (message
  // with the line number) for a text cell with a body line that does not start with '#'.
  //   * CRLF / CR -> LF and a leading BOM dropped; blank lines at the ends of every chunk dropped;
  //   * a file without a separator is ONE code cell (the whole text, comments included, no preamble);
  //   * text before the first separator is the preamble (kept, not a cell);
  //   * a text cell's `code` is its text: each body line without its '#' and one following space.
  function parse(text) {
    const lines = normalise(text).split("\n");
    const pre = [], cells = [];
    let cur = null;
    lines.forEach((line, i) => {
      const m = SEP.exec(line);
      if (m) {
        let title = cleanTitle(m[1]), kind = "code";
        const md = MD.exec(title);
        if (md) { kind = "md"; title = cleanTitle(md[1]); }
        cur = { kind, title, body: [], from: i + 2 };   // `from`: 1-based number of the first body line
        cells.push(cur);
      } else (cur ? cur.body : pre).push(line);
    });
    if (!cells.length) return { preamble: "", cells: [{ kind: "code", title: "", code: trimLines(pre).join("\n") }] };
    const out = [];
    for (const c of cells) {
      if (c.kind === "md") {
        const body = trimLines(c.body);
        let lead = 0;   // leading blank lines dropped (for the error's line number)
        while (lead < c.body.length && isBlank(c.body[lead])) ++lead;
        const text = body.map((l, k) => {
          if (isBlank(l)) return "";
          if (l[0] !== "#") throw new Error(`line ${c.from + lead + k}: a text cell ("# %% [md]") needs every body line to start with '#'`);
          return l.slice(l.startsWith("# ") ? 2 : 1);
        });
        out.push({ kind: "md", title: c.title, code: text.join("\n") });
      } else out.push({ kind: "code", title: c.title, code: trimLines(c.body).join("\n") });
    }
    return { preamble: trimLines(pre).join("\n"), cells: out };
  }

  // serialise({ preamble, cells }) -> the canonical file text: LF only, the preamble (if any) then one blank
  // line, every cell as its separator line and its body, ONE blank line between cells, exactly one final newline.
  // A canonical file is a fixed point of serialise(parse(x)). A title is one line; a code cell title that would
  // read as "[md]" loses that prefix. A code line that itself starts with "# %%" would start a new cell on reload.
  function serialise(nb) {
    const parts = [];
    const pre = trimLines(normalise(nb.preamble || "").split("\n")).join("\n");
    if (pre) parts.push(pre + "\n");
    for (const c of nb.cells || []) {
      let title = cleanTitle(c.title);
      let body = trimLines(normalise(c.code || "").split("\n"));
      if (c.kind === "md") body = body.map((l) => (l === "" ? "#" : "# " + l));
      else { const md = MD.exec(title); if (md) title = cleanTitle(md[1]); }
      const head = "# %%" + (c.kind === "md" ? " [md]" : "") + (title ? " " + title : "");
      parts.push([head, ...body].join("\n") + "\n");
    }
    return parts.join("\n");
  }

  // The canonical form of a file text (for "is this file canonical?").
  const canonical = (text) => serialise(parse(text));

  // insertCells(cells, index, parsed) -> a NEW array: the parsed notebook's cells inserted after position `index`
  // (-1: at the front). A non-empty preamble becomes a first, untitled code cell so that nothing is lost.
  function insertCells(cells, index, parsed) {
    const add = parsed.cells.slice();
    if (parsed.preamble) add.unshift({ kind: "code", title: "", code: parsed.preamble });
    const at = Math.min(Math.max(index + 1, 0), cells.length);
    return cells.slice(0, at).concat(add, cells.slice(at));
  }
  // replaceAll(parsed) -> the new notebook: { preamble, cells } (a new session: the caller restarts it).
  const replaceAll = (parsed) => ({ preamble: parsed.preamble, cells: parsed.cells.slice() });

  return { parse, serialise, canonical, insertCells, replaceAll, SEP };
})();

if (typeof module !== "undefined" && module.exports) module.exports = OCFile;
