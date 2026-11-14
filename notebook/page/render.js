// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// render.js -- oc-serve/1 value envelopes ([P] Sec. 18.5) -> DOM, math through KaTeX.
//
// The binary emits structured JSON and no TeX ([P] Sec. 18.1.2); this file is the only
// place where TeX is produced. TeX is built exclusively from validated integers and fixed
// macro text -- no string coming from the session is ever interpolated into TeX or into
// innerHTML (strings go into text nodes), so a value cannot inject markup. A type without a
// renderer here falls back to its `canonical` text, as the spec requires.
//
// (slot-24 NB-RENDER R2) Added: the render hints of [P] Sec. 18.5 (`hint`: poly, qexp,
// abgroup, hilbert, layers, partition), the member form D-m (members + calls), polynomials
// over Z / Z/ell^k / F_ell with signed coefficients, abelian groups, q-series, Jordan types,
// Loewy layers, Lists of Records as tables, paged tables for large matrices, and the
// page-level LaTeX switch. The switch changes PRESENTATION only: with LaTeX off every leaf is
// its canonical text, the tables stay. Every value block keeps its canonical text in the DOM
// (data-text on each rendered math node, the hidden canonical <pre> of each value), so the
// text behind a display can always be copied or exported (plainText()).
//
// (slot-24 NB-RENDER R3) Added: the hint kinds cyclic ((+)_i Z/ell^{e_i}), eigen (a character
// value in the residue field F_ell[w]/(pi_w)) and lesmap (a map of the Dickson long exact
// sequence, labelled by its source and target); the "compute" button on a declared handle and
// the "show all" button on a shape-only Matrix, both of which ask the page (app.js) to evaluate
// a `display(...)` cell -- render.js itself never talks to the session; stmtSource(), the
// statement text behind a value frame's line / col.
"use strict";

const OCRender = (() => {
  const INT_RE = /^-?[0-9]+$/;
  const isInt = (x) => (typeof x === "number" && Number.isInteger(x)) || (typeof x === "string" && INT_RE.test(x));
  const ti = (x) => { if (!isInt(x)) throw new Error("not an int"); return String(x); };
  const MAX_TEX_ENTRIES = 900;   // a larger matrix is shown as a paged table
  const PAGE = 50;               // the paged table's window, rows and columns
  const MAX_TEX_TERMS = 400;     // a longer polynomial / q-series is shown as text
  const MAX_SHOW_ALL = 100000;   // "show all" up to this many entries (the 1 MiB frame cap, [P] Sec. 18.4)

  // The actions of the value being rendered (set by renderValue for the duration of one
  // synchronous render): {compute(), showAll(n)} from app.js, or null (fixtures, tests).
  let curActions = null;

  // ---- the page-level LaTeX switch (default on; remembered per viewer, never required) ----
  let texOn = true;
  try { const s = localStorage.getItem("oc_tex"); if (s === "off") texOn = false; } catch (e) { /* storage blocked: default */ }
  function setTeX(on) {
    texOn = !!on;
    try { localStorage.setItem("oc_tex", texOn ? "on" : "off"); } catch (e) { /* not remembered */ }
  }
  const texEnabled = () => texOn;

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  const field = (p) => `\\mathbb{F}_{${ti(p)}}`;

  // ---- unwrapping envelopes: a List of Int envelopes -> plain values ----
  const plainOf = (e) => (e && typeof e === "object" && "t" in e && "v" in e) ? e.v : e;
  function intList(env) {
    if (!env || env.t !== "List" || !Array.isArray(env.v) || env.truncated) return null;
    const out = [];
    for (const x of env.v) { const y = plainOf(x); if (!isInt(y)) return null; out.push(String(y)); }
    return out;
  }
  // A Record envelope -> {key: envelope}
  function recOf(env) {
    if (!env || env.t !== "Record" || !Array.isArray(env.v)) return null;
    const o = {};
    for (const pr of env.v) if (Array.isArray(pr) && pr.length === 2) o[String(pr[0])] = pr[1];
    return o;
  }

  // ---- rings and variables (whitelisted: no hint text reaches TeX unchecked) ----
  function ringTeX(h) {
    if (!h) return null;
    if (h.ring === "Z") return "\\mathbb{Z}";
    if (h.ring === "F_ell") return isInt(h.ell) ? field(h.ell) : "\\mathbb{F}_{\\ell}";
    if (h.ring === "Z/ell^k") {
      const l = isInt(h.ell) ? ti(h.ell) : "\\ell", k = isInt(h.k) ? ti(h.k) : "k";
      return `\\mathbb{Z}/${l}^{${k}}`;
    }
    return null;
  }
  // The console's ring text ("F_ell", "Z/ell^3", "Z") at the prime p. Over Z/ell^k a handle refuses
  // its `p` / `ell` members by design ([E] sec:ring (h)); the prime is then read from the canonical
  // text's "Z/<int>^<int>" (integers only, validated), else written as \ell.
  function primeFromCanonical(canon) {
    const m = /Z\/([0-9]+)\^([0-9]+)/.exec(canon || "");
    return m ? m[1] : null;
  }
  function ringTextTeX(text, p) {
    if (text === "F_ell") return field(p);
    if (text === "Z") return "\\mathbb{Z}";
    const m = /^Z\/ell\^([0-9]+)$/.exec(text || "");
    return m ? `\\mathbb{Z}/${isInt(p) ? ti(p) : "\\ell"}^{${m[1]}}` : null;
  }
  function varTeX(h) {
    switch (h && h.var) {
      case "X": return "X"; case "Y": return "Y"; case "T": return "T"; case "q": return "q"; case "w": return "w";
      case "zeta": return isInt(h.ell) ? `\\zeta_{${ti(h.ell)}}` : "\\zeta";
      default: return "X";
    }
  }

  // A polynomial from its coefficients in the LABELLED order, printed HIGH -> LOW (the
  // conventional form; the order label of the envelope or hint is honoured, never assumed).
  // Signed integer coefficients are written with their sign (over Z).
  function polyTeX(coeffs, order, varName) {
    if (order !== "low_to_high" && order !== "high_to_low") return null;
    if (!Array.isArray(coeffs) || coeffs.length > MAX_TEX_TERMS) return null;
    const lo = order === "low_to_high" ? coeffs.slice() : coeffs.slice().reverse();
    let out = "";
    for (let k = lo.length - 1; k >= 0; --k) {
      let c = ti(lo[k]);
      if (/^-?0+$/.test(c)) continue;
      const neg = c[0] === "-";
      if (neg) c = c.slice(1);
      const mon = k === 0 ? "" : (k === 1 ? varName : `${varName}^{${k}}`);
      const term = k === 0 ? c : (c === "1" ? mon : `${c}${mon}`);
      out += out === "" ? (neg ? `-${term}` : term) : (neg ? ` - ${term}` : ` + ${term}`);
    }
    return out === "" ? "0" : out;
  }

  // a_0 + a_1 q + ... + O(q^N)
  function qexpTeX(coeffs) {
    if (!Array.isArray(coeffs) || coeffs.length > MAX_TEX_TERMS) return null;
    const terms = [];
    coeffs.forEach((c0, n) => {
      const c = ti(c0);
      if (c === "0") return;
      const mon = n === 0 ? "" : (n === 1 ? "q" : `q^{${n}}`);
      terms.push(n === 0 ? c : (c === "1" ? mon : `${c}${mon}`));
    });
    const N = coeffs.length;
    return `${terms.length ? terms.join(" + ") : "0"} + O(q^{${N}})`;
  }

  // {free_rank, torsion: [[p, k, mult], ...]} -> Z^r (+) (Z/p^k)^m ...
  function abgroupTeX(rec) {
    if (!rec) return null;
    const r = ti(plainOf(rec.free_rank));
    const tl = rec.torsion;
    if (!tl || tl.t !== "List" || !Array.isArray(tl.v)) return null;
    const parts = [];
    if (r !== "0") parts.push(r === "1" ? "\\mathbb{Z}" : `\\mathbb{Z}^{${r}}`);
    for (const e of tl.v) {
      const t = intList(e);
      if (!t || t.length !== 3) return null;
      const [p, k, m] = t;
      const cyc = k === "1" ? `\\mathbb{Z}/${p}` : `\\mathbb{Z}/${p}^{${k}}`;
      parts.push(m === "1" ? cyc : `(${cyc})^{${m}}`);
    }
    return parts.length ? parts.join(" \\oplus ") : "0";
  }

  // A Jordan type [3, 1, 1] -> (3, 1^{2})
  function partitionTeX(xs) {
    if (!xs) return null;
    if (!xs.length) return "\\varnothing";
    const parts = [];
    for (let i = 0; i < xs.length;) {
      let j = i; while (j < xs.length && xs[j] === xs[i]) ++j;
      parts.push(j - i === 1 ? xs[i] : `${xs[i]}^{${j - i}}`);
      i = j;
    }
    return `(${parts.join(",\\ ")})`;
  }

  function matrixTeX(v) {
    const r = v.rows, c = v.cols;
    if (!Number.isInteger(r) || !Number.isInteger(c)) return null;
    const where = `\\in M_{${r}\\times ${c}}(${field(v.p)})`;
    if (r === 0 || c === 0) return `0 ${where}`;
    if (!Array.isArray(v.entries) || r * c > MAX_TEX_ENTRIES) return null;
    const rows = v.entries.map((row) => row.map(ti).join(" & "));
    return `\\begin{pmatrix} ${rows.join(" \\\\ ")} \\end{pmatrix} ${where}`;
  }

  // H^q = (+)_i Z/ell^{e_i}, the exponents e_i -> (Z/ell^e)^m (+) ...
  function cyclicSumTeX(ell, exps) {
    if (!exps) return null;
    if (!exps.length) return "0";
    const counts = new Map();
    for (const e of exps) counts.set(e, (counts.get(e) || 0) + 1);
    const keys = [...counts.keys()].sort((a, b) => Number(b) - Number(a));
    return keys.map((e) => {
      const cyc = e === "1" ? `\\mathbb{Z}/${ti(ell)}` : `\\mathbb{Z}/${ti(ell)}^{${e}}`;
      const m = counts.get(e);
      return m === 1 ? cyc : `(${cyc})^{${m}}`;
    }).join(" \\oplus ");
  }

  // A map of the Dickson long exact sequence
  //   0 -> H^0(M_n (x) det) -> H^0(M_{n'}) -> H^0(Q_n) -> H^1(M_n (x) det) -> H^1(M_{n'}) -> H^1(Q_n) -> 0,
  // M_k = Sym^k, n' = n + ell + 1 ([E] sec:thetaobj): its name -> [symbol, source, target].
  const LES_MAPS = {
    theta0: ["\\theta^{0}", ["n", 0], ["up", 0]],
    pi0: ["\\pi^{0}", ["up", 0], ["Q", 0]],
    delta: ["\\delta", ["Q", 0], ["n", 1]],
    theta1: ["\\theta^{1}", ["n", 1], ["up", 1]],
    pi1: ["\\pi^{1}", ["up", 1], ["Q", 1]],
  };
  function lesTermTeX(term, n, nUp) {
    const [w, q] = term;
    const N = isInt(n) ? ti(n) : "n", U = isInt(nUp) ? ti(nUp) : "n'";
    const coeff = w === "n" ? `\\mathrm{Sym}^{${N}}\\otimes\\det` : (w === "up" ? `\\mathrm{Sym}^{${U}}` : `Q_{${N}}`);
    return `H^{${q}}(${coeff})`;
  }
  function lesmapTeX(env, h) {
    const m = Object.prototype.hasOwnProperty.call(LES_MAPS, h.map) ? LES_MAPS[h.map] : null;
    if (!m || env.t !== "Matrix" || !env.v) return null;
    const mat = matrixTeX(env.v);
    const head = `${m[0]}\\colon ${lesTermTeX(m[1], h.n, h.n_up)} \\to ${lesTermTeX(m[2], h.n, h.n_up)}`;
    return mat === null ? null : `${head},\\quad ${mat}`;
  }

  // A character value P_p(w) in the residue field k_m = F_ell[w]/(pi_w) = F_{ell^f} (a Poly envelope).
  function eigenTeX(env, h) {
    if (env.t !== "Poly" || !env.v) return null;
    const v = env.v;
    const p = polyTeX(v.coeffs, v.order, "w");
    if (p === null) return null;
    const f = isInt(h.f) ? ti(h.f) : null;
    if (f === "1") return `${p} \\in ${field(v.ell)}`;
    const big = f === null ? "\\mathbb{F}_{\\ell^{f}}" : `\\mathbb{F}_{${ti(v.ell)}^{${f}}}`;
    return `${p} \\in ${field(v.ell)}[w]/(\\pi_w) \\cong ${big}`;
  }

  // (slot-24 NB-TOURFIX) a product of polynomials with multiplicities: a List of {coefficients, multiplicity[, ...]}
  // Records -> "(X + 24)^{2} (X - 2049)" (one factor of multiplicity 1 without parentheses); null if anything is off.
  function factoredTeX(env, h) {
    if (!env || env.t !== "List" || !Array.isArray(env.v) || env.truncated || !env.v.length || env.v.length > 64) return null;
    const x = varTeX(h), parts = [];
    for (const e of env.v) {
      const r = recOf(e);
      const cs = r && intList(r.coefficients), m = r && plainOf(r.multiplicity);
      if (!cs || !isInt(m)) return null;
      const p = polyTeX(cs, h.order, x);
      if (p === null) return null;
      parts.push([p, ti(m)]);
    }
    if (parts.length === 1 && parts[0][1] === "1") return parts[0][0];
    return parts.map(([p, m]) => `(${p})${m === "1" ? "" : `^{${m}}`}`).join("\\,");
  }

  // TeX by a render hint ([P] Sec. 18.5), or null.
  function hintTeX(env) {
    const h = env.hint;
    if (!h || typeof h !== "object") return null;
    switch (h.kind) {
      case "poly": {
        const cs = intList(env);
        if (!cs) return null;
        const R = ringTeX(h), x = varTeX(h);
        const p = polyTeX(cs, h.order, x);
        if (p === null) return null;
        if (h.var === "zeta") return R === "\\mathbb{Z}" && isInt(h.ell) ? `${p} \\in \\mathbb{Z}[\\zeta_{${ti(h.ell)}}]` : p;
        return R ? `${p} \\in ${R}[${x}]` : p;
      }
      case "qexp": {
        const cs = intList(env);
        if (!cs) return null;
        const q = qexpTeX(cs), R = ringTeX(h);
        return q === null ? null : (R ? `${q} \\in ${R}[[q]]` : q);
      }
      case "abgroup": return abgroupTeX(recOf(env));
      case "hilbert": { const xs = intList(env); return xs ? `(${xs.join(",\\ ")})` : null; }
      case "layers": { const xs = intList(env); return xs ? (xs.length ? xs.join(" \\mid ") : "0") : null; }
      case "partition": return partitionTeX(intList(env));
      case "cyclic": {
        const xs = intList(env);
        return xs && isInt(h.ell) ? cyclicSumTeX(h.ell, xs) : null;
      }
      case "eigen": return eigenTeX(env, h);
      case "lesmap": return lesmapTeX(env, h);
      case "factored": {
        const f = factoredTeX(env, h), R = ringTeX(h);
        return f === null ? null : (R ? `${f} \\in ${R}[${varTeX(h)}]` : f);
      }
      default: return null;
    }
  }

  // TeX for an envelope, or null when the type has no math rendering (or LaTeX is off).
  function toTeX(env, depth) {
    if (!texOn) return null;
    return toTeXRaw(env, depth);
  }
  function toTeXRaw(env, depth) {
    if (!env || typeof env !== "object" || env.truncated) return null;
    const v = env.v;
    try {
      if (env.hint) { const t = hintTeX(env); if (t !== null) return t; }
      switch (env.t) {
        case "Int": return ti(v);
        case "Fp": return `${ti(v.value)} \\in ${field(v.p)}`;
        case "Poly": {
          const p = polyTeX(v.coeffs, v.order, "X");
          return p === null ? null : `${p} \\in ${field(v.ell)}[X]`;
        }
        case "Matrix": return matrixTeX(v);
        case "GF": {
          const p = polyTeX(v.polynomial.coeffs, v.polynomial.order, "X");
          return p === null ? null : `\\mathbb{F}_{${ti(v.ell)}^{${ti(v.r)}}} = ${field(v.ell)}[X]/(${p})`;
        }
        case "GFElt": {
          const p = polyTeX(v.coeffs, v.order, "Y");
          return p === null ? null : `${p} \\in \\mathbb{F}_{${ti(v.ell)}^{${ti(v.r)}}}`;
        }
        case "LocalSystem": {
          const R = ringTextTeX(v.coefficient_ring, isInt(v.p) ? v.p : primeFromCanonical(env.canonical));
          if (R === null) return null;
          const base = v.coefficient_ring === "F_ell" ? R : `(${R})`;
          return `\\mathrm{Sym}^{${ti(v.n)}}\\big(${base}^{2}\\big),\\quad d_V = ${ti(v.d_V)}`;
        }
        case "Cohomology": {
          if (env.state !== "computed") return null;
          const degs = Array.isArray(v.degrees) ? v.degrees : (Array.isArray(v.dims) ? v.dims.map((_, i) => i) : []);
          if (v.coefficient_ring === "F_ell" && Array.isArray(v.dims))
            return v.dims.map((d, i) => `\\dim_{${field(v.ell)}} H^{${ti(degs[i])}} = ${ti(d)}`).join(",\\quad ");
          const ell = isInt(v.ell) ? v.ell : primeFromCanonical(env.canonical);
          if (Array.isArray(v.invariant_factors) && isInt(ell) && /^Z\/ell\^[0-9]+$/.test(v.coefficient_ring || ""))
            return v.invariant_factors.map((ex, i) => `H^{${ti(degs[i])}} \\cong ${cyclicSumTeX(ell, ex.map(ti))}`).join(",\\quad ");
          if (Array.isArray(v.lengths))
            return v.lengths.map((d, i) => `\\ell\\big(H^{${ti(degs[i])}}\\big) = ${ti(d)}`).join(",\\quad ");
          return null;
        }
        case "DescentRecord": {
          const m = v && v.members;
          if (!m || !m.groups_inv || !Array.isArray(m.groups_inv.v) || !m.groups_inv.v.length) return null;
          const N = ti(plainOf(m.N)), n = ti(plainOf(m.n));
          const parts = m.groups_inv.v.map((g, q) => `H^{${q}} \\cong ${abgroupTeX(recOf(g))}`);
          return `H^q\\big(\\Gamma_0(${N}),\\, \\mathrm{Sym}^{${n}}\\mathbb{Z}^2\\big):\\quad ${parts.join(",\\quad ")}`;
        }
        case "Record": {
          // {free_rank, torsion} read on its own (e.g. D.groups_inv[1]): the same group as the hinted path
          if (Array.isArray(v) && v.length === 2 && v[0][0] === "free_rank" && v[1][0] === "torsion") return abgroupTeX(recOf(env));
          return null;
        }
        case "List": {
          if (!Array.isArray(v) || v.length > 64 || (depth || 0) > 2) return null;
          const parts = [];
          for (const e of v) {
            const t = toTeXRaw(e, (depth || 0) + 1);
            if (t === null || /\\in|\\quad|=|\\cong/.test(t)) return null;   // only plain members make a TeX list
            parts.push(t);
          }
          return `\\left[\\,${parts.join(",\\ ")}\\,\\right]`;
        }
        default: return null;
      }
    } catch (e) {
      return null;
    }
  }

  function katexInto(node, tex, display) {
    if (typeof katex === "undefined") return false;
    try {
      katex.render(tex, node, { throwOnError: false, displayMode: !!display, trust: false, strict: "ignore", maxExpand: 1000 });
      return true;
    } catch (e) {
      return false;
    }
  }

  function jsonDetails(label, obj) {
    const d = el("details", "json");
    d.appendChild(el("summary", "muted", label));
    d.appendChild(el("pre", "", JSON.stringify(obj, null, 2)));
    return d;
  }

  // A key/value table (Record pairs, members, calls).
  function kvTable(pairs, depth) {
    const tbl = el("table", "rec");
    for (const [k, e] of pairs) {
      const tr = el("tr");
      tr.appendChild(el("td", "k", k));
      const td = el("td");
      td.appendChild(body(e, depth + 1));
      tr.appendChild(td);
      tbl.appendChild(tr);
    }
    return tbl;
  }

  // A List of Records -> one table, columns = the union of the keys in first-seen order.
  function recordListTable(items, depth) {
    const recs = items.map((e) => e.v);
    const cols = [];
    for (const r of recs) for (const pr of r) if (!cols.includes(String(pr[0]))) cols.push(String(pr[0]));
    const tbl = el("table", "rec grid");
    const hr = el("tr");
    for (const c of cols) hr.appendChild(el("th", "k", c));
    tbl.appendChild(hr);
    for (const r of recs) {
      const tr = el("tr");
      const o = {};
      for (const pr of r) o[String(pr[0])] = pr[1];
      for (const c of cols) {
        const td = el("td");
        if (c in o) td.appendChild(body(o[c], depth + 1));
        tr.appendChild(td);
      }
      tbl.appendChild(tr);
    }
    return tbl;
  }

  // A large Matrix with entries: a paged table of text nodes (PAGE x PAGE), sticky indices.
  function pagedMatrix(v) {
    const wrap = el("div", "paged");
    const info = el("div", "muted small");
    const nav = el("div", "paged-nav");
    const grid = el("div", "paged-grid");
    let r0 = 0, c0 = 0;
    const btn = (label, dr, dc) => {
      const b = el("button", "", label); b.type = "button";
      b.addEventListener("click", () => {
        r0 = Math.min(Math.max(0, r0 + dr), Math.max(0, v.rows - 1));
        c0 = Math.min(Math.max(0, c0 + dc), Math.max(0, v.cols - 1));
        draw();
      });
      return b;
    };
    nav.append(btn("↑", -PAGE, 0), btn("↓", PAGE, 0), btn("←", 0, -PAGE), btn("→", 0, PAGE));
    function draw() {
      grid.textContent = "";
      const tbl = el("table", "mat");
      const hr = el("tr"); hr.appendChild(el("th", "idx", ""));
      const r1 = Math.min(v.rows, r0 + PAGE), c1 = Math.min(v.cols, c0 + PAGE);
      for (let j = c0; j < c1; ++j) hr.appendChild(el("th", "idx", String(j)));
      tbl.appendChild(hr);
      for (let i = r0; i < r1; ++i) {
        const tr = el("tr"); tr.appendChild(el("th", "idx", String(i)));
        for (let j = c0; j < c1; ++j) tr.appendChild(el("td", "", String(v.entries[i][j])));
        tbl.appendChild(tr);
      }
      grid.appendChild(tbl);
      info.textContent = `${v.rows} × ${v.cols} over F_${v.p}; rows ${r0}..${r1 - 1}, columns ${c0}..${c1 - 1}`;
    }
    draw();
    wrap.append(info, nav, grid);
    return wrap;
  }

  // A rendered math node that keeps the text behind it.
  function mathNode(tex, text, display) {
    const m = el("div", "math");
    m.dataset.text = text;
    m.title = text.length <= 400 ? text : text.slice(0, 400) + "…";
    if (katexInto(m, tex, display)) return m;
    return null;
  }

  // The member form D-m (and the generic form): members table, calls table.
  function memberBody(env, depth, wrap) {
    const v = env.v;
    const pairs = Object.keys(v.members || {}).map((k) => [k, v.members[k]]);
    if (pairs.length) wrap.appendChild(kvTable(pairs, depth));
    if (Array.isArray(v.calls) && v.calls.length) {
      const cp = [];
      for (const c of v.calls) {
        if (!Array.isArray(c) || c.length !== 3) continue;
        const args = Array.isArray(c[1]) ? c[1].map((a) => (isInt(a) ? String(a) : "?")).join(", ") : "";
        cp.push([`.${c[0]}(${args})`, c[2]]);
      }
      if (cp.length) wrap.appendChild(kvTable(cp, depth));
    }
  }

  // The body of an envelope (no header).
  function body(env, depth) {
    if (!env || typeof env !== "object") return el("span", "muted", "?");
    const tex = toTeX(env, 0);
    const v = env.v;
    const memberForm = v && typeof v === "object" && !Array.isArray(v) && v.members && (env.generic || Array.isArray(v.calls));
    if (tex !== null) {
      const m = mathNode(tex, env.canonical, env.t === "Matrix");
      if (m && !memberForm) return m;
      if (m && memberForm) {
        const wrap = el("div");
        wrap.appendChild(m);
        if (depth < 3) memberBody(env, depth, wrap);
        return wrap;
      }
    }
    if (env.t === "Record" && Array.isArray(v) && !env.truncated && depth < 5)
      return kvTable(v.map((pr) => [String(pr[0]), pr[1]]), depth);
    if (env.t === "List" && Array.isArray(v) && !env.truncated && depth < 5 && v.length > 0 && v.length <= 200 &&
        v.every((e) => e && e.t === "Record" && Array.isArray(e.v)))
      return recordListTable(v, depth);
    if (env.t === "List" && Array.isArray(v) && !env.truncated && depth < 5 && v.length <= 64 &&
        v.some((e) => toTeX(e, 0) !== null || (e && (e.t === "Record" || e.t === "List")))) {
      const ol = el("ol", "", undefined);
      ol.start = 0;
      for (const e of v) {
        const li = el("li");
        li.appendChild(body(e, depth + 1));
        ol.appendChild(li);
      }
      return ol;
    }
    if (env.t === "Matrix" && v && Array.isArray(v.entries) && v.rows * v.cols > MAX_TEX_ENTRIES) return pagedMatrix(v);
    if (env.t === "String" && texOn) {
      const dec = certifiedDecimal(v);
      if (dec !== null) {
        const d = el("div", "cert-dec", `certified: ${dec}`);
        d.title = `stored as ${v} (signed hexadecimal, ${dec} in decimal)`;
        d.dataset.text = env.canonical;
        return d;
      }
    }
    const wrap = el("div");
    wrap.appendChild(el("pre", "canon", env.canonical));
    if (env.t === "Matrix" && env.truncated && v && Number.isInteger(v.rows)) {
      const n = v.rows * v.cols;
      const note = el("div", "muted small", `shape only (${v.rows} × ${v.cols} = ${n} entries, over print_max_entries)`);
      const act = curActions;
      if (act && typeof act.showAll === "function" && n <= MAX_SHOW_ALL) {
        const b = el("button", "show-all", "show all");
        b.type = "button";
        b.title = `re-run this statement as display(...) with print_max_entries = ${n} (then restore it)`;
        b.addEventListener("click", () => { b.disabled = true; act.showAll(n); });
        note.appendChild(document.createTextNode(" "));
        note.appendChild(b);
      } else if (n > MAX_SHOW_ALL) {
        note.appendChild(document.createTextNode(`; too large for one frame, use submatrix(...)`));
      }
      wrap.appendChild(note);
    }
    if (memberForm && depth < 3) {
      memberBody(env, depth, wrap);
    } else if (v && typeof v === "object" && depth < 2 &&
               ["Block", "Decomposition", "Report", "ExtRecord", "Settings", "Model", "Complex"].includes(env.t)) {
      wrap.appendChild(jsonDetails("json", v));
    }
    return wrap;
  }

  // The plain text behind a value: exactly the canonical text the binary sent.
  const plainText = (env) => (env && typeof env.canonical === "string" ? env.canonical : "");

  function copyText(text, button) {
    const done = () => { if (button) { const t = button.textContent; button.textContent = "copied"; setTimeout(() => { button.textContent = t; }, 1200); } };
    try {
      if (navigator.clipboard && window.isSecureContext) { navigator.clipboard.writeText(text).then(done, () => showCopyBox(text)); return; }
    } catch (e) { /* fall through */ }
    showCopyBox(text);
  }
  // Outside a secure context (plain http on the LAN) there is no clipboard API: a selected text box.
  function showCopyBox(text) {
    const box = el("div", "copy-box");
    const ta = el("textarea", "copy-ta");
    ta.value = text; ta.readOnly = true; ta.rows = Math.min(20, Math.max(3, text.split("\n").length));
    const close = el("button", "", "close"); close.type = "button";
    close.addEventListener("click", () => box.remove());
    box.append(el("div", "muted small", "Select and copy (Ctrl+C):"), ta, close);
    document.body.appendChild(box);
    ta.focus(); ta.select();
  }

  // The source text of the statement that starts at (line, col) (1-based) of a cell: up to the
  // first `;`, newline or `#` outside brackets and strings. null when nothing is there.
  function stmtSource(src, line, col) {
    if (typeof src !== "string" || !Number.isInteger(line) || !Number.isInteger(col) || line < 1 || col < 1) return null;
    const lines = src.split("\n");
    if (line > lines.length) return null;
    let off = 0;
    for (let i = 0; i < line - 1; ++i) off += lines[i].length + 1;
    off += col - 1;
    let depth = 0, inStr = false, out = "";
    for (let i = off; i < src.length; ++i) {
      const c = src[i];
      if (inStr) {
        out += c;
        if (c === "\\" && i + 1 < src.length) { out += src[++i]; continue; }
        if (c === '"') inStr = false;
        continue;
      }
      if (c === '"') { inStr = true; out += c; continue; }
      if (depth === 0 && (c === ";" || c === "#" || c === "\n")) break;
      if (c === "(" || c === "[" || c === "{") ++depth;
      if (c === ")" || c === "]" || c === "}") { if (depth === 0) break; --depth; }
      out += c;
    }
    out = out.trim();
    return out === "" || depth !== 0 || inStr ? null : out;
  }

  // A full value block: header (type, state, hash, toggles) + body. `actions` (optional, from
  // app.js): {compute(), showAll(n)} -- the "compute" button of a declared handle and the
  // "show all" button of a shape-only Matrix.
  function renderValue(env, actions) {
    const box = el("div", "value");
    box._env = env;
    box._actions = actions || null;
    box.dataset.text = plainText(env);
    const head = el("div", "vhead");
    let h = env.t;
    if (env.state && env.state !== "computed") h += ` (${env.state})`;
    if (env.truncated) h += " [truncated]";
    if (env.hash) h += `  hash ${env.hash}`;
    head.appendChild(document.createTextNode(h));
    const bText = el("button", "", "text");
    bText.title = "show the canonical text behind this display";
    const bCopy = el("button", "", "copy");
    bCopy.title = "copy the canonical text";
    const bJson = el("button", "", "json");
    const bTeX = copyTeX(env) !== null ? el("button", "", "LaTeX") : null;
    if (bTeX) bTeX.title = "copy this value as LaTeX source (the TeX the page renders)";
    for (const b of [bText, bCopy, bTeX, bJson].filter(Boolean)) { b.type = "button"; head.appendChild(b); }
    if (env.state === "declared" && actions && typeof actions.compute === "function") {
      const bComp = el("button", "compute", "compute");
      bComp.type = "button";
      bComp.title = "force this handle: runs display(<this statement>) in the session";
      bComp.addEventListener("click", () => { bComp.disabled = true; actions.compute(); });
      head.insertBefore(bComp, bText);
    }
    box.appendChild(head);
    curActions = actions || null;
    let main;
    try { main = body(env, 0); } finally { curActions = null; }
    const canon = el("pre", "canon", env.canonical);
    canon.hidden = true;
    const raw = el("pre", "canon", JSON.stringify(env, null, 2));
    raw.hidden = true;
    box.appendChild(main);
    box.appendChild(canon);
    box.appendChild(raw);
    bText.addEventListener("click", () => { canon.hidden = !canon.hidden; });
    bCopy.addEventListener("click", () => copyText(plainText(env), bCopy));
    if (bTeX) bTeX.addEventListener("click", () => copyText(copyTeX(env), bTeX));
    bJson.addEventListener("click", () => { raw.hidden = !raw.hidden; });
    return box;
  }

  // (slot-27 NB-A' C1) "Copy as LaTeX": the TeX the page builds for a value (toTeXRaw / the hint path), whether or not the
  // LaTeX switch is on; null when the type has no math rendering. Computed in the page from the envelope: no TeX from the binary.
  function copyTeX(env) { return toTeXRaw(env, 0); }
  // The TeX of one cell's output: every value block that has TeX, each as a display formula, blank-line separated; "" if none.
  function outputTeX(outNode) {
    const parts = [];
    for (const n of Array.from(outNode.children)) {
      if (!n.classList.contains("value") || !n._env) continue;
      const t = copyTeX(n._env);
      if (t !== null) parts.push("\\[\n" + t + "\n\\]");
    }
    return parts.join("\n\n");
  }

  // Re-render every value block under `root` (the LaTeX switch); the envelopes are kept.
  function rerenderAll(root) {
    for (const box of Array.from((root || document).querySelectorAll("div.value"))) {
      if (box._env) box.replaceWith(renderValue(box._env, box._actions));
    }
  }

  // The plain text of one cell's output, in order: stdout / log text as printed, each value as
  // its canonical text, progress and status lines as shown.
  function outputText(outNode) {
    const lines = [];
    for (const n of Array.from(outNode.children)) {
      if (n.classList.contains("value")) lines.push((n.dataset.text || "") + "\n");
      else lines.push(n.textContent.endsWith("\n") ? n.textContent : n.textContent + "\n");
    }
    return lines.join("");
  }

  // ---- the atlas panel (slot-20 NB-4, [P] v3.11 Sec. 18.16 (g)) ----
  // Every TeX string below is built from integers checked by ti() and fixed macro text; the
  // store's strings (type, group, reason) only ever reach text nodes.

  // An element of F_{ell^r} = F_ell[Y]/(C_{ell,r}): coefficients LOW -> HIGH (the store's order).
  function atlasEltTeX(coeffs) {
    if (!Array.isArray(coeffs) || coeffs.length === 0) return null;
    try { return polyTeX(coeffs, "low_to_high", "Y"); } catch (e) { return null; }
  }

  // F_{ell^r} = F_ell[Y]/(C) from header.fields, or F_ell for r = 1.
  function atlasFieldTeX(ell, f) {
    try {
      if (ti(f.r) === "1") return `\\mathbb{F}_{${ti(ell)}}`;
      const c = polyTeX(f.C, "low_to_high", "Y");
      return `\\mathbb{F}_{${ti(ell)}^{${ti(f.r)}}} = ${field(ell)}[Y]/(${c})`;
    } catch (e) { return null; }
  }

  // chi_{a,b}(q) = q^a + q^b, and the twist theta^j (x) #base.
  function atlasChiTeX(chi) {
    try { return Array.isArray(chi) && chi.length === 2 ? `\\chi_{${ti(chi[0])},${ti(chi[1])}}` : null; } catch (e) { return null; }
  }
  function atlasTwistTeX(twist, base) {
    try { return ti(twist) === "0" ? `\\#${ti(base)}` : `\\theta^{${ti(twist)}}\\otimes\\#${ti(base)}`; } catch (e) { return null; }
  }

  // A small math span (falls back to the TeX source as text when KaTeX is missing or LaTeX is off).
  function mathSpan(tex) {
    const s = el("span", "math-inline");
    if (tex === null || !texOn || !katexInto(s, tex, false)) s.textContent = tex === null ? "?" : tex;
    return s;
  }

  // ---- (slot-26 NBMATH A1) text cells: paragraphs, $...$ / $$...$$ through KaTeX, a minimal markdown ----
  // The text of a text cell is the user's own (or comes from the read-only examples directory). It is parsed into
  // blocks here and built with text nodes and KaTeX only -- never innerHTML -- and KaTeX runs with trust: false,
  // so no text can inject markup or links. This is not a value renderer: no toTeXRaw / hintTeX case, so the render
  // coverage gate ([P] Sec. 18.5, render_coverage.tsv) is unaffected. Syntax:
  //   a blank line ends a block; consecutive lines form a paragraph;
  //   $...$ inline math, $$...$$ display math (may span lines), \$ a literal dollar sign;
  //   "- " list items; "> " a statement box; "| a | b |" table rows (a row may continue over lines that do not start
  //   with "|", until a line ends with "|" and the row has the first row's number of cells; a row of dashes after the
  //   first row makes it the header); **bold**, `code`.
  // With LaTeX off the cell shows its source verbatim (the plain text stays extractable; Copy / Export use the source).

  // Inline segments of one paragraph: [{t: "text" | "math" | "dmath" | "code", s} | {t: "bold", c: [segments]}].
  function mdInline(s) {
    const out = [];
    let buf = "";
    const flush = () => { if (buf) { out.push({ t: "text", s: buf }); buf = ""; } };
    // The index of the next unescaped `delim` at or after `from` (-1 if none).
    const close = (delim, from) => {
      for (let j = from; j < s.length; ++j) {
        if (s[j] === "\\") { ++j; continue; }
        if (s.startsWith(delim, j)) return j;
      }
      return -1;
    };
    let i = 0;
    while (i < s.length) {
      const c = s[i];
      if (c === "\\" && s[i + 1] === "$") { buf += "$"; i += 2; continue; }
      if (s.startsWith("$$", i)) {
        const j = close("$$", i + 2);
        if (j > i + 2) { flush(); out.push({ t: "dmath", s: s.slice(i + 2, j).trim() }); i = j + 2; continue; }
        buf += "$$"; i += 2; continue;
      }
      if (c === "$") {
        const j = close("$", i + 1);
        if (j > i + 1) { flush(); out.push({ t: "math", s: s.slice(i + 1, j) }); i = j + 1; continue; }
        buf += c; ++i; continue;
      }
      if (c === "`") {
        const j = s.indexOf("`", i + 1);
        if (j > i + 1) { flush(); out.push({ t: "code", s: s.slice(i + 1, j) }); i = j + 1; continue; }
        buf += c; ++i; continue;
      }
      if (s.startsWith("**", i)) {
        const j = s.indexOf("**", i + 2);
        if (j > i + 2) { flush(); out.push({ t: "bold", c: mdInline(s.slice(i + 2, j)) }); i = j + 2; continue; }
      }
      buf += c; ++i;
    }
    flush();
    return out;
  }

  // Table cells of one row "| a | b |" (a "|" inside $...$ or `...` does not split).
  function mdRowCells(row) {
    const s = row.trim().replace(/^\|/, "").replace(/\|$/, "");
    const cells = [];
    let cur = "", math = false, code = false;
    for (let i = 0; i < s.length; ++i) {
      const c = s[i];
      if (c === "\\" && i + 1 < s.length) { cur += c + s[++i]; continue; }
      if (c === "$" && !code) math = !math;
      if (c === "`" && !math) code = !code;
      if (c === "|" && !math && !code) { cells.push(cur.trim()); cur = ""; continue; }
      cur += c;
    }
    cells.push(cur.trim());
    return cells;
  }

  // Blocks: [{t: "p", c: inline} | {t: "display", s} | {t: "ul", items: [inline]} | {t: "quote", blocks}
  //          | {t: "table", head: [inline] | null, rows: [[inline]]}].
  function mdBlocks(src) {
    const lines = String(src || "").replace(/\r\n?/g, "\n").split("\n");
    const blocks = [];
    const blank = (l) => /^[ \t]*$/.test(l);
    const starts = (l) => /^[ \t]*(\$\$|- |\||>)/.test(l);
    let i = 0;
    while (i < lines.length) {
      const l = lines[i];
      if (blank(l)) { ++i; continue; }
      const t = l.trim();
      if (t.startsWith("$$")) {                       // display math, possibly over several lines
        const acc = [t.slice(2)];
        let j = i, closed = acc[0].endsWith("$$") && acc[0].length >= 2;
        while (!closed && ++j < lines.length) { acc.push(lines[j]); if (lines[j].trim().endsWith("$$")) closed = true; }
        if (closed) {
          const body = acc.join("\n").trim();
          blocks.push({ t: "display", s: body.slice(0, -2).trim() });
          i = j + 1; continue;
        }
        // an unclosed $$: an ordinary paragraph (shown as its source)
      }
      if (t.startsWith(">")) {                        // a statement box: its lines without "> ", parsed as blocks
        const acc = [];
        while (i < lines.length && lines[i].trim().startsWith(">")) { acc.push(lines[i].trim().replace(/^> ?/, "")); ++i; }
        blocks.push({ t: "quote", blocks: mdBlocks(acc.join("\n")) });
        continue;
      }
      if (t.startsWith("|")) {                        // a table
        const rows = [];
        let head = null, ncols = 0;
        while (i < lines.length && lines[i].trim().startsWith("|")) {
          // a row ends at a line ending in "|" once it has as many cells as the first row (continuation lines do not start with "|")
          let row = lines[i].trim();
          const more = () => !row.endsWith("|") || (ncols > 0 && mdRowCells(row).length < ncols);
          while (more() && i + 1 < lines.length && !blank(lines[i + 1]) && !lines[i + 1].trim().startsWith("|")) row += " " + lines[++i].trim();
          ++i;
          const cells = mdRowCells(row);
          if (!ncols) ncols = cells.length;
          if (cells.every((c) => /^:?-{3,}:?$/.test(c))) { if (rows.length === 1 && head === null) head = rows.pop(); continue; }
          rows.push(cells.map(mdInline));
        }
        blocks.push({ t: "table", head, rows });
        continue;
      }
      if (t.startsWith("- ")) {                       // a list; a line that starts no block continues the last item
        const items = [];
        while (i < lines.length && !blank(lines[i])) {
          const u = lines[i].trim();
          if (u.startsWith("- ")) items.push(u.slice(2));
          else if (starts(lines[i])) break;
          else items[items.length - 1] += "\n" + u;
          ++i;
        }
        blocks.push({ t: "ul", items: items.map(mdInline) });
        continue;
      }
      const acc = [l];                                // a paragraph
      ++i;
      while (i < lines.length && !blank(lines[i]) && !starts(lines[i])) acc.push(lines[i++]);
      blocks.push({ t: "p", c: mdInline(acc.join("\n")) });
    }
    return blocks;
  }

  // Every TeX string of a text, in order: [{tex, display}] (for the KaTeX check of the example files).
  function mdMath(src) {
    const out = [];
    const walkI = (segs) => { for (const g of segs) { if (g.t === "math") out.push({ tex: g.s, display: false });
      else if (g.t === "dmath") out.push({ tex: g.s, display: true }); else if (g.t === "bold") walkI(g.c); } };
    const walkB = (bs) => { for (const b of bs) {
      if (b.t === "display") out.push({ tex: b.s, display: true });
      else if (b.t === "p") walkI(b.c);
      else if (b.t === "ul") b.items.forEach(walkI);
      else if (b.t === "quote") walkB(b.blocks);
      else if (b.t === "table") { if (b.head) b.head.forEach(walkI); b.rows.forEach((r) => r.forEach(walkI)); } } };
    walkB(mdBlocks(src));
    return out;
  }

  function mdMathNode(tex, display) {
    const n = el(display ? "div" : "span", display ? "md-display" : "md-math");
    const src = display ? `$$${tex}$$` : `$${tex}$`;
    n.dataset.text = src;
    n.title = tex.length <= 400 ? tex : tex.slice(0, 400) + "…";
    if (!katexInto(n, tex, display)) n.textContent = src;
    return n;
  }
  function mdInlineInto(node, segs) {
    for (const g of segs) {
      if (g.t === "text") node.appendChild(document.createTextNode(g.s));
      else if (g.t === "math") node.appendChild(mdMathNode(g.s, false));
      else if (g.t === "dmath") node.appendChild(mdMathNode(g.s, true));
      else if (g.t === "code") node.appendChild(el("code", "", g.s));
      else if (g.t === "bold") { const b = el("strong"); mdInlineInto(b, g.c); node.appendChild(b); }
    }
    return node;
  }
  function mdBlocksInto(node, blocks) {
    for (const b of blocks) {
      if (b.t === "p") node.appendChild(mdInlineInto(el("div", "md-p"), b.c));
      else if (b.t === "display") node.appendChild(mdMathNode(b.s, true));
      else if (b.t === "quote") node.appendChild(mdBlocksInto(el("div", "md-quote"), b.blocks));
      else if (b.t === "ul") { const ul = el("ul", "md-ul"); for (const it of b.items) ul.appendChild(mdInlineInto(el("li"), it)); node.appendChild(ul); }
      else if (b.t === "table") {
        const tbl = el("table", "md-table");
        if (b.head) { const tr = el("tr"); for (const c of b.head) tr.appendChild(mdInlineInto(el("th"), c)); tbl.appendChild(tr); }
        for (const r of b.rows) { const tr = el("tr"); for (const c of r) tr.appendChild(mdInlineInto(el("td"), c)); tbl.appendChild(tr); }
        const wrap = el("div", "md-tablewrap");
        wrap.appendChild(tbl);
        node.appendChild(wrap);
      }
    }
    return node;
  }

  // Render the text `src` of a text cell into `node` (its children are replaced). LaTeX off, or no KaTeX: the source verbatim.
  function renderMarkdown(node, src) {
    node.textContent = "";
    node.dataset.text = src;
    const raw = !texOn || typeof katex === "undefined";
    node.classList[raw ? "add" : "remove"]("md-raw");
    if (raw) { node.textContent = src; return node; }
    return mdBlocksInto(node, mdBlocks(src));
  }

  // ---- (slot-26 NBMATH, owner question 8) a certified order stored in signed hexadecimal ----
  // The HeckeZ record stores a certified integer as "certified:0x2b3" (G0.conventions hex=canonical_signed). With LaTeX on,
  // a String value of exactly that form is shown in decimal ("certified: 691"), the stored form on hover and in the
  // canonical text (text / copy / export are unchanged). Pure presentation: nothing reaches TeX.
  const CERT_HEX = /^certified:(-?)0x([0-9a-f]+)$/;
  function certifiedDecimal(s) {
    const m = typeof s === "string" ? CERT_HEX.exec(s) : null;
    if (!m || m[2].length > 4096 || typeof BigInt === "undefined") return null;
    return m[1] + BigInt("0x" + m[2]).toString(10);
  }

  // (NBCONFIRM S4 fix round 1, audit M2) The page's re-send rule after a `done` frame `fr` of a cell. `asked` lists the
  // call sites ("line:col") at which this cell has already asked for a confirmation, `tokens` the tokens presented so far.
  // Returns {action: "finish"} (not a confirmation request), {action: "ask", site} (show the dialog, re-send with one more
  // token) or {action: "stop", site, message} (do not ask again). A re-send re-executes the cell's statements before the
  // call, with the bindings the stopped run left; a cell that rebinds what it reads (k = k + 2; r = descent(1, 60 + k))
  // changes the call's arguments, hence its token, on every round and would never converge. So a site asks AT MOST ONCE
  // per cell run (one retry): when the same site asks again after its confirmation, the page stops and explains.
  function confirmStep(asked, tokens, fr) {
    if (!fr || fr.status !== "confirm_required" || !fr.confirm || typeof fr.confirm.token !== "string") return { action: "finish" };
    const st = fr.confirm.site;
    const site = st && typeof st === "object" ? `${st.line}:${st.col}` : String(st);
    const where = st && typeof st === "object" ? `line ${st.line}, column ${st.col}` : "the same place";
    if (asked.includes(site))
      return { action: "stop", site, message:
        `not run: the call at ${where} asked for a confirmation again after it was confirmed. Its arguments (or the ` +
        `settings) changed between the two runs of the cell: the cell rebinds a variable it reads (e.g. k = k + 2), or ` +
        `the call sits in a loop over changing arguments. The re-send has already run the statements before the call a ` +
        `second time (with their side effects, earlier confirmed calls included); their bindings are kept. Make the cell ` +
        `idempotent: compute the arguments in an earlier cell and put one expensive call per cell.` };
    if (tokens.length >= 16) return { action: "stop", site, message: "not run: more than 16 confirmations in one cell; split the cell" };
    return { action: "ask", site };
  }

  return { renderValue, toTeX, confirmStep, toTeXRaw, el, katexInto, mathSpan, atlasEltTeX, atlasFieldTeX, atlasChiTeX, atlasTwistTeX, isInt,
           setTeX, texEnabled, rerenderAll, plainText, outputText, copyTeX, outputTeX, copyText, stmtSource, renderMarkdown,
           _pure: { mdInline, mdBlocks, mdMath, certifiedDecimal, polyTeX, qexpTeX, abgroupTeX, partitionTeX, cyclicSumTeX, hintTeX, recOf, intList, stmtSource, eigenTeX, lesmapTeX, factoredTeX, confirmStep } };
})();

if (typeof module !== "undefined" && module.exports) module.exports = OCRender;
