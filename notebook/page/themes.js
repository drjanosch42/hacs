// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// themes.js -- page themes (slot-22 WP-NBT, owner request FR-1): the colour presets, the theme
// switcher in the header and the theme editor. Client-side only: a theme is a set of CSS custom
// properties on <html>; nothing goes to the gateway, switching never reloads the page or the session.
//
// Theme JSON (`oc-notebook-theme/1`), the import / export format:
//   {"schema":"oc-notebook-theme/1","name":"…","base":"light|dark","tokens":{"bg":"#rrggbb",…}}
// Unknown keys are ignored, a missing token is taken from the base preset, a value that is not
// `#rrggbb` refuses the whole theme (values are validated before they reach setProperty).
// The choice is kept in localStorage["oc-notebook-theme"] of this browser; every access is wrapped,
// so the page works without storage (private window): the theme then lasts until the tab closes.
//
// Loaded WITHOUT `defer` from <head>, so the stored theme is applied before the first paint; the
// controls are built on DOMContentLoaded. The pure functions are exported for node unit tests.
"use strict";

const OCTheme = (() => {
  // The block between the markers is strict JSON: test_gateway.py reads it with json.loads and
  // checks every preset (all tokens, `#rrggbb`, the contrast floors of `pairs`) in Python.
  const DATA = /* OC-THEME-DATA-BEGIN */ {
    "schema": "oc-notebook-theme/1",
    "tokens": ["bg", "fg", "muted", "line", "panel", "accent", "ok", "err", "warn", "code", "katex", "sel", "focus"],
    "labels": {
      "bg": "page background", "fg": "text", "muted": "secondary text", "line": "rules and borders",
      "panel": "cells, header, inputs", "accent": "links, accents", "ok": "status: ok", "err": "errors",
      "warn": "warnings, running cell", "code": "code panes", "katex": "math (KaTeX)", "sel": "text selection",
      "focus": "focus ring"
    },
    "pairs": [
      ["fg", "bg", 4.5], ["fg", "panel", 4.5], ["fg", "code", 4.5], ["fg", "sel", 4.5],
      ["muted", "bg", 4.5], ["muted", "panel", 4.5], ["muted", "code", 4.5],
      ["accent", "bg", 4.5], ["accent", "panel", 4.5],
      ["ok", "bg", 4.5], ["ok", "panel", 4.5], ["err", "bg", 4.5], ["err", "panel", 4.5],
      ["warn", "bg", 4.5], ["warn", "panel", 4.5],
      ["katex", "bg", 4.5], ["katex", "panel", 4.5],
      ["focus", "bg", 3.0], ["focus", "panel", 3.0]
    ],
    "presets": {
      "light": {"name": "Light", "base": "light", "tokens": {
        "bg": "#fbfbfa", "fg": "#1d1d1f", "muted": "#6b6b70", "line": "#dcdcdf", "panel": "#ffffff",
        "accent": "#2f5fb3", "ok": "#2a7a3d", "err": "#b3261e", "warn": "#9a6400", "code": "#f3f3f1",
        "katex": "#1d1d1f", "sel": "#cddcf5", "focus": "#2f5fb3"}},
      "dark": {"name": "Dark", "base": "dark", "tokens": {
        "bg": "#17181a", "fg": "#e6e6e8", "muted": "#9a9aa2", "line": "#333539", "panel": "#1f2023",
        "accent": "#7ea6f0", "ok": "#6cc684", "err": "#f08a80", "warn": "#e5b45a", "code": "#26272b",
        "katex": "#e6e6e8", "sel": "#2f4a78", "focus": "#7ea6f0"}},
      "high-contrast-light": {"name": "High contrast (light)", "base": "light", "tokens": {
        "bg": "#ffffff", "fg": "#000000", "muted": "#3a3a3a", "line": "#000000", "panel": "#ffffff",
        "accent": "#0033aa", "ok": "#005a1e", "err": "#a30000", "warn": "#6b4000", "code": "#f0f0f0",
        "katex": "#000000", "sel": "#ffe066", "focus": "#0033aa"}},
      "high-contrast-dark": {"name": "High contrast (dark)", "base": "dark", "tokens": {
        "bg": "#000000", "fg": "#ffffff", "muted": "#d0d0d0", "line": "#ffffff", "panel": "#000000",
        "accent": "#7fb2ff", "ok": "#5cf07a", "err": "#ff8a80", "warn": "#ffd24d", "code": "#1a1a1a",
        "katex": "#ffffff", "sel": "#004a99", "focus": "#ffd24d"}}
    }
  } /* OC-THEME-DATA-END */;

  const SCHEMA = DATA.schema;
  const TOKENS = DATA.tokens;
  const STORE_KEY = "oc-notebook-theme";
  const HEX = /^#[0-9a-fA-F]{6}$/;
  let storageOk = true;
  const MODES = ["auto", "light", "dark", "high-contrast-light", "high-contrast-dark", "custom"];

  // ---------------------------------------------------------------- pure functions
  function relLuminance(hex) {
    const ch = (i) => {
      const c = parseInt(hex.slice(i, i + 2), 16) / 255;
      return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * ch(1) + 0.7152 * ch(3) + 0.0722 * ch(5);
  }

  // The WCAG 2.x contrast ratio, 1 … 21.
  function contrast(a, b) {
    const la = relLuminance(a), lb = relLuminance(b);
    return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
  }

  // [{fg, bg, ratio, floor, ok}] for every pair of DATA.pairs.
  function checkContrast(tokens) {
    return DATA.pairs.map(([f, b, floor]) => {
      const ratio = contrast(tokens[f], tokens[b]);
      return { fg: f, bg: b, ratio, floor, ok: ratio >= floor };
    });
  }

  function presetTheme(id) {
    const p = DATA.presets[id];
    return { schema: SCHEMA, name: p.name, base: p.base, tokens: Object.assign({}, p.tokens) };
  }

  // Parse a theme from JSON text or an object. Returns {theme, notes} or {error}.
  function parseTheme(input) {
    let obj = input;
    if (typeof input === "string") {
      try { obj = JSON.parse(input); } catch (e) { return { error: "not valid JSON: " + e.message }; }
    }
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) return { error: "a theme is a JSON object" };
    if (obj.schema !== SCHEMA) return { error: `"schema" must be "${SCHEMA}"` };
    const base = obj.base === undefined ? "light" : obj.base;
    if (base !== "light" && base !== "dark") return { error: '"base" must be "light" or "dark"' };
    let name = obj.name === undefined ? "custom" : obj.name;
    if (typeof name !== "string") return { error: '"name" must be a string' };
    name = name.trim().slice(0, 64) || "custom";
    const given = obj.tokens === undefined ? {} : obj.tokens;
    if (!given || typeof given !== "object" || Array.isArray(given)) return { error: '"tokens" must be an object' };
    const tokens = Object.assign({}, DATA.presets[base].tokens);
    const notes = [];
    for (const k of Object.keys(given)) {
      if (!TOKENS.includes(k)) { notes.push(`unknown token "${k}" ignored`); continue; }
      const v = given[k];
      if (typeof v !== "string" || !HEX.test(v)) return { error: `token "${k}": "${String(v)}" is not #rrggbb` };
      tokens[k] = v.toLowerCase();
    }
    const missing = TOKENS.filter((k) => !(k in given));
    if (missing.length) notes.push(`taken from the ${base} preset: ${missing.join(", ")}`);
    for (const k of Object.keys(obj)) {
      if (!["schema", "name", "base", "tokens"].includes(k)) notes.push(`unknown key "${k}" ignored`);
    }
    return { theme: { schema: SCHEMA, name, base, tokens }, notes };
  }

  // Canonical JSON: fixed key order, every token, lower-case values. parse(serialize(t)) == t.
  function serializeTheme(t) {
    const tokens = {};
    for (const k of TOKENS) tokens[k] = t.tokens[k].toLowerCase();
    return JSON.stringify({ schema: SCHEMA, name: t.name, base: t.base, tokens }, null, 2) + "\n";
  }

  // ---------------------------------------------------------------- state and storage
  // Stored: {"mode": <MODES>, "custom": <theme or absent>}. Corrupt or foreign data is ignored.
  function loadState() {
    let raw = null;
    try { raw = window.localStorage.getItem(STORE_KEY); } catch (e) { raw = null; storageOk = false; }
    const st = { mode: "auto", custom: null };
    if (!raw) return st;
    try {
      const o = JSON.parse(raw);
      if (o && MODES.includes(o.mode)) st.mode = o.mode;
      if (o && o.custom) { const r = parseTheme(o.custom); if (r.theme) st.custom = r.theme; }
    } catch (e) { /* ignored: fall back to auto */ }
    if (st.mode === "custom" && !st.custom) st.mode = "auto";
    return st;
  }

  function saveState(st) {
    try {
      window.localStorage.setItem(STORE_KEY, JSON.stringify({ mode: st.mode, custom: st.custom || undefined }));
      return true;
    } catch (e) { return false; }
  }

  function systemDark() {
    try { return window.matchMedia("(prefers-color-scheme: dark)").matches; } catch (e) { return false; }
  }

  // The theme in force for a state (auto resolves through prefers-color-scheme).
  function effective(st) {
    if (st.mode === "custom" && st.custom) return st.custom;
    if (st.mode === "auto") return presetTheme(systemDark() ? "dark" : "light");
    return presetTheme(st.mode);
  }

  // Apply a state to <html>. Auto clears every inline property, so style.css (which carries the
  // light and dark presets for the no-script case) decides; otherwise every token is set inline.
  function apply(st) {
    const root = document.documentElement;
    if (st.mode === "auto") {
      root.removeAttribute("data-theme");
      for (const k of TOKENS) root.style.removeProperty("--" + k);
      root.style.removeProperty("color-scheme");
      return;
    }
    const t = effective(st);
    root.setAttribute("data-theme", t.base);
    for (const k of TOKENS) {
      if (HEX.test(t.tokens[k])) root.style.setProperty("--" + k, t.tokens[k]);
    }
    root.style.setProperty("color-scheme", t.base);
  }

  let state = { mode: "auto", custom: null };
  if (typeof document !== "undefined") {
    state = loadState();
    apply(state);
  }

  // ---------------------------------------------------------------- the controls
  function ui() {
    const $ = (id) => document.getElementById(id);
    const sel = $("theme-select"), btn = $("btn-theme"), warn = $("theme-warn"), box = $("theme-editor");
    if (!sel || !box) return;
    const mk = (tag, attrs, ...kids) => {
      const e = document.createElement(tag);
      for (const [k, v] of Object.entries(attrs || {})) {
        if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else e.setAttribute(k, v);
      }
      for (const c of kids) e.append(c);
      return e;
    };

    // editor skeleton (built once; contents refreshed by sync())
    const title = mk("span", { class: "theme-title" });
    const closeBtn = mk("button", { type: "button", text: "close" });
    const inputs = {};
    const grid = mk("div", { class: "theme-grid" });
    for (const k of TOKENS) {
      const inp = mk("input", { type: "color", id: "theme-tok-" + k, "aria-label": k });
      const hex = mk("code", { class: "theme-hex" });
      inputs[k] = { inp, hex };
      grid.append(mk("label", { for: "theme-tok-" + k, class: "theme-tok" },
        mk("span", { class: "theme-swatch-name", text: k }), mk("span", { class: "muted small", text: DATA.labels[k] })),
      mk("span", { class: "theme-tok-val" }, inp, hex));
      inp.addEventListener("input", () => editToken(k, inp.value));
    }
    const nameIn = mk("input", { type: "text", size: "18", maxlength: "64", "aria-label": "theme name" });
    const baseSel = mk("select", { "aria-label": "base" },
      mk("option", { value: "light", text: "light" }), mk("option", { value: "dark", text: "dark" }));
    const resetSel = mk("select", { "aria-label": "preset to reset to" });
    for (const id of Object.keys(DATA.presets)) resetSel.append(mk("option", { value: id, text: DATA.presets[id].name }));
    const resetBtn = mk("button", { type: "button", text: "reset to preset" });
    const exportBtn = mk("button", { type: "button", text: "export file" });
    const fileIn = mk("input", { type: "file", accept: ".json,application/json", class: "theme-file", "aria-label": "import file" });
    const importBtn = mk("button", { type: "button", text: "import file…" });
    const jsonArea = mk("textarea", { class: "theme-json", rows: "8", spellcheck: "false", "aria-label": "theme JSON" });
    const applyJsonBtn = mk("button", { type: "button", text: "apply JSON" });
    const msg = mk("p", { class: "theme-msg small", role: "status" });
    const report = mk("div", { class: "theme-report small" });
    const storeNote = mk("p", { class: "muted small" });

    box.append(
      mk("div", { class: "cell-bar" }, mk("span", { class: "label", text: "theme" }), title, mk("span", { class: "spacer" }), closeBtn),
      mk("div", { class: "theme-body" },
        mk("div", { class: "theme-row" }, mk("label", {}, "name ", nameIn), mk("label", {}, "base ", baseSel)),
        grid,
        mk("div", { class: "theme-row" }, resetSel, resetBtn),
        mk("h4", { text: "Contrast (WCAG AA)" }), report,
        mk("h4", { text: "Import / export (oc-notebook-theme/1)" }),
        mk("div", { class: "theme-row" }, exportBtn, importBtn, fileIn),
        jsonArea, mk("div", { class: "theme-row" }, applyJsonBtn), msg, storeNote));

    function say(text, cls) { msg.textContent = text || ""; msg.className = "theme-msg small " + (cls || ""); }

    function persist() {
      storageOk = saveState(state);
      noteStorage();
    }
    function noteStorage() {
      storeNote.textContent = storageOk
        ? "Kept in this browser (localStorage); other browsers and the gateway never see it."
        : "Browser storage is unavailable: the theme lasts only until this tab closes.";
    }

    function sync() {
      const t = effective(state);
      sel.value = state.mode;
      const customOpt = sel.querySelector('option[value="custom"]');
      if (customOpt) customOpt.textContent = state.custom ? `Custom: ${state.custom.name}` : "Custom…";
      title.textContent = state.mode === "auto" ? `auto → ${t.name}` : t.name;
      for (const k of TOKENS) { inputs[k].inp.value = t.tokens[k]; inputs[k].hex.textContent = t.tokens[k]; }
      nameIn.value = t.name;
      baseSel.value = t.base;
      const res = checkContrast(t.tokens);
      report.replaceChildren(...res.map((r) => mk("div", { class: "theme-pair " + (r.ok ? "" : "bad") },
        mk("span", { class: "theme-chip", style: `color:${t.tokens[r.fg]};background:${t.tokens[r.bg]}`, text: "Aa" }),
        mk("span", { text: ` ${r.fg} / ${r.bg}  ${r.ratio.toFixed(2)} : 1 ` }),
        mk("span", { class: r.ok ? "muted" : "err", text: r.ok ? `≥ ${r.floor}` : `below ${r.floor} (AA)` }))));
      const bad = res.filter((r) => !r.ok);
      if (warn) {
        warn.hidden = bad.length === 0;
        warn.textContent = bad.length ? `low contrast (${bad.length})` : "";
        warn.title = bad.map((r) => `${r.fg}/${r.bg} ${r.ratio.toFixed(2)}:1 < ${r.floor}:1`).join("\n");
      }
      jsonArea.value = serializeTheme(t);
    }

    // Any edit turns the theme into the custom one, seeded from the theme in force.
    function ensureCustom() {
      if (state.mode !== "custom") {
        const t = effective(state);
        state.custom = { schema: SCHEMA, name: "custom", base: t.base, tokens: Object.assign({}, t.tokens) };
        state.mode = "custom";
      }
    }
    function editToken(k, v) {
      if (!HEX.test(v)) return;
      ensureCustom();
      state.custom.tokens[k] = v.toLowerCase();
      apply(state); persist(); sync();
    }
    function setCustom(theme, notes) {
      state.custom = theme; state.mode = "custom";
      apply(state); persist(); sync();
      const bad = checkContrast(theme.tokens).filter((r) => !r.ok).length;
      say(`applied "${theme.name}"` + (notes && notes.length ? ` (${notes.join("; ")})` : "") +
          (bad ? ` — warning: ${bad} pair(s) below WCAG AA, see above` : ""), bad ? "err" : "");
    }
    function importText(text) {
      const r = parseTheme(text);
      if (r.error) { say("refused: " + r.error, "err"); return; }
      setCustom(r.theme, r.notes);
    }

    sel.addEventListener("change", () => {
      const m = sel.value;
      if (m === "custom" && !state.custom) { ensureCustom(); box.hidden = false; }
      state.mode = m;
      apply(state); persist(); sync(); say("");
      if (m === "custom") box.hidden = false;
    });
    if (btn) btn.addEventListener("click", () => { box.hidden = !box.hidden; });
    closeBtn.addEventListener("click", () => { box.hidden = true; });
    nameIn.addEventListener("change", () => {
      ensureCustom(); state.custom.name = nameIn.value.trim().slice(0, 64) || "custom"; persist(); sync();
    });
    baseSel.addEventListener("change", () => { ensureCustom(); state.custom.base = baseSel.value; apply(state); persist(); sync(); });
    resetBtn.addEventListener("click", () => {
      const p = presetTheme(resetSel.value);
      setCustom({ schema: SCHEMA, name: state.custom ? state.custom.name : "custom", base: p.base, tokens: p.tokens });
    });
    exportBtn.addEventListener("click", () => {
      const t = effective(state);
      const blob = new Blob([serializeTheme(t)], { type: "application/json" });
      const a = mk("a", { href: URL.createObjectURL(blob),
        download: "oc-theme-" + (t.name.replace(/[^A-Za-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "custom") + ".json" });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      say(`exported "${t.name}"`);
    });
    importBtn.addEventListener("click", () => fileIn.click());
    fileIn.addEventListener("change", () => {
      const f = fileIn.files && fileIn.files[0];
      if (!f) return;
      if (f.size > 65536) { say("refused: the file is larger than 64 KiB", "err"); fileIn.value = ""; return; }
      const rd = new FileReader();
      rd.onload = () => { importText(String(rd.result)); fileIn.value = ""; };
      rd.onerror = () => { say("refused: the file could not be read", "err"); fileIn.value = ""; };
      rd.readAsText(f);
    });
    applyJsonBtn.addEventListener("click", () => importText(jsonArea.value));
    try {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (state.mode === "auto") sync(); });
    } catch (e) { /* old browsers: auto still follows the OS through CSS, the editor shows the value at load */ }
    noteStorage(); sync();
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", ui); else ui();
  }

  return { DATA, SCHEMA, TOKENS, MODES, relLuminance, contrast, checkContrast, presetTheme, parseTheme, serializeTheme,
    state: () => state };
})();

if (typeof module !== "undefined" && module.exports) module.exports = OCTheme;
