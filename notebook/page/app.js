// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception

// app.js -- the notebook page: cells, run / interrupt / restart, completion, help, the
// atlas stub. Talks oc-gateway/1 over one WebSocket: frames from the binary are relayed
// unmodified (oc-serve/1, [P] Sec. 18.4); the gateway's own messages have kind "gw".
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const el = OCRender.el;

  // ---- token and tab id (the token travels in the URL fragment, never to the server log) ----
  function store(key, val) { try { if (val === undefined) return sessionStorage.getItem(key); sessionStorage.setItem(key, val); } catch (e) { return null; } return val; }
  const m = /(?:^#|&)token=([A-Za-z0-9_-]+)/.exec(location.hash);
  if (m) { store("oc_token", m[1]); history.replaceState(null, "", location.pathname); }
  let tab = store("oc_tab");
  if (!tab) {
    const b = new Uint8Array(8); crypto.getRandomValues(b);
    tab = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
    store("oc_tab", tab);
  }

  // ---- connection ----
  let ws = null, authorized = false, live = false, wantReconnect = true;
  const pending = new Map();     // request id -> handler {frame(fr), done(fr)}
  let reqCounter = 0;

  function setStatus(text, cls) { const s = $("status"); s.textContent = text; s.className = "status " + cls; }

  function connect() {
    const token = store("oc_token");
    if (!token) { $("token-box").hidden = false; setStatus("no token", "off"); return; }
    ws = new WebSocket(`ws://${location.host}/ws`);
    setStatus("connecting", "busy");
    ws.onopen = () => ws.send(JSON.stringify({ op: "auth", token, tab }));
    ws.onmessage = (ev) => { let fr; try { fr = JSON.parse(ev.data); } catch (e) { return; } onFrame(fr); };
    ws.onclose = () => {
      live = false; authorized = false;
      failAll("the connection closed");
      setStatus("disconnected", "off");
      if (wantReconnect) setTimeout(connect, 3000);
    };
  }

  function send(obj) { if (ws && ws.readyState === 1) { ws.send(JSON.stringify(obj)); return true; } return false; }

  function request(obj, handler) {
    const id = `r${++reqCounter}`;
    obj.id = id;
    pending.set(id, handler);
    if (!live || !send(obj)) { pending.delete(id); handler.done({ kind: "done", id, status: "rejected", exit: 1, error: { kind: "no_session", message: "no running session" } }); }
    return id;
  }

  function failAll(why) {
    for (const [id, h] of pending) h.done({ kind: "done", id, status: "error", exit: 3, error: { kind: "session", message: why } });
    pending.clear();
  }

  function onFrame(fr) {
    if (fr.kind === "gw") return onGateway(fr);
    if (fr.kind === "hello") { live = true; showHello(fr); setStatus("ready", "on"); return; }
    if (fr.kind === "fatal") { showInfo("last fatal", `${fr.reason}: ${fr.detail || ""}`); return; }
    if (fr.kind === "atlas_hello") { atlasHello(fr); return; }
    if (fr.kind === "atlas_fatal") { atlasNote(`atlas server ended: ${fr.reason}`); return; }
    const h = fr.id != null ? pending.get(fr.id) : null;
    if (!h) return;
    if (fr.kind === "done" || fr.kind === "atlas_reply") { pending.delete(fr.id); h.done(fr); } else h.frame(fr);
  }

  function onGateway(fr) {
    switch (fr.event) {
      case "authorized": authorized = true; $("token-box").hidden = true; setStatus("starting session", "busy"); break;
      case "unauthorized": wantReconnect = false; store("oc_token", ""); $("token-box").hidden = false; setStatus("token refused", "off"); break;
      case "session":
        showInfo("gateway limits", JSON.stringify(fr.limits)); showInfo("not served", (fr.unserved || []).join(", "));
        atlasConfigured(fr.atlas || null);
        break;
      case "atlas_exited": {
        atlasNote(`atlas server ended (${fr.reason}${fr.code != null ? ", exit " + fr.code : ""}); the next query restarts it`);
        const h = fr.id != null ? pending.get(fr.id) : null;
        if (h) { pending.delete(fr.id); h.done({ kind: "atlas_reply", id: fr.id, status: "error", error: { kind: "atlas_exited", message: fr.reason } }); }
        break;
      }
      case "rejected": {
        const h = fr.id != null ? pending.get(fr.id) : null;
        const done = { kind: "done", id: fr.id, status: "rejected", exit: 1, error: fr.error, origin: "gateway" };
        if (h) { pending.delete(fr.id); h.done(done); }
        break;
      }
      case "killing": showInfo("killing", `${fr.reason}: ${fr.detail || ""}`); break;
      case "cell_wall": {   // (NBCONFIRM S4) the gateway raised the wall limit of a confirmed cell
        const h = fr.id != null ? pending.get(fr.id) : null;
        if (h) h.frame(fr);
        break;
      }
      case "exited":
        live = false;
        failAll(`the session ended (${fr.reason}${fr.code != null ? ", exit " + fr.code : ""}); restart it`);
        setStatus(`session ended: ${fr.reason}`, "off");
        break;
      case "refused": live = false; wantReconnect = false; setStatus(`session refused: ${fr.reason}`, "off"); showInfo("refused", fr.detail || fr.reason); break;
      case "replaced": wantReconnect = false; setStatus("replaced by another connection of this tab", "off"); break;
      default: break;
    }
  }

  // ---- session panel ----
  function showInfo(k, v) {
    const dl = $("session-info");
    for (const dt of dl.querySelectorAll("dt")) if (dt.textContent === k) { dt.nextSibling.textContent = v; return; }
    dl.appendChild(el("dt", "", k)); dl.appendChild(el("dd", "", v));
  }
  function showHello(h) {
    $("session-info").textContent = "";
    showInfo("proto", `${h.proto} / json ${h.json_version} / console ${h.console}`);
    showInfo("build", `${(h.build.commit || "").slice(0, 12)} ${h.build.config}`);
    showInfo("pid", String(h.build.pid));
    showInfo("served", `${h.counts.verbs} verbs, ${h.counts.overloads} overloads, ${h.counts.types} types`);
    showInfo("rss cap", `${h.limits.rss_cap_mib} MiB`);
  }

  // ---- cells ----
  // (slot-24 NB-FILE) The cell list lives in one notebook object (cellbox.js makeNotebook): `cells` is its array.
  // It also holds the file's preamble, the unsaved mark and the execution count; this file only does the DOM and the session.
  let lastFocus = null;
  const nb = OCCells.makeNotebook({
    mount(cell, after) { if (after) after.root.after(cell.root); else $("cells").appendChild(cell.root); },
    unmount(cell) { cell.root.remove(); },
    wire(cell) {
      const { ta } = cell;
      if (cell.bRun) cell.bRun.addEventListener("click", () => runCell(cell));
      cell.bDel.addEventListener("click", () => { nb.remove(cell); showNbState(); });
      ta.addEventListener("keydown", (e) => onKey(e, cell));
      ta.addEventListener("input", () => { ta.rows = Math.max(2, ta.value.split("\n").length); markDirty(); });
      ta.addEventListener("focus", () => { lastFocus = cell; });
      cell.titleEl.addEventListener("input", markDirty);
      cell.titleEl.addEventListener("focus", () => { lastFocus = cell; });
    },
  });
  const cells = nb.cells;

  function markDirty() { nb.dirty = true; showNbState(); }
  function showNbState() { $("nb-name").textContent = (nb.fileName || "unsaved notebook") + (nb.dirty ? " •" : ""); $("nb-name").title = nb.dirty ? "unsaved changes" : "no unsaved changes"; }
  function addCell(src, after, opts) { const c = nb.add(src, after, opts); showNbState(); return c; }
  const currentCell = () => (lastFocus && cells.includes(lastFocus) ? lastFocus : cells[cells.length - 1] || null);

  // (slot-24 NB-OUTBOX) One cell runs at a time, in submission order; a submitted cell shows [...]
  // (queued) until its turn, then [*], then its execution count. The source is read at submission.
  const queue = OCCells.makeQueue(execJob);

  function runCell(cell, group) {
    return new Promise((resolve) => {
      if (cell.kind === "md" || cell.state === "running" || cell.state === "queued") return resolve(true);
      OCCells.setState(cell, "queued");
      queue.push({ cell, src: cell.ta.value, group: group || null, resolve });
    });
  }

  function execJob(job) {
    const { cell, src, group } = job;
    return new Promise((resolve) => {
      if (!cells.includes(cell)) { job.resolve(false); return resolve(); }
      if (group && group.aborted) { OCCells.cancelQueued(cell); job.resolve(false); return resolve(); }
      OCCells.beginRun(cell, ++nb.execCount);   // clears the previous output box
      let lastOut = null;             // {pre, text, stream}
      const t0 = performance.now();
      setStatus("running", "busy");
      // (NBCONFIRM S4, [P] Sec. 18.4 / 18.19) A call outside its silent cost envelope stops the cell with
      // status "confirm_required"; after the dialog the SAME source is re-sent with the tokens so far
      // (one per expensive call). The re-send RE-EXECUTES the statements before that call, with the
      // bindings the stopped run left: their side effects happen again and earlier confirmed calls run
      // again in full; a cell that rebinds what it reads (k = k + 2) is not idempotent. A site asks at
      // most once per cell run (OCRender.confirmStep): asked again, the page stops and explains.
      const asked = [];   // the call sites that have asked in this cell run
      const attempt = (tokens) => {
        const prog = { line: null, wall: null };   // the live progress line of a confirmed run
        const req = { op: "eval", src };
        if (tokens.length) req.confirm = tokens;
        let ended = false;   // done may run synchronously inside request() (no session)
        const id = request(req, {
          frame(fr) { lastOut = appendFrame(cell, cell.out, fr, lastOut, src, prog); OCCells.reveal(cell); },
          done(fr) {
            ended = true; runningId = null; killTimer.cancel();
            const step = OCRender.confirmStep(asked, tokens, fr);
            if (step.action === "stop") return finish(Object.assign({}, fr, { status: "not run" }), step.message);
            if (step.action === "ask") {
              asked.push(step.site);
              OCCells.reveal(cell);
              confirmDialog(fr, (run) => {
                if (!run || !cells.includes(cell)) return finish(Object.assign({}, fr, { status: "not run" }), "not run: the confirmation was declined");
                cell.out.textContent = ""; lastOut = null;
                attempt(tokens.concat([fr.confirm.token]));
              });
              return;
            }
            finish(fr, null);
          },
        });
        if (!ended) runningId = id;   // (audit N1) never a stale id after a failed send
      };
      const finish = (fr, note) => {
          const ok = fr.status === "ok";
          let line = `${fr.status} (exit ${fr.exit})`;
          if (fr.perf) line += ` \u00b7 ${fr.perf.elapsed_ms} ms \u00b7 rss ${fr.perf.rss_mib} MiB (peak ${fr.perf.peak_rss_mib})`;
          else line += ` \u00b7 ${Math.round(performance.now() - t0)} ms`;
          if (fr.origin === "gateway") line += " \u00b7 refused by the gateway";
          if (note) cell.out.appendChild(el("pre", "out err", note));
          else if (fr.error) {
            const e = fr.error;
            cell.out.appendChild(el("pre", "out err", `${e.kind}: ${e.message || ""}${e.detail ? "\n" + e.detail : ""}`));
          }
          OCCells.finishRun(cell, ok, line);
          if (!ok && group) group.aborted = true;
          if (live && !queue.pending()) setStatus("ready", "on");
          job.resolve(ok);
          resolve();
      };
      attempt([]);
    });
  }

  // ---- (NBCONFIRM S4) the confirmation dialog, the live progress line, the kill fallback ----
  let runningId = null;
  function fmtSeconds(s) {
    if (typeof s !== "number" || !isFinite(s)) return "?";
    if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
    if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
    if (s < 86400) return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
    return `${(s / 86400).toFixed(1)} days`;
  }
  function fmtMiB(m) { return typeof m !== "number" || !isFinite(m) ? "?" : m >= 1024 ? `${(m / 1024).toFixed(1)} GiB` : `${Math.round(m)} MiB`; }
  // `fr` is a done frame with status "confirm_required"; `answer(run)` is called once.
  function confirmDialog(fr, answer) {
    const c = fr.confirm, strong = c.level === "strong";
    const overlay = el("div", "confirm-overlay");
    const box = el("div", "confirm-dialog" + (strong ? " strong" : ""));
    box.setAttribute("role", "dialog"); box.setAttribute("aria-modal", "true"); box.setAttribute("aria-labelledby", "confirm-title");
    const title = el("h3", "", strong ? "Strong warning: run this long computation?" : "Run this computation?");
    title.id = "confirm-title";
    box.appendChild(title);
    if (strong) box.appendChild(el("div", "confirm-banner", "STRONG WARNING: " + (c.reasons || []).join("; ")));
    box.appendChild(el("div", "confirm-call", c.call || c.verb || ""));
    box.appendChild(el("div", "muted", `outside its silent envelope: ${c.bound || ""}`));
    const dl = el("dl", "confirm-est");
    const row = (k, v) => { dl.appendChild(el("dt", "", k)); dl.appendChild(el("dd", "", v)); };
    row("wall time", `~ ${fmtSeconds(c.wall_s)} (up to ~ ${fmtSeconds(c.wall_hi)})`);
    row("memory", `~ ${fmtMiB(c.rss_mib)} (up to ~ ${fmtMiB(c.rss_hi)})` + (c.mem_limit_mib ? `, limit ${fmtMiB(c.mem_limit_mib)}` : ""));
    row("threads", String(c.threads));
    row("model", `${c.model}; ${c.calibration}` + (c.extrapolated ? `; extrapolated \u00d7${(c.x_factor || 1).toFixed(1)}` : "") + (c.low_confidence ? "; low confidence" : ""));
    if (typeof c.kill_wall_s === "number") row("session wall limit", `${fmtSeconds(c.kill_wall_s)} at most (the gateway kills the session beyond it)`);
    box.appendChild(dl);
    for (const n of c.notes || []) box.appendChild(el("div", "muted", "note: " + n));
    // (NBCONFIRM S4 fix round 1, audit M2 / L1)
    box.appendChild(el("div", "muted", "Run re-sends the whole cell: its statements before this call run again (side effects and earlier confirmed calls included)."));
    box.appendChild(el("div", "muted", "The estimate is for one run of the call; if the cell repeats it (a loop), every repetition runs under this one confirmation, bounded by the session wall limit."));
    const btns = el("div", "confirm-buttons");
    const run = el("button", "confirm-run", "Run"), cancel = el("button", "", "Cancel");
    if (strong) {
      const lab = el("label", "confirm-ack");
      const ck = el("input"); ck.type = "checkbox";
      lab.appendChild(ck); lab.appendChild(document.createTextNode(" I have read the warning and want to run it"));
      box.appendChild(lab);
      run.disabled = true;
      ck.addEventListener("change", () => { run.disabled = !ck.checked; });
    }
    btns.appendChild(cancel); btns.appendChild(run); box.appendChild(btns);
    overlay.appendChild(box); document.body.appendChild(overlay);
    let answered = false;
    const close = (yes) => { if (answered) return; answered = true; overlay.remove(); answer(yes); };
    run.addEventListener("click", () => close(true));
    cancel.addEventListener("click", () => close(false));
    overlay.addEventListener("keydown", (e) => { if (e.key === "Escape") close(false); });
    cancel.focus();
  }
  // Interrupt is cooperative: the running call stops at its next cancellation point ([P] Sec. 18.6; NBCONFIRM S5: every
  // statement and call, and the long verbs' loop steps). After 10 s with no `done` (a step longer than that, or a call with
  // no point inside), offer "Kill session" (the restart op: every binding of the session is lost).
  const killTimer = (() => {
    let t = null;
    const btn = () => $("btn-kill");
    return {
      arm() { if (t || !runningId) return; t = setTimeout(() => { t = null; if (runningId) { btn().hidden = false; setStatus("interrupt pending: the running call has not stopped", "busy"); } }, 10000); },
      cancel() { if (t) { clearTimeout(t); t = null; } if (btn()) btn().hidden = true; },
    };
  })();

  // One frame of an eval into `box`: stdout / log text, progress lines, value blocks. Returns the
  // open text run (or null). `src` is the evaluated source (for the value actions below).
  function appendFrame(cell, box, fr, lastOut, src, live) {
    if (fr.kind === "gw" && fr.event === "cell_wall") {   // (NBCONFIRM S4)
      if (live) live.wall = fr.cell_wall_s;
      box.appendChild(el("div", "progress", `#~ gateway: wall limit of this confirmed cell ${fmtSeconds(fr.cell_wall_s)}`));
      return null;
    }
    if (fr.kind === "progress" && fr.confirmed && live) {   // (NBCONFIRM S4) one live line per confirmed run
      const c = fr.confirmed;
      if (!live.line || live.line.dataset.call !== c.call) {
        live.line = el("div", "progress live");
        live.line.dataset.call = c.call;
        box.appendChild(live.line);
      }
      live.line.textContent = `#~ running ${c.call}: ${fmtSeconds(c.elapsed_s)} of ~ ${fmtSeconds(c.wall_s)} (up to ~ ${fmtSeconds(c.wall_hi)}), resident ${fmtMiB(c.resident_mib)}` +
        (live.wall ? ` \u00b7 wall limit ${fmtSeconds(live.wall)}` : "");
      live.line.title = fr.text;
      return null;
    }
    if (fr.kind === "out") {
      if (!lastOut || lastOut.stream !== fr.stream) {
        lastOut = { pre: el("pre", fr.stream === "log" ? "out log" : "out"), text: "", stream: fr.stream };
        box.appendChild(lastOut.pre);
      }
      lastOut.text += fr.text;
      lastOut.pre.textContent = lastOut.text;
      return lastOut;
    }
    if (fr.kind === "progress") {
      box.appendChild(el("div", "progress", "#~ " + fr.text));
      return null;
    }
    if (fr.kind === "value") {
      // The REPL echo (or display's) printed the canonical text just before this frame: replace it.
      const echo = fr.value.canonical + "\n";
      if (lastOut && lastOut.stream === "stdout" && lastOut.text.endsWith(echo)) {
        lastOut.text = lastOut.text.slice(0, -echo.length);
        lastOut.pre.textContent = lastOut.text;
        if (!lastOut.text) lastOut.pre.remove();
      }
      const holder = { box: null };
      holder.box = OCRender.renderValue(fr.value, valueActions(cell, fr, src, holder));
      box.appendChild(holder.box);
      return null;
    }
    return lastOut;
  }

  // (slot-24 NB-RENDER R3) The statement behind a value frame, for re-evaluation: the source text
  // at its line / col; a frame sent by display(x) points at the call, so display(x) -> x. null when
  // it cannot be recovered (display with several arguments, an unbalanced text).
  function frameExpr(src, fr) {
    let e = OCRender.stmtSource(src, fr.line, fr.col);
    if (!e) return null;
    const m = /^display\s*\(([\s\S]*)\)$/.exec(e);
    if (m) {
      const inner = m[1].trim();
      let depth = 0, inStr = false;
      for (let i = 0; i < inner.length; ++i) {
        const c = inner[i];
        if (inStr) { if (c === "\\") ++i; else if (c === '"') inStr = false; continue; }
        if (c === '"') inStr = true;
        else if ("([{".includes(c)) ++depth;
        else if (")]}".includes(c)) { if (--depth < 0) return null; }
        else if (c === "," && depth === 0) return null;
      }
      e = inner;
    }
    return e || null;
  }

  // A side evaluation of `src` whose output goes right after `anchor` (inside the cell's output).
  function sideEval(cell, src, anchor, done) {
    const box = el("div", "side");
    box.appendChild(el("div", "muted small", src));
    anchor.after(box);
    let lastOut = null;
    request({ op: "eval", src }, {
      frame(fr) { lastOut = appendFrame(cell, box, fr, lastOut, src); },
      done(fr) {
        if (fr.status !== "ok" && fr.error) box.appendChild(el("pre", "out err", `${fr.error.kind}: ${fr.error.message || ""}`));
        if (done) done(fr, box);
      },
    });
  }

  // The actions of one value block: "compute" (a declared handle) and "show all" (a shape-only Matrix).
  function valueActions(cell, fr, src, holder) {
    const expr = frameExpr(src, fr);
    if (!expr) return null;
    return {
      compute() {
        sideEval(cell, `display(${expr})`, holder.box, (d) => { if (d.status === "ok") holder.box.remove(); });
      },
      showAll(n) {
        // Read print_max_entries, raise it for one display(...), then restore it whatever happened.
        let old = null;
        request({ op: "eval", src: "settings().print_max_entries.value" }, {
          frame(f) { if (f.kind === "value" && f.value && OCRender.isInt(f.value.v)) old = String(f.value.v); },
          done() {
            const restore = () => { if (old !== null) request({ op: "eval", src: `set print_max_entries = ${old};` }, { frame() {}, done() {} }); };
            sideEval(cell, `set print_max_entries = ${n};\ndisplay(${expr})`, holder.box, (d) => {
              restore();
              if (d.status === "ok") holder.box.remove();
            });
          },
        });
      },
    };
  }

  async function runAll() {
    const group = { aborted: false };
    await Promise.all(cells.filter((c) => c.kind !== "md").map((c) => runCell(c, group)));
  }

  // ---- completion ----
  let pop = null;
  function closePop() { if (pop) { pop.remove(); pop = null; } }
  function insertAtCursor(ta, text) {
    const s = ta.selectionStart;
    ta.setRangeText(text, s, ta.selectionEnd, "end");
    ta.dispatchEvent(new Event("input"));
  }
  function complete(cell) {
    const ta = cell.ta;
    const before = ta.value.slice(0, ta.selectionStart);
    const mm = /[A-Za-z_][A-Za-z0-9_]*$/.exec(before);
    if (!mm) { insertAtCursor(ta, "    "); return; }
    const prefix = mm[0];
    request({ op: "complete", prefix }, {
      items: [],
      frame(fr) { if (fr.kind === "completions") this.items = fr.items || []; },
      done() {
        const items = this.items;
        closePop();
        if (!items.length) return;
        let common = items[0].text;
        for (const it of items) { let i = 0; while (i < common.length && common[i] === it.text[i]) ++i; common = common.slice(0, i); }
        if (common.length > prefix.length) insertAtCursor(ta, common.slice(prefix.length));
        if (items.length === 1) return;
        pop = el("div", "complete-pop");
        for (const it of items.slice(0, 200)) {
          const row = el("div", "", it.sig ? it.sig : `${it.text}  (${it.kind})`);
          row.addEventListener("mousedown", (e) => {
            e.preventDefault();
            const cur = /[A-Za-z_][A-Za-z0-9_]*$/.exec(ta.value.slice(0, ta.selectionStart));
            insertAtCursor(ta, it.text.slice(cur ? cur[0].length : 0));
            closePop(); ta.focus();
          });
          pop.appendChild(row);
        }
        const r = ta.getBoundingClientRect();
        pop.style.left = `${r.left + window.scrollX}px`;
        pop.style.top = `${r.bottom + window.scrollY}px`;
        document.body.appendChild(pop);
      },
    });
  }

  function onKey(e, cell) {
    if (cell.kind === "md") {   // a text cell: Shift+Enter shows the text and moves on; nothing runs
      if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); OCCells.setMdEditing(cell, false); const i = cells.indexOf(cell); if (cells[i + 1]) (cells[i + 1].ta.hidden ? cells[i + 1].bEdit : cells[i + 1].ta).focus(); }
      return;
    }
    if (e.key === "Enter" && (e.shiftKey || e.ctrlKey || e.metaKey || e.altKey)) {
      // Shift+Enter: run, go to the next cell (a new one after the last); Ctrl/Cmd+Enter: run, stay; Alt+Enter: run, new cell below.
      e.preventDefault(); closePop(); runCell(cell);
      const i = cells.indexOf(cell);
      if (e.altKey) addCell("", cell).ta.focus();
      else if (e.shiftKey) { if (i === cells.length - 1) addCell("", cell).ta.focus(); else cells[i + 1].ta.focus(); }
    }
    else if (e.key === "Tab" && !e.shiftKey) { e.preventDefault(); if (!cell.running) complete(cell); }
    else if (e.key === "Escape") closePop();
  }

  // ---- help ----
  function renderHelp(fr) {
    const box = $("help-out"); box.textContent = "";
    for (const v of fr.verbs || []) {
      const d = el("div");
      d.appendChild(el("strong", "", v.name));
      if (v.group) d.appendChild(el("span", "muted", `  ${v.group}`));
      for (const o of v.overloads || []) {
        d.appendChild(el("div", "sig", o.sig));
        if (o.doc) d.appendChild(el("div", "muted", o.doc));
      }
      box.appendChild(d);
    }
    for (const [label, list] of [["types", fr.types], ["settings", fr.settings]]) {
      if (!list || !list.length) continue;
      box.appendChild(el("h4", "", label));
      for (const t of list) box.appendChild(el("div", "sig", typeof t === "string" ? t : (t.name ? `${t.name}${t.doc ? " -- " + t.doc : ""}` : JSON.stringify(t))));
    }
    if (!box.childNodes.length) box.textContent = "(nothing served under that name)";
  }

  // ---- the atlas panel (slot-20 NB-4, [P] v3.11 Sec. 18.16 (g)) ----
  // Queries go page -> gateway as {op:"atlas", id, query, ...fields}; the gateway forwards them to the
  // session's `--serve-atlas` child and relays its `atlas_hello` / `atlas_reply` frames unmodified.
  // No store string reaches innerHTML or TeX: text nodes only, TeX from validated integers (render.js).
  const atlas = { configured: false, header: null, fnv: null, ell: null, store: null };
  const isUint = (x) => Number.isInteger(x) && x >= 0;

  function atlasConfigured(a) {
    atlas.configured = !!(a && a.configured);
    $("atlas-none").hidden = atlas.configured;
    $("atlas-store").hidden = !atlas.configured;
    if (atlas.configured) { atlas.store = a.store; showAtlasInfo("store", String(a.store || "")); }
  }
  function showAtlasInfo(k, v) {
    const dl = $("atlas-store-info");
    for (const dt of dl.querySelectorAll("dt")) if (dt.textContent === k) { dt.nextSibling.textContent = v; return; }
    dl.appendChild(el("dt", "", k)); dl.appendChild(el("dd", "", v));
  }
  function atlasHello(h) {
    const s = h.store || {};
    atlas.fnv = s.fnv; atlas.ell = s.ell;
    showAtlasInfo("store", `${s.name} (${s.bytes} bytes)`);
    showAtlasInfo("fnv", String(s.fnv));
    showAtlasInfo("ell", String(s.ell));
    showAtlasInfo("counts", `${s.cells} cells, ${s.systems} systems, ${s.occurrences} occurrences`);
    showAtlasInfo("index", `${(h.index || {}).source} ${(h.index || {}).fnv || ""}`);
  }
  function atlasNote(text) { showAtlasInfo("note", text); }

  function atlasRequest(query, fields, done) {
    const obj = Object.assign({ op: "atlas", query }, fields || {});
    request(obj, { frame() {}, done });
    // The request line the binary receives (the "copy request" button reproduces it).
    const line = Object.assign({ op: query }, fields || {});
    return JSON.stringify(line);
  }

  function view(title) {
    $("atlas-view").hidden = false;
    $("atlas-view-title").textContent = title;
    const b = $("atlas-view-body"); b.textContent = "";
    return b;
  }
  function footer(box, reqLine, fr) {
    const f = el("div", "atlas-foot muted");
    f.appendChild(document.createTextNode(`store ${fr && fr.store ? fr.store : atlas.fnv || "?"} · `));
    const b = el("button", "", "copy request");
    b.type = "button";
    b.addEventListener("click", () => { try { navigator.clipboard.writeText(reqLine); b.textContent = "copied"; } catch (e) { b.textContent = reqLine; } });
    f.appendChild(b);
    if (atlas.ell === 5 || atlas.ell === 7 || atlas.ell === 11 || atlas.ell === 13) {
      const c = el("button", "", `insert atlas(${atlas.ell}) cell`);
      c.type = "button";
      c.addEventListener("click", () => addCell(`A = atlas(${atlas.ell});\nA`).ta.focus());
      f.appendChild(c);
    }
    box.appendChild(f);
  }
  function failed(box, fr) {
    const e = fr.error || {};
    box.appendChild(el("pre", "out err", `${fr.status}: ${e.kind || ""} ${e.message || ""}`));
  }
  function table(head) {
    const t = el("table", "rec atlas-table");
    const tr = el("tr");
    for (const h of head) tr.appendChild(el("th", "", h));
    t.appendChild(tr);
    return t;
  }
  function td(tr, content) {
    const c = el("td");
    if (content instanceof Node) c.appendChild(content); else c.textContent = String(content);
    tr.appendChild(c);
    return c;
  }
  function link(text, onClick) {
    const a = el("button", "linkish", text);
    a.type = "button";
    a.addEventListener("click", onClick);
    return a;
  }

  function showHeader() {
    const reqLine = atlasRequest("info", {}, (fr) => {
      const box = view("store header");
      if (fr.status !== "ok") return failed(box, fr);
      const r = fr.result, h = r.header;
      atlas.header = h;
      const dl = el("dl", "kv");
      const kv = (k, v) => { dl.appendChild(el("dt", "", k)); const dd = el("dd"); if (v instanceof Node) dd.appendChild(v); else dd.textContent = String(v); dl.appendChild(dd); };
      kv("schema", h.schema); kv("ell", h.ell); kv("n_max", h.n_max); kv("primes T", (h.primes || []).join(", "));
      kv("cells", r.cells); kv("systems", r.systems); kv("collected", r.collected);
      kv("types", `eisenstein ${r.types.eisenstein}, cuspidal ${r.types.cuspidal}`);
      kv("r profile", (r.r_profile || []).map((p) => `${p[0]}:${p[1]}`).join("  "));
      kv("chi", `${h.chi_count} characters chi_{a,b}, ${h.chi_in_sigma} in Sigma`);
      if (h.s_sep) kv("S_sep", `{${(h.s_sep.primes || []).join(", ")}} separates=${h.s_sep.separates} unseparated=${h.s_sep.unseparated}`);
      if (h.s0_anemic) kv("S_0 anemic", `{${(h.s0_anemic.primes || []).join(", ")}} separates=${h.s0_anemic.separates}`);
      box.appendChild(dl);
      const fl = el("details");
      fl.appendChild(el("summary", "muted", `the canonical fields (${(h.fields || []).length})`));
      for (const f of (h.fields || []).slice(0, 80)) { const d = el("div"); d.appendChild(OCRender.mathSpan(OCRender.atlasFieldTeX(h.ell, f))); fl.appendChild(d); }
      box.appendChild(fl);
      footer(box, reqLine, fr);
    });
  }

  function pager(box, res, limit, go) {
    const p = el("div", "atlas-pager muted");
    const n = res.items.length;
    p.appendChild(document.createTextNode(`${n ? res.offset + 1 : 0}–${res.offset + n} of ${res.total}  `));
    if (res.offset > 0) p.appendChild(link("‹ prev", () => go(Math.max(0, res.offset - limit))));
    if (res.next !== null && res.next !== undefined) p.appendChild(link("next ›", () => go(res.next)));
    box.appendChild(p);
  }

  function showCells(offset) {
    const limit = 100;
    const reqLine = atlasRequest("cells", { offset, limit }, (fr) => {
      const box = view("cells");
      if (fr.status !== "ok") return failed(box, fr);
      const res = fr.result;
      pager(box, res, limit, showCells);
      const t = table(["n", "dim H^0, H^1, H^2", "ideals", "local", "U_ell"]);
      for (const c of res.items) {
        const tr = el("tr");
        td(tr, link(String(c.n), () => showCell(c.n)));
        td(tr, (c.dims || []).join(", "));
        td(tr, c.ideals); td(tr, c.local_ideals);
        td(tr, c.u_ell ? (c.u_ell.included ? "included" : `not included (${c.u_ell.reason})`) : "");
        t.appendChild(tr);
      }
      box.appendChild(t);
      footer(box, reqLine, fr);
    });
  }

  function showCell(n) {
    if (!isUint(n)) return;
    const reqLine = atlasRequest("cell", { n }, (fr) => {
      const box = view(`cell n = ${n}`);
      if (fr.status !== "ok") return failed(box, fr);
      const c = fr.result.cell;
      box.appendChild(el("div", "", `weight k = ${n + 2}: dims ${(c.dims || []).join(", ")}, ${c.ideals} ideals (${c.local_ideals} local), U_ell ${c.u_ell && c.u_ell.included ? "included" : "not included"}`));
      const t = table(["ideal", "system"]);
      for (const o of fr.result.occurrences) {
        const tr = el("tr");
        td(tr, o.ideal);
        td(tr, link(`#${o.sid}`, () => showSystem(o.sid)));
        t.appendChild(tr);
      }
      box.appendChild(t);
      box.appendChild(link(`all systems occurring at n = ${n}`, () => findSystems({ n }, 0)));
      footer(box, reqLine, fr);
    });
  }

  function systemBlock(s) {
    const box = el("div", "atlas-system");
    const head = el("div", "vhead");
    head.appendChild(document.createTextNode(`system #${s.id} · r = ${s.r} · ${s.type === "eisenstein" ? "Eisenstein" : "cuspidal"} · `));
    head.appendChild(OCRender.mathSpan(OCRender.atlasTwistTeX(s.twist, s.base)));
    head.appendChild(document.createTextNode(` · ${s.collected ? "collected" : "twist only"}${s.in_H1 ? " · in H^1" : ""} · `));
    if (s.chi) head.appendChild(OCRender.mathSpan(OCRender.atlasChiTeX(s.chi)));
    box.appendChild(head);
    const h = atlas.header;
    if (h && Array.isArray(h.fields)) {
      const f = h.fields.find((x) => x.r === s.r);
      if (f) { const d = el("div", "muted"); d.appendChild(OCRender.mathSpan(OCRender.atlasFieldTeX(h.ell, f))); box.appendChild(d); }
    }
    const primes = h && Array.isArray(h.primes) ? h.primes : null;
    const t = table(["q", "a_q"]);
    (s.eigenvalues || []).forEach((v, i) => {
      const tr = el("tr");
      td(tr, primes && isUint(primes[i]) ? String(primes[i]) : `T[${i}]`);
      td(tr, OCRender.mathSpan(OCRender.atlasEltTeX(v)));
      t.appendChild(tr);
    });
    box.appendChild(t);
    if ((s.occurrences || []).length) {
      const o = table(["n", "ideal", "group", "dim", "a_ell"]);
      for (const oc of s.occurrences) {
        const tr = el("tr");
        td(tr, link(String(oc.n), () => showCell(oc.n)));
        td(tr, oc.ideal); td(tr, oc.group); td(tr, oc.dim);
        const a = oc.a_ell || {};
        td(tr, a.status === "known" ? OCRender.mathSpan(OCRender.atlasEltTeX(a.value)) : `unknown (${a.reason})`);
        o.appendChild(tr);
      }
      box.appendChild(o);
    }
    box.appendChild(link(`the twists of #${s.base}`, () => findSystems({ base: s.base }, 0)));
    return box;
  }

  // The header (primes, fields) first when it is not known yet: one atlas request in flight at a time.
  function withHeader(fn) {
    if (atlas.header) return fn();
    atlasRequest("info", {}, (fr) => { atlas.header = fr.status === "ok" ? fr.result.header : {}; fn(); });
  }

  function showSystem(sid) {
    if (!isUint(sid)) return;
    if (!atlas.header) return withHeader(() => showSystem(sid));
    const reqLine = atlasRequest("system", { sid }, (fr) => {
      const box = view(`system #${sid}`);
      if (fr.status !== "ok") return failed(box, fr);
      box.appendChild(systemBlock(fr.result.system));
      footer(box, reqLine, fr);
    });
  }

  function findSystems(filter, offset) {
    if (!atlas.header) return withHeader(() => findSystems(filter, offset));
    const limit = 25;
    const reqLine = atlasRequest("systems", { filter, offset, limit }, (fr) => {
      const box = view(`systems ${JSON.stringify(filter)}`);
      if (fr.status !== "ok") return failed(box, fr);
      const res = fr.result;
      pager(box, res, limit, (o) => findSystems(filter, o));
      const t = table(["#", "r", "type", "twist", "collected", "first a_q"]);
      for (const s of res.items) {
        const tr = el("tr");
        td(tr, link(`#${s.id}`, () => showSystem(s.id)));
        td(tr, s.r); td(tr, s.type);
        td(tr, OCRender.mathSpan(OCRender.atlasTwistTeX(s.twist, s.base)));
        td(tr, s.collected ? "yes" : "no");
        td(tr, OCRender.mathSpan(OCRender.atlasEltTeX((s.eigenvalues || [])[0])));
        t.appendChild(tr);
      }
      box.appendChild(t);
      footer(box, reqLine, fr);
    });
  }

  function uintField(id) {
    const t = $(id).value.trim();
    if (t === "") return undefined;
    if (!/^[0-9]{1,10}$/.test(t)) return null;
    return Number(t);
  }

  // ---- wiring ----
  $("btn-add").addEventListener("click", () => addCell("").ta.focus());
  $("btn-add-md").addEventListener("click", () => addCell("", currentCell(), { kind: "md" }).ta.focus());

  // ---- (slot-24 NB-RENDER R2) the LaTeX switch and the plain-text export ----
  // The switch changes presentation only: the envelopes are kept and re-rendered; values never change.
  const texBox = $("tex-toggle");
  texBox.checked = OCRender.texEnabled();
  texBox.addEventListener("change", () => { OCRender.setTeX(texBox.checked); OCRender.rerenderAll(document); });
  function download(name, text, type) {
    const blob = new Blob([text], { type: `${type};charset=utf-8` });
    const a = el("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }
  $("btn-export").addEventListener("click", () => download("notebook_output.txt", OCCells.exportText(OCCells.snapshot(cells)), "text/plain"));
  $("btn-export-code").addEventListener("click", () => download("notebook_code.oc", OCCells.exportCode(OCCells.snapshot(cells)), "text/plain"));
  $("btn-export-json").addEventListener("click", () => download("notebook.json", OCCells.exportJSON(OCCells.snapshot(cells)), "application/json"));
  $("btn-run-all").addEventListener("click", runAll);
  $("btn-interrupt").addEventListener("click", () => {
    for (const j of queue.clear()) { OCCells.cancelQueued(j.cell); j.resolve(false); }   // queued cells are dropped
    send({ op: "interrupt" });
    killTimer.arm();
  });
  $("btn-kill").addEventListener("click", () => {
    if (!window.confirm("Kill the session? The running call stops, and every binding of the session is lost.")) return;
    killTimer.cancel();
    newSession();
  });
  // A new session: queued cells are dropped, running requests fail, the process restarts, the execution count starts again.
  function newSession() {
    for (const j of queue.clear()) { OCCells.cancelQueued(j.cell); j.resolve(false); }
    failAll("session restarted");
    setStatus("restarting", "busy");
    send({ op: "restart" });
    nb.restart();
  }
  $("btn-restart").addEventListener("click", newSession);

  // ---- (slot-24 NB-FILE) open / insert / save a notebook file (.oc with `# %%` cells; ocfile.js) ----
  const MAX_FILE = 1048576;
  function fileNote(text, bad) { const n = $("file-note"); n.textContent = text; n.className = "file-note " + (bad ? "bad" : "muted"); n.hidden = !text; }
  function readFile(file) {
    return new Promise((resolve, reject) => {
      if (file.size > MAX_FILE) return reject(new Error(`${file.name}: larger than 1 MiB`));
      const fr = new FileReader();
      fr.onload = () => resolve(String(fr.result));
      fr.onerror = () => reject(new Error(`${file.name}: cannot be read`));
      fr.readAsText(file);
    });
  }
  function parseOrNote(text, name) {
    try { return OCFile.parse(text); } catch (e) { fileNote(`${name}: ${e.message}`, true); return null; }
  }
  // Open as an entire notebook: the cells are replaced and the session restarts (after a confirmation if unsaved).
  function openText(text, name) {
    const parsed = parseOrNote(text, name);
    if (!parsed) return false;
    if (nb.dirty && !window.confirm("The notebook has unsaved changes. Replace all cells and restart the session?")) return false;
    newSession();
    nb.openParsed(parsed, name);
    lastFocus = null;
    showNbState();
    fileNote(`opened ${name}: ${cells.length} cell${cells.length === 1 ? "" : "s"}`);
    return true;
  }
  // Insert the file's cells below `after` (the current cell by default); the session is untouched.
  function insertText(text, name, after) {
    const parsed = parseOrNote(text, name);
    if (!parsed) return false;
    const made = nb.insertParsed(parsed, after === undefined ? currentCell() : after);
    showNbState();
    fileNote(`inserted ${name}: ${made.length} cell${made.length === 1 ? "" : "s"}`);
    if (made.length) made[0].ta.focus();
    return true;
  }
  let fileMode = "open";
  const fileInput = $("file-input");
  function chooseFile(mode) { fileMode = mode; fileInput.value = ""; fileInput.click(); }
  fileInput.addEventListener("change", async () => {
    const f = fileInput.files && fileInput.files[0];
    if (!f) return;
    try { const t = await readFile(f); fileMode === "insert" ? insertText(t, f.name) : openText(t, f.name); }
    catch (e) { fileNote(e.message, true); }
  });
  $("btn-open").addEventListener("click", () => chooseFile("open"));
  $("btn-insert").addEventListener("click", () => chooseFile("insert"));
  $("btn-save").addEventListener("click", () => {
    let name = nb.fileName || "notebook.oc";
    if (!/\.oc$/.test(name)) name += ".oc";
    download(name, nb.toFile(), "text/plain");
    nb.fileName = name; nb.dirty = false; showNbState();
  });
  // Drag and drop: a file dropped on a cell is inserted below it; dropped elsewhere it opens as the notebook.
  document.addEventListener("dragover", (e) => { if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files")) e.preventDefault(); });
  document.addEventListener("drop", async (e) => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    e.preventDefault();
    const wrap = e.target && e.target.closest ? e.target.closest(".cell-wrap") : null;
    const after = wrap ? cells.find((c) => c.root === wrap) : null;
    try { const t = await readFile(f); after ? insertText(t, f.name, after) : openText(t, f.name); }
    catch (err) { fileNote(err.message, true); }
  });
  window.addEventListener("beforeunload", (e) => { if (nb.dirty) { e.preventDefault(); e.returnValue = ""; } });
  // The examples the gateway offers (a read-only listing of one fixed directory; [P] Sec. 18.18): open one as the notebook.
  fetch("/examples/index.json", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null)).then((ix) => {
    if (!ix || !Array.isArray(ix.files) || !ix.files.length) return;
    const sel = $("example-select");
    for (const n of ix.files) if (/^[A-Za-z0-9][A-Za-z0-9_-]*\.oc$/.test(n)) { const o = el("option", "", n); o.value = n; sel.appendChild(o); }
    sel.hidden = false;
    sel.addEventListener("change", async () => {
      const n = sel.value; sel.value = "";
      if (!n) return;
      try { const r = await fetch("/examples/" + encodeURIComponent(n), { cache: "no-store" }); if (!r.ok) throw new Error(`${n}: ${r.status}`); openText(await r.text(), n); }
      catch (e) { fileNote(e.message, true); }
    });
  }).catch(() => {});
  $("help-form").addEventListener("submit", (e) => {
    e.preventDefault();
    request({ op: "help", topic: $("help-input").value.trim() }, { frame(fr) { if (fr.kind === "help") renderHelp(fr); }, done(fr) { if (fr.status !== "ok") $("help-out").textContent = fr.error ? fr.error.message : fr.status; } });
  });
  $("atlas-go").addEventListener("click", () => {
    const ell = $("atlas-ell").value;
    if (!/^(5|7|11|13)$/.test(ell)) return;
    addCell(`A = atlas(${ell});\nA`).ta.focus();
  });
  $("atlas-info").addEventListener("click", showHeader);
  $("atlas-cells").addEventListener("click", () => showCells(0));
  $("atlas-view-close").addEventListener("click", () => { $("atlas-view").hidden = true; });
  $("atlas-sys-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const sid = uintField("atlas-sid");
    if (sid === undefined || sid === null) { $("atlas-sid").focus(); return; }
    showSystem(sid);
  });
  $("atlas-filter-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const filter = {};
    for (const [key, id] of [["r", "af-r"], ["base", "af-base"], ["twist", "af-twist"], ["n", "af-n"]]) {
      const v = uintField(id);
      if (v === null) { $(id).focus(); return; }
      if (v !== undefined) filter[key] = v;
    }
    const ty = $("af-type").value;
    if (ty === "eisenstein" || ty === "cuspidal") filter.type = ty;
    const co = $("af-coll").value;
    if (co === "true" || co === "false") filter.collected = co === "true";
    findSystems(filter, 0);
  });
  $("token-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const t = $("token-input").value.trim();
    if (!/^[A-Za-z0-9_-]+$/.test(t)) return;
    store("oc_token", t); $("token-input").value = ""; wantReconnect = true; connect();
  });
  document.addEventListener("click", (e) => { if (pop && !pop.contains(e.target)) closePop(); });

  addCell("M = matrix(7, [[1, 2], [3, 4]])\ncharpoly(M)");
  addCell("H = cohomology(symmetric_power(sl2z_model(), 10, 691));\nH.dims;\nH");
  nb.dirty = false; showNbState();
  connect();
})();
