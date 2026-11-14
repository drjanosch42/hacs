# The notebook page (`notebook/page/`)

The browser side of the read-only notebook ([P] `docs/console/00-console-interface-plan.md` §18): served by
`notebook/gateway/oc_gateway.py` under `/` and `/static/`, installed by `cmake --install` (with
`-DOC_BUILD_NOTEBOOK=ON`) into `<prefix>/share/oc-notebook/page/`. No build step, no external network asset; math is
rendered by the vendored KaTeX (`notebook/vendor/katex/`).

| File | Role |
|---|---|
| `index.html` | the page skeleton: header (run controls, theme switcher), cells, side panel |
| `style.css` | all styling; every colour goes through a theme token (`--bg`, `--fg`, …) |
| `themes.js` | the theme presets, the switcher and the theme editor (loaded first, without `defer`) |
| `render.js` | `oc-serve/1` value envelopes → DOM / KaTeX |
| `ocfile.js` | the notebook file format (`# %%` cells): parse, serialise, insert, replace (pure functions) |
| `cellbox.js` | the cell as a code box + a separate output box, state labels, the run queue, the exports |
| `app.js` | cells, the WebSocket session, completion, help, the atlas panel |

## What renders, the LaTeX switch and the text behind a display (slot-24 NB-RENDER R1 + R2)

The binary sends typed JSON and never TeX ([P] §18.5); `render.js` alone turns it into TeX. It typesets
`Int`, `Fp`, `Poly`, `Matrix` (≤ 900 entries; larger ones with entries as a paged 50 × 50 table), `GF`,
`GFElt`, `LocalSystem` and computed `Cohomology` over `F_ell` **and** `Z/ell^k` (`H^q ≅ ⊕ (Z/ℓ^e)^m`), and
`DescentRecord` (`H^q(Γ_0(N), Sym^n Z^2) ≅ Z^r ⊕ ⊕ (Z/p^k)^m`). The member-form types (`HeckeOperator`, `Pairing`,
`MaximalIdeal`, `HeckeAlgebra`, `LoewyProfile`, `ExtensionClass`, `BoundaryCohomology`, `ConeCohomology`,
`DicksonLES`, `DescentRecord`) show a members table plus their calls (`A(q)` with `charpoly(A)`, `character(p)`,
`jordan(p)`, …). Wherever the envelope carries a **render hint** (`Console/serve/render_hints.tsv`) a list is
typeset as what it is: a polynomial over `Z`, `Z/ℓ^k` or `F_ℓ` (in the labelled coefficient order), a q-series
`Σ a_n q^n + O(q^N)`, an abelian group, a Hilbert function, Loewy layers, a Jordan type. A List of Records is one
table. Anything else shows its canonical text.

- **LaTeX on / off** (header checkbox, default on, remembered in this browser only; storage failures are
  ignored): off shows every value as its canonical text (tables stay). The envelopes are kept and re-rendered,
  so values are identical in both states; only the presentation changes.
- **Text behind a display**: every value block keeps its canonical text (`data-text`, the hidden `text` view);
  each typeset node carries its own text in `data-text` / its tooltip. Per value: `text` (show) and `copy`;
  per cell: `Copy text` (the cell's whole output as plain text, in order); header: `Export text` (every cell's
  source and output as a `.txt` download). Outside a secure context (plain `http` on the LAN) the copy falls back
  to a selected text box.
- **Tests**: `notebook/gateway/test_render.mjs` (node + the vendored KaTeX, run by `test_gateway.py`) over the
  frozen envelopes `notebook/gateway/fixtures/envelopes.ndjson`, re-captured from a real `--serve` session by
  `notebook/gateway/capture_fixtures.py --binary <build>/hacs`.

## Code box and output box (slot-24 NB-OUTBOX)

- **Two boxes.** A cell is the editable **code box** (state label, *Run*, *Copy code*, *Delete*) and below it a separate,
  read-only **output box**: a thin left rule, no card, no fill, the same look in text and LaTeX mode. It is hidden until
  the cell has run. The code box holds only code; *Copy code* copies only code.
- **Output copy.** *Copy output* in the output box (visible on hover or focus, always on touch devices, always in the
  tab order) copies the plain text behind every display, with LaTeX on or off. Per-value *text* / *copy* stay.
- **Copy as LaTeX** (slot 27). A value block has *LaTeX* (when the value has a math rendering), the output box *Copy as LaTeX* (the values as
  `\\[ … \\]` formulas), a text cell *Copy as LaTeX* (its source). The TeX is the page's own (`toTeXRaw`), switch-independent; none comes from the binary.
- **Re-run replaces.** The old output is cleared when the new run starts. An error (`parse_error: …`, a failed
  `assert`, a refusal) goes into the output box, whose rule turns red.
- **State label** in the code bar: `[ ]` not run, `[…]` queued, `[*]` running (a spinner; dashed frame while queued),
  `[k]` ran as execution `k`. One cell runs at a time, in submission order; *Run all* queues every cell (a failure drops
  the rest of that batch); *Interrupt* interrupts the running cell and drops the queued ones. The source is read when a
  cell is submitted.
- **Keys.** Shift+Enter runs and moves to the next cell (a new one after the last), Ctrl/Cmd+Enter runs and stays,
  Alt+Enter runs and inserts a cell below, Tab completes.
- **Export.** *Export text* (as before), *Export code* (`notebook_code.oc`: the code only, replayable with `--script`),
  *Export JSON* (`notebook.json`, `oc-notebook-export/1`: per cell `index`, `execution_count`, `status`, `code`, `output`
  as separate fields).
- **Tests.** The cell logic (`cellbox.js`) is plain JS; `notebook/gateway/test_render.mjs` runs it under a minimal fake
  DOM: separate boxes, hidden before the run, re-run replaces, error mark, copy returns plain text with LaTeX on, code
  copy is code only, queue order, exports.

## Confirming a long computation (NBCONFIRM S4; [P] §18.19)

- A call outside its verb's silent cost envelope whose estimate needs a confirmation stops the cell with
  `confirm_required`. A **dialog** shows the call, the crossed bound, the estimate (central and upper edge of the wall time
  and of the memory, the memory limit, threads, model, calibration, extrapolation, low confidence), the notes and the session
  wall limit, with *Run* / *Cancel* (Escape cancels). A **strong warning** has a red banner with its reasons; *Run* is enabled
  only after the checkbox. *Run* clears the cell's output and re-sends the same source with the tokens so far (one per
  expensive call: a cell with two takes two dialogs). The re-send **re-executes** the statements before the call, with the
  bindings the stopped run left: their side effects happen again and earlier confirmed calls run again in full (S4 fix round,
  audit M2). A cell that rebinds what it reads (`k = k + 2; r = descent(1, 60 + k)`) changes the call's arguments on every
  round, so the page asks **at most once per call site per cell run**: asked again at the same site, it ends the cell as
  `not run` and says why (compute the arguments in an earlier cell; one expensive call per cell). One confirmation covers
  every execution of the same call in the cell (a loop), bounded by the session wall limit. *Cancel* ends the
  cell as `not run` (a *Run all* stops there). A hard refusal (memory, C:) is an ordinary error: there is nothing to confirm.
- While a confirmed run is in progress **one live progress line** (`#~ running …: elapsed of ~ estimate (up to ~ …), resident …
  · wall limit …`) is updated from the heartbeat every 10 s; the gateway's raised wall limit is shown once.
- **Kill session.** The interrupt is cooperative: since NBCONFIRM S5 the running call stops at its next cancellation point
  (every statement and call, and the long verbs' loop steps: per Hecke prime, torsion prime, level, stage; [P] §18.6), so
  only a step longer than 10 s or a call with no point inside keeps running. When no `done` arrives within 10 s of *Interrupt*, a red **Kill session** button appears; it asks once and then
  restarts the session (every binding is lost).

## Notebook files: open, insert, save (slot-24 NB-FILE; [P] §18.18)

A notebook is saved and loaded as a plain **`.oc` script with `# %%` cell separators** (the Jupytext / VS Code percent
convention). `#` is an OC comment, so the file is still a script: `hacs --script nb.oc` runs every cell in order.

```
# SPDX-FileCopyrightText: ...        <- the preamble: text before the first "# %%"; kept, not a cell
# %% [md] Why Hecke operators        <- a text cell: not run; every body line starts with '#'
# T_p acts on H^*; its eigenvalues are ...
# %% T_2 on H^1                      <- a code cell; the title is optional
T = hecke(H, 2);
display(T.A(1));
```

- **Cells.** `# %%` (end of line or white space after it) starts a cell; the rest of the line is its **title**, shown in
  the cell bar (editable there). `# %% [md] title` is a **text cell** (rendered as text, *Edit* / *Done*, no *Run*, skipped
  by *Run all*). `#%%`, `# %%%` and an indented `# %%` are not separators; a code line that starts with `# %%` would start
  a new cell on reload.
- **Preamble.** Text before the first separator is the notebook's **preamble** (a license header, a description): it is
  kept and written back first. It is not a cell and is not shown; *Insert file as cell…* inserts a non-empty preamble as
  a first, untitled cell so that nothing is lost.
- **No separator.** A file without any `# %%` is loaded as **one cell** (the whole text, comments included).
- **Canonical form** (what *Save* writes): LF line ends (CRLF and CR are converted, a BOM dropped), blank lines at the
  ends of the preamble and of each cell dropped, one blank line after the preamble, **one blank line between cells**, one
  final newline. Load → save is **byte-identical** for a canonical file, and canonicalising twice changes nothing.
  Text cells are written with `# ` before each line (`#` for an empty one).
- **Outputs are not saved.** Only code, titles and text. *Export JSON* (`oc-notebook-export/1`, which now also carries
  `kind` and `title` per cell) is the way to keep outputs. *Export code* leaves text cells out.
- **Actions** (header): **Open notebook…** replaces every cell and **restarts the session** (queued cells dropped,
  execution count back to 0); with unsaved changes it asks first. **Insert file as cell…** inserts the file's cells below
  the current cell (the last cell you edited), the session untouched. **Save notebook (.oc)** downloads the canonical file
  and clears the unsaved mark (shown as a bullet after the name beside the status). **+ text** adds a text cell. Drag a
  file onto a cell to insert it below that cell; drop it elsewhere to open it. Files are read locally (≤ 1 MiB).
- **Examples…** appears when the gateway offers a directory of example notebooks (`--examples-dir`, default the tour
  `notebook/examples/notebook/`: the `showcase_*.oc` notebooks first, then `01_…` to `10_…` and `tour_all.oc`, the whole tour as one notebook); choosing one opens it as
  the notebook. A text cell with a body line that does not start with `#` is refused with its line number.
- **Restart session** also resets the execution count (as Colab does); cells and their outputs stay.
- **Tests.** `notebook/gateway/test_render.mjs`: parse / serialise round trip, canonicalisation, the edge cases above,
  every tour file byte-identical on a round trip and `tour_all.oc` equal to the ten files' cells, insert as cells, open as
  session, save; `test_gateway.py`: the examples route and its confinement; `Console/scripts/057_s24_notebook_file.oc`: a
  notebook-format file runs as a script.

## Text cells with formulas, reading panels (slot-26 NBMATH A1; [P] 3.15.0-pre.8 §18.18 (e), (g))

- **Typeset text.** With LaTeX on, a text cell renders `$…$` (inline) and `$$…$$` (display, may span lines) through the vendored
  KaTeX (`trust: false`), plus a minimal markdown: a blank line ends a block, `- ` list items, `> ` a statement box, `| a | b |`
  table rows (a row may continue over lines that do not start with `|`; a dash row after the first row makes it the header),
  `**bold**`, `` `code` ``, `\$` a literal dollar. `OCRender.renderMarkdown` builds it from text nodes and KaTeX only (never
  `innerHTML`); it is not a value renderer, so the render coverage gate is unaffected. With LaTeX off the cell shows its
  source verbatim; *Copy*, the exports and *Save* always use the source.
- **Reading panels.** A text cell titled `Reading…` folds (closed by default; open when the notebook is opened from a file
  named `showcase_*.oc`). The examples use it for one two-column table per output: *Mathematics* ↔ *What HACS computed, and
  how* (the verb, the raw field names with a gloss, the contract label, the validation name).
- **Certified orders in decimal.** A String value `certified:0x2b3` (the HeckeZ record's signed hexadecimal) shows as
  `certified: 691` with LaTeX on, the stored form on hover; the canonical text is unchanged.
- **Showcases.** *Examples…* lists the `showcase_*.oc` notebooks first (one theorem each, theorem first), then the tour.
- **Tests.** `test_render.mjs`: the markdown parser (paragraphs, display math over lines, escaped and unmatched dollars,
  lists, statement boxes, tables with continued rows), HTML staying text, KaTeX without links under `trust: false`, the
  switch, the reading fold (and its state across a re-render), the decimal display; and every formula of every example
  text cell under KaTeX with `throwOnError: true` (`texcells=` on the `RENDER_TEST` line). `test_gateway.py`
  `test_showcases`: every showcase runs ok / exit 0 in one served session.

## `display`, compute and show all (slot-24 NB-RENDER R3 + R4)

- **Write cells with `display(...)`**. `print(...)` sends plain text, which the page shows as text. `display(x, y, ...)`
  sends each argument as a value, and the page renders it. It also forces a declared handle first. The tour
  `notebook/examples/notebook/01…10_*.oc` is written this way (`notebook/examples/README.md`).
- **compute**: a value that arrives while still *declared* (e.g. a bare `H` right after `H = cohomology(...)`) has a
  *compute* button. It evaluates `display(<that statement>)` in the session and replaces the block with the result.
  `OCRender.stmtSource` recovers the statement from the frame's `line` / `col`.
- **show all**: a `Matrix` larger than `print_max_entries` arrives shape-only and gets a *show all* button (up to
  100 000 entries, the 1 MiB frame cap). The page reads the current `print_max_entries`, displays the matrix once under
  a cap raised to its size (a paged 50 × 50 table when it is over 900 entries), and then restores the old value.
- **More hints**: `cyclic` (⊕ Z/ℓ^e, the torsion of a `hecke_ring_algebra` block), `eigen` (a character value
  `m.character(p)` in F_ℓ[w]/(π_w) ≅ F_{ℓ^f}) and `lesmap` (each map of a `DicksonLES`, with its source and target).
- **Coverage**: `Console/serve/render_coverage.tsv` records how each type and Record schema renders (`tex`, `hint`,
  `table`, or `allow` with a reason). `ConsoleReadOnlyTagValidation` arm (R) checks it against the registry, the
  verb sources, `render_hints.tsv` and the `case` labels of `render.js`. If you add a TeX case here, mark the type
  `tex` there. If you remove one, the gate fails.

## Themes (slot-22 WP-NBT, owner request FR-1)

**Choosing a theme.** The selector in the header offers *Auto (system)* (the default: light or dark after the
operating system's / browser's `prefers-color-scheme`), *Light*, *Dark*, *High contrast (light)*, *High contrast
(dark)* and *Custom*. The choice applies at once; the page is not reloaded and the session (its bindings, the cells,
a running cell) is untouched.

**Defining your own theme.** *Theme…* opens the editor:

- one colour picker per token (table below); any change switches to *Custom*, seeded from the theme in force, and is
  applied live;
- *name* and *base* (`light` / `dark`: the base decides the browser's native widgets — scroll bars, form controls —
  and supplies every token a theme file leaves out);
- *reset to preset* replaces the custom colours by those of a preset;
- **Contrast (WCAG AA)** lists the WCAG 2.x contrast ratio of every checked pair (table below). A pair below its floor
  is marked, and the header shows a **low contrast (k)** badge (its tooltip lists the pairs). This is a warning, not a
  block: the theme stays applied;
- *export file* downloads the theme in force as `oc-theme-<name>.json`; *import file…* reads one (≤ 64 KiB, read
  locally with `FileReader`, nothing is uploaded); the JSON pane shows the theme in force and *apply JSON* applies an
  edited or pasted one.

**Where it is kept.** In this browser's `localStorage` under the key `oc-notebook-theme` (the selected mode and the
custom theme). It survives a reload, is per browser and per gateway address (origin), and never reaches the gateway
or the binary. Without storage (a private window, storage blocked) everything works, but the choice lasts only until
the tab closes; the editor says so. Clearing the site data restores *Auto*.

### Tokens

| Token | Used for |
|---|---|
| `bg` | page background |
| `fg` | text |
| `muted` | secondary text (labels, logs, the side panel's keys) |
| `line` | rules and borders (decorative: not contrast-checked) |
| `panel` | cells, header, inputs, buttons, the editor |
| `accent` | links, link-like buttons, the atlas view frame |
| `ok` / `err` / `warn` | status, errors and error frames, warnings and the running cell's frame |
| `code` | the cell source panes and the JSON pane |
| `katex` | rendered math (KaTeX inherits it; its rules and radicals use the same colour) |
| `sel` | the background of selected text |
| `focus` | the keyboard focus ring and the focused cell's frame |

### Contrast pairs and floors

Text pairs, floor **4.5 : 1**: `fg` on `bg` / `panel` / `code` / `sel`; `muted` on `bg` / `panel` / `code`; `accent`,
`ok`, `err`, `warn`, `katex` on `bg` and on `panel`. UI pair, floor **3 : 1**: `focus` on `bg` / `panel`. The four
presets meet every floor (checked in Python by `notebook/gateway/test_gateway.py`, class `Themes`).

### The theme file: `oc-notebook-theme/1`

```json
{
  "schema": "oc-notebook-theme/1",
  "name": "night",
  "base": "dark",
  "tokens": {
    "bg": "#20142b", "fg": "#e6e6e8", "muted": "#a99bb5", "line": "#333539", "panel": "#2a1c38",
    "accent": "#ffb86c", "ok": "#6cc684", "err": "#f08a80", "warn": "#e5b45a", "code": "#311f42",
    "katex": "#8be9fd", "sel": "#2f4a78", "focus": "#7ea6f0"
  }
}
```

- `schema` is required and must be exactly `oc-notebook-theme/1`.
- `base` is `light` or `dark` (default `light`); `name` a string (trimmed, at most 64 characters, default `custom`).
- `tokens`: every value must be `#rrggbb` (six hex digits; stored lower-case). A missing token is taken from the base
  preset; an unknown token or top-level key is ignored with a note.
- Anything else — invalid JSON, a wrong schema, a value like `red`, `#fff` or `#000000;}…` — refuses the whole file
  with a message; the theme in force is kept. Values are validated before they reach the page's style, so a theme file
  cannot inject CSS.
- Export writes the canonical form (all thirteen tokens, fixed order), so export → import → export is byte-identical.

Developers: the presets live in the strict-JSON block between the `OC-THEME-DATA` markers of `themes.js` (read by the
Python self-test); the light and dark values must also stand in `style.css`'s `:root` blocks (the no-script and *Auto*
case); a new colour in `style.css` must be a token (`test_gateway.py` refuses a literal colour outside `:root`).
