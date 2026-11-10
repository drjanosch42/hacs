# oc_gateway — the read-only notebook gateway (slot-19 NB-3)

`oc_gateway.py` is a stdlib-only Python 3 script (no pip dependency; x86-64 and aarch64 / Jetson) that serves the
notebook page (`notebook/page/`) and the vendored KaTeX (`notebook/vendor/katex/`, katex 0.19.0, MIT, see its
`PROVENANCE.txt`). It maps **one WebSocket per browser tab** onto **one `hacs --serve` child** speaking
`oc-serve/1` ([P] `docs/console/00-console-interface-plan.md` §18). The binary must be built with
`-DOC_BUILD_NOTEBOOK=ON`.

```bash
python3 notebook/gateway/oc_gateway.py --binary <build>/hacs            # loopback only (default)
python3 notebook/gateway/oc_gateway.py --binary <build>/hacs --bind 192.168.1.20   # reachable from the LAN
# installed (cmake --install): python3 <prefix>/share/oc-notebook/gateway/oc_gateway.py
```

It prints `OC_GATEWAY listening=…`, `OC_GATEWAY token=…` and `OC_GATEWAY url=http://<bind>:<port>/#token=…`; open the
URL (the token is in the URL fragment, which the browser never sends to the server; the page moves it to
`sessionStorage` and strips it from the address bar).

## Exposure rules (owner decision 2026-10-03; [P] §18.12)

* **Bind**: `--bind` takes a literal IP address, default `127.0.0.1`. A public address is **refused at startup**
  (exit 2). The unspecified address (`0.0.0.0` / `::`) is accepted only if every address configured on the host is
  loopback / private / link-local (read from `/proc/net/fib_trie` and `/proc/net/if_inet6`; on a host where they
  cannot be read it is refused) — prefer binding the LAN interface's own address.
* **Peers**: connections are accepted only from loopback, RFC 1918 (`10/8`, `172.16/12`, `192.168/16`),
  `169.254/16`, `fe80::/10`, `fc00::/7` (IPv4-mapped IPv6 peers are unwrapped); every other peer is dropped at
  accept. `100.64/10` (RFC 6598 shared address space) is **not** accepted: it is carrier-grade-NAT and overlay-VPN
  space (e.g. Tailscale), not the local network.
* **Token**: 256 bit (`secrets.token_urlsafe(32)`), new for every gateway process, printed once. Every WebSocket must
  present it — first frame `{"op":"auth","token":…,"tab":…}` or an `Authorization: Bearer` header on the upgrade —
  compared with `hmac.compare_digest`. No child is spawned before. Static files (page, KaTeX) are served without it
  (they hold no secret). A cross-origin WebSocket upgrade (`Origin` ≠ `Host`) is refused with 403.
* **No TLS in v1.** LAN traffic — the token included — is **cleartext**: anyone who can sniff the LAN segment can
  read and replay the token while the gateway runs. Use loopback (or an SSH tunnel to a loopback-bound gateway)
  when that matters. A token holder is trusted: the read-only allow-list is a safety rail, not a sandbox ([P] §18.1.4).

## Sessions

* One tab = one session = one child (`tab` id kept in the tab's `sessionStorage`; a second connection of the same tab
  replaces the first). `--max-sessions` (default 4) bounds the total.
* Child environment = the whitelist of [P] §18.9 only: `PATH`, `HOME`, `LANG`, `LC_ALL`, `TMPDIR` (set to the session's
  directory), `ENGINE_THREADS` (`--threads`), `ENGINE_CUDA_DISABLE=1`, and, if given, `ENGINE_CUDA_DEVICE` /
  `ENGINE_HOST_THREADS`. Nothing of the gateway's own environment (`GOLDEN_*`, `HECKE_*`, …) reaches the child.
  Working directory: a fresh empty directory under `--scratch` (default `~/.cache/oc-notebook`), removed afterwards.
* Read-only enforcement, second line: the gateway reads `read_only_verbs.json` (the JSON twin,
  `docs/latex/cli_manual/`; installed beside the gateway), refuses a session whose `hello.counts` disagree with it,
  and refuses (`gw` `rejected`, kind `forbidden`) a cell that calls a verb with no read-only overload (a lexical scan
  outside strings and `#` comments). The binary's dispatch-time check stays authoritative (it alone sees overloads,
  e.g. `save_state(x, path)` → `forbidden_overload`).
* Resource guard: the child gets `--rss-cap-mib N` (`--rss-cap-mib`, default 4096); the gateway SIGKILLs the child's
  process group at `N + max(64, N/10)` MiB, at the per-request wall limit (`--cell-wall-s`, default 1800) and CPU
  limit (`--cell-cpu-s`, default off), and closes a session idle for `--idle-timeout-s` (default 3600). A session is
  refused when `MemAvailable` < N (rule 19: predict first). After a kill or a `fatal` the page offers *Restart
  session* (a fresh child, a new `hello`; all bindings are lost).
* Interrupt is cooperative ([P] §18.6): it acts at the next statement or served-verb call; use *Restart session* as the
  hard cancel.
* Cost confirmation (NBCONFIRM S4, [P] §18.19): a call outside its silent cost envelope whose estimate needs a
  confirmation ends its cell with `done{status:"confirm_required", confirm:{token, …}}`; the page shows the estimate
  in a dialog and re-sends the same cell with `"confirm":[token, …]`. The gateway forwards `confirm` (≤ 16 tokens of
  1–64 printable ASCII characters) and sets the wall limit of a confirmed cell to max(`--cell-wall-s`, 3 · wall_hi + 60),
  at most `--cell-wall-max-s` (default 21600 = 6 h; 0 = keep `--cell-wall-s`), from the child's own estimate, taken only
  from the `confirmed` member of the child's `progress` frame for the in-flight request (sent at the start of every
  confirmed run, before its work; S4 fix round, audit M1: presenting a token raises nothing), never from a page value; event
  `{"kind":"gw","event":"cell_wall","id","cell_wall_s","reason":"confirmed"}`. The RSS cap is `--rss-cap-mib` only and
  cannot be raised from the page; the child also refuses, before the run, a call whose estimated peak exceeds it. After an
  Interrupt with no `done` for 10 s the page offers *Kill session* (the `restart` op).

## Protocol between page and gateway (`oc-gateway/1`)

Page → gateway: `auth` (first), then `eval` / `complete` / `help` / `describe` / `atlas` (with `id`, one request in flight per
session; a second is `rejected` `busy`), `interrupt`, `restart`. Gateway → page: every `oc-serve/1` frame of the
child **unmodified**, plus the gateway's own messages `{"kind":"gw","event":…}` — `authorized`, `unauthorized`,
`session` (limits, the unserved verbs), `rejected` (`id`, `error`), `cell_wall` (NBCONFIRM S4), `killing`, `exited` (`code`, `reason`:
`rss_cap` / `cell_wall` / `cell_cpu` / `idle` / `fatal_<reason>` / `stream_cut` / `exited`), `refused`, `replaced`.

## The atlas channel (slot-20 NB-4; [P] §18.16)

```bash
python3 notebook/gateway/oc_gateway.py --binary <build>/hacs \
    --atlas-store <atlas-dir>/ell691/hecke_atlas_ell691.json \
    --atlas-expect-fnv 0x73c7210047a81b97
```

With `--atlas-store <store.json>` (a `hecke_atlas/v1` store; the path is fixed here and never comes from the page) a
session's first atlas query starts **one** `hacs --serve-atlas` child for that session (protocol
`oc-serve-atlas/1`): the same inert whitelist environment (`PATH`, `HOME`, `LANG`, `LC_ALL`, `TMPDIR`), its own empty
working directory and process group, `--atlas-rss-cap-mib N` (default 1024; SIGKILL at `N + max(64, N/10)` MiB) and a
per-request wall limit `--atlas-wall-s` (default 60). The child opens the store read-only, recomputes its FNV-1a-64
(refused if it differs from `--atlas-expect-fnv`) and checks the index sidecar `<store.json>.ocidx` (or
`--atlas-index`) field by field against the store; without a sidecar it builds the same index in memory. On Σ(691)
the open takes ≈ 1 s with the store in the page cache.

Page → gateway: `{"op":"atlas","id","query":"info"|"cells"|"cell"|"system"|"systems", …}` with the fields of [P]
§18.16 (e) only (`offset`, `limit`, `n`, `sid`, `filter` with `r`, `type`, `collected`, `base`, `twist`, `n`), their
JSON types checked here; anything else is `gw` `rejected` `protocol`; without `--atlas-store` it is `rejected`
`no_atlas`; one atlas request in flight per session (`busy`). The child's `atlas_hello` / `atlas_reply` /
`atlas_fatal` frames are relayed unmodified; the gateway adds `atlas_exited` and `session.atlas =
{configured, store}`. The sidecar is written once by the owner (`hacs --atlas --index <store.json>`); the
gateway never writes it.

The page's atlas panel shows the store header, the cells (with their occurrences), single systems (eigenvalues in
`F_{ell^r}` as polynomials in `Y` via KaTeX, `chi_{a,b}`, `theta^j ⊗ #base`, occurrences) and filtered, paged system
lists; every result carries the store FNV and a *copy request* button. The in-process `atlas(ell)` cell (`ell ∈ {5, 7,
11, 13}`) is still offered.

## Page themes (slot-22 WP-NBT, owner request FR-1)

The page has a theme switcher in its header (*Auto (system)*, *Light*, *Dark*, *High contrast (light)*, *High contrast
(dark)*, *Custom…*) and a theme editor (*Theme…*): one colour picker per token, live preview, the WCAG contrast of the
checked pairs, export / import of a theme as JSON (`oc-notebook-theme/1`). It is **client-side only**: the theme is a
set of CSS custom properties on the page, kept in the browser's `localStorage`; the gateway serves the static
`themes.js` like every other page file and never sees a theme (no new route, no new verb, no server state, CSP
unchanged). Switching or editing never reloads the page or touches the session. User instructions and the JSON
format: [`notebook/page/README.md`](../page/README.md).

## Example notebooks (slot-24 NB-FILE)

`GET /examples/index.json` lists and `GET /examples/<name>.oc` serves the `.oc` files of **one fixed, flat directory**
(`--examples-dir`; default `notebook/examples/notebook/`, installed to `<prefix>/share/oc-notebook/examples`), read-only,
as `text/plain` with `Content-Security-Policy: default-src 'none'`. A request is served only when its name **equals a name
in the directory listing** (a regular file, not a symlink, <= 256 KiB, name `[A-Za-z0-9][A-Za-z0-9_-]*\.oc`); the URL is never
joined to a filesystem path, so `..`, `%2f`, a second slash, a trailing slash, NUL and other extensions all give 404, and
nothing but the listed names is reachable. Like the other static files it needs no token (the files hold no secret); a
session still needs one. With no such directory the route answers 404 and the page offers no *Examples* list. Format and
page actions: [P] §18.18 and [`notebook/page/README.md`](../page/README.md).

## Self-test

```bash
python3 notebook/gateway/test_gateway.py --binary <build>/hacs --scratch <dir outside the tree> [--scripts all] [-v]
```

Stdlib `unittest` against a real `--serve` binary (≈ 10 s, < 0.5 GB): address rules, refused public bind, server-side
drop of a non-private peer, token required (missing / wrong / non-auth first frame / cross-origin; no child spawned),
static file confinement, child environment = whitelist (the gateway itself runs with `GOLDEN_EMIT=1` etc.), empty
working directory, served-script round trip byte-equal to the `.expected` body (`--scripts all`: the 34 runnable
numbered scripts), unserved verb refused by the gateway and untagged overload refused by the binary, interrupt +
busy, one session per tab, wall-limit kill + restart, RSS cap, read-only-list mismatch refuses the session; and (NB-4) the
atlas query check, the atlas channel end to end on a Σ(13) store the binary builds into the scratch (one `--serve-atlas`
child per session, whitelist environment, empty working directory, the store unchanged), `no_atlas` without a store, and
a bad atlas configuration refused at startup; and (slot-22 WP-NBT) the page themes: `themes.js` / `style.css` served with
the right type and CSP, `themes.js` loaded before `app.js` without `defer`, every preset complete and `#rrggbb` with
the WCAG contrast ratios **computed in Python** at or above the AA floors (`THEME_CONTRAST presets=4 pairs=19
checked=76 below_aa=0` on stderr), no literal colour in `style.css` outside its `:root` blocks (whose light / dark
values must equal the `light` / `dark` presets), and, when `node` is on `PATH`, the pure functions of `themes.js`
(parsing, refusals, contrast, export → import round trip; skipped otherwise). Last
line: `GATEWAY_TEST ran=<k> failures=<f> errors=<e> verdict=OK|FAIL`. It is **not** a registered validation: the
gateway lives outside the C++ build by owner decision, and a validation would make `--all` depend on a Python
interpreter and on spawning a server; run it by hand (or in a close window) on a notebook build.

**Process rule (NBCONFIRM S4 audit L3).** A package that touches `src/hacs/Console/` or `notebook/` builds
its merge candidate with `-DOC_BUILD_NOTEBOOK=ON` and runs `ConsoleServeValidation`, `ConsoleServeAtlasValidation`,
`ConsoleReadOnlyTagValidation` and this self-test before its audit (they `[SKIP]` in a default build; see the catalogue,
"Slot 25 NBCONFIRM S4").
