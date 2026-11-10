<!-- SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Examples (`notebook/examples/`)

Two kinds of snippet live here.

- **The notebook tour** (`notebook/`, files `01_…` to `10_…`). It is a short course, ordered by the mathematics: each step uses
  only what the earlier steps introduced. Every quantity is shown with `display(...)`, so in the notebook it arrives as a
  *value* and is rendered: TeX for groups, polynomials, matrices and characters, and tables for records. In a script the
  same file prints the canonical text. Each tour file is a **notebook file** ([P] §18.18): `# %%` lines separate titled cells (the SPDX header is the preamble, the step's prose a text cell), so *Open notebook…* (or the *Examples…* list) loads it as a multi-cell notebook; `tour_all.oc` is the whole tour as one notebook. A `# %%` line is a comment, so every file still runs with `hacs --script <file>`, with the same transcript.
- **The phenomena-gallery snippets** (the 21 `*.oc` files in this directory). Each is one worked phenomenon of [PG]
  (`docs/latex/phenomena_gallery.tex`), which typesets the file and its transcript verbatim. `gen_phenomena_outputs.sh --check`
  guards both. These snippets report through `print(...)`, so the notebook shows their output as text. Moving them to `display`
  is a separate follow-up, because it re-baselines the gallery's transcripts.

## The course, and where each gallery phenomenon fits

| Step | Tour file | What it introduces | Gallery snippets that build on it |
|---|---|---|---|
| 1 | `01_model_and_local_system.oc` | the model of SL_2(Z)\H and the coefficients Sym^n F_ell^2 | – |
| 2 | `02_cohomology.oc` | H^q as a lazy handle; Eichler–Shimura; small primes and Dickson degrees | `cohomology_dimensions`, `forms_vs_cohomology_24_7`, `level_n_old_forms_2_7`, `nebentypus_7_13` |
| 3 | `03_hecke_operators.oc` | T_p as matrices, characteristic polynomials | `dihedral_wilton_10_23`, `pow_ordinary_rank_13` |
| 4 | `04_eisenstein_ideal.oc` | maximal ideals, their characters T_p ↦ F_{ell^f}, the local algebra T_m | `ramanujan_10_691`, `kummer_ribet_30_37`, `kummer_unit_37_59`, `eisenstein_sign_cone_10_691` |
| 5 | `05_module_structure.oc` | H_m over T_m: Loewy and socle layers, Jordan types | `visible_congruence_48_7`, `multiplicity_two_34_5`, `loewy4_258_37` |
| 6 | `06_dickson_les.oc` | the Dickson long exact sequence and its five maps | `dickson_theta_16_7` |
| 7 | `07_ring_hecke_algebra.oc` | the Hecke algebra over Z/ell^K, lattices with torsion | `hidden_congruence_300_23`, `hida_control_13`, `hida_tower_13_10`, `hida_lambda_11_12` |
| 8 | `08_integral_descent.oc` | H^q over Z as abelian groups, char_Z(T_p) and its factorisation over Z | `descent_torsion_beta`, `descent_congruence_primes` |
| 9 | `09_integral_hecke_module.oc` | T_p over Z as [[tau, M], [0, F]] on H^1 = T (+) Z^r: F exact in a certified basis, torsion included (`module_Z()`, slot-25 EXPORT-Z step 2) | – |
| 10 | `10_congruences_and_extensions_over_Z.oc` | the certified record `hecke_structure`: C_T = 691 at level 1 (Ramanujan), torsion-by-free extensions and what `non_split` means; why the block M of step 9 does not decide | `nonsplit_torsion_16_2` |

**Any order, one session.** A setting lasts for the whole session (a notebook keeps one session across its cells), so
every tour file states the mathematical setting it needs: `01`–`07` begin with `set coefficient_ring = "F_ell";` (the
default), and `08`, `09`, `10` begin with `set coefficient_ring = "Z";`. No file restores the setting at its end, because the next file states its own. The files therefore run in any order, one
after another, in one session or one script. `reset_settings()`, or the page's restart, also returns to the defaults.

The cost of each tour file is below a few seconds and 0.5 GiB. Every verb a tour file uses is `read_only`, so every file runs in a
served session.
