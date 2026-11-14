<!-- SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Changelog

All notable user-visible changes. The format follows Keep a Changelog; versions follow SemVer (`0.x`: interfaces may still
change).

## [0.2.1] - 2026-11-14

Feature additions that change no computed value of the defaults: new HeckeZ routes and a faster untrusted factor
finder, each with its own validation; the golden reference (18 385 entries, tier A 12 154, tier B 6 231) and the four
determinism digests are unchanged. The default order of the theta search stays `lex`, so the settings text and the record
ids of HeckeZ editions 1-3 are reproduced byte for byte. (Version 0.3.0 is reserved for the change of that default and
HeckeZ edition 4.) The specification is [C] 1.28.0, which states these routes.

### Mathematics
- HeckeZ, Eisenstein U_p systems with a nontrivial character psi (HZ-EISUP (b); [C] 1.28.0, `prop:heckeZeisupchar`): the
  Eisenstein U_p eigen-systems at level N > 1 are now also computed and certified over Z when the Eisenstein part
  carries a character psi != 1 (theta polynomial, the U_p characteristic polynomials, the conductor and order of psi).
- HeckeZ, the free-block read beyond the word ring of the export (EXW): where a record's export read would stop at a
  word-size or torsion-precision limit, L / l^K is read on the proven free block over Z instead (rule (alpha): only
  where the pending word is written), for l < 2^63; records of edition 3 that were `refused(export_word)` or
  `refused(k1_route)` can now become final when recomputed.
- HeckeZ, the theta search: a new settings value `theta_order` (`lex` | `lex_then_greedy`); `lex_then_greedy` adds a
  greedy per-operator continuation where the lexicographic search ends without a primitive theta. The default is `lex`.
  The g_P "is a bound" predicate is now checked at every consumer (no record of editions 1-3 is concerned).
- The l-adic lattice at the Ramanujan congruence: `HeckeLadicCellValidation` asserts the index-691 lattice
  (v_691(det) = 1).

### Software
- New campaign spec key `hz_theta_order` (`lex` / `lex_then_greedy`, `structure = hecke` only).
- The untrusted big-integer factor finder multiplies modulo n by an NTT product with Barrett reduction from
  `ENGINE_BIGNTT_MIN_LIMBS` limbs on (default 1024), on a per-worker pool; a route knob that cannot change a value.
- The rollback arms (`=0`) of the eleven `HECKE_DESCENT_*` levers, `HECKE_FIELD_DIET=0` and `ENGINE_VNNI=0` / `off` are
  retired: a default build refuses them (exit 1), a build with `OC_ENABLE_LEGACY=ON` still reads them. The default of
  `ENGINE_VNNI2_FLOOR` is now 2^20 (a speed choice).
- `ENGINE_HOST_ELIM_HOOK` prints an `EFFECTIVE` check line after a campaign that used it.
- New validations `HeckeZCrossedHomOracleValidation`, `HeckeZThetaSearchValidation`, `HeckeZExportWordValidation`,
  `HeckeZEisUpCharValidation`, `BigIntNttValidation` (standard tier): 294 validations (75 fast, 171 standard, 48 slow;
  `--all` runs 246).
- Notebook (`--serve`, `OC_BUILD_NOTEBOOK=ON`): TeX and minimal markdown in text cells, folding reading panels,
  certified orders in decimal, "Copy as LaTeX" per value; the ten tour notebooks rewritten (statement in TeX, one
  assert per stated fact) and two showcase notebooks (Ramanujan 691 over Z; cohomology and the boundary exact
  sequence); a console example `ramanujan_691_over_Z.oc`.

### Data
- No new data release: `hacs-data` stays at 0.2.0 (HeckeZ edition 3). Edition 3 was computed with the 0.2.0 routes;
  recomputing its pending records with this release can give final records (see Mathematics).

### Documents
- [C] 1.28.0 (signed): the order of the theta search, the Eisenstein U_p systems with characters, the free-block reads,
  and a note that the retired environment rollbacks are live only in `OC_ENABLE_LEGACY=ON` builds.
- [PG] 1.5.5: the certified Ramanujan congruence over Z at (1, 10) (C_T = Z/691) and its level-11 companion; [CLI] 1.5.15.
- [C] gl2_contract 1.28.0, [E] hecke_extensions 1.25.3, [U] hida_hecke_algebra 0.8.1, [CLI] cli_manual 1.5.15, [PG] phenomena_gallery 1.5.5

## [0.2.0] - 2026-11-11

Two additions to the mathematics, each with its own validation; the golden reference (18 385 entries, tier A 12 154, tier B
6 231) and the four determinism digests are unchanged.

### Mathematics
- HeckeZ, the Eisenstein U_p systems over Z at level N > 1 (HZ-EISUP; [C] 1.27.0, `prop:heckeZeisup`): on the full
  level-N records whose anemic Eisenstein count is one, the U_p eigen-systems of the Eisenstein part are computed and
  certified over Z (CRT under explicit bounds), so these records no longer stop at `blocked(U_p)`. Of the 67 records of
  HeckeZ edition 2 that were `blocked(U_p)`, 64 become final; 3 full level-N records stay pending (N = 167, 173, 197).
- Hida theory, H2 ([U] 0.8.1, part 7, `pr:u-H2`; part 7 was signed in 0.8.0): the discriminant series D_F on the eleven
  ordinary blocks that are ramified (non-reduced mod l), in ten cells; ten are of case (A), and (k, l) = (36, 181) is of
  case (B) with lambda = 2, ramified at every weight of its family.

### Software
- New validations `HeckeZEisUpValidation` and `HeckeHidaDiscLambdaValidation` (standard tier): 289 validations (75 fast,
  166 standard, 48 slow; `--all` runs 241).

### Data
- The companion data release is `hacs-data` 0.2.0 (HeckeZ edition 3, computed with this release's Eisenstein U_p route).

### Documents
- [C] gl2_contract 1.27.0, [E] hecke_extensions 1.25.3, [U] hida_hecke_algebra 0.8.1, [CLI] cli_manual 1.5.13, [PG] phenomena_gallery 1.5.4

## [0.1.1] - 2026-11-10

Citation metadata and title pages; no change of code behaviour or of any computed result (the golden reference, the
validation suite and its digests are unchanged).

### Citation
- Zenodo DOIs: the concept DOI 10.5281/zenodo.24118841 (all versions) in `CITATION.cff` (`doi`, `identifiers`, together
  with the 0.1.0 version DOI 10.5281/zenodo.24118842) and as a badge in `README.md`; `date-released` added.
- The companion data set `hacs-data` is cited with its concept DOI 10.5281/zenodo.24118877 (`CITATION.cff`
  `references`, `README.md`).

### Documents
- The title pages of the specification ([C]), the extensions document ([E]) and the Hida document ([U]) give only the
  version, the date and a pointer to the revision history; the revision histories record the change. No mathematical
  content changes.
- [C] gl2_contract 1.26.4, [E] hecke_extensions 1.25.3, [U] hida_hecke_algebra 0.7.3, [CLI] cli_manual 1.5.13, [PG] phenomena_gallery 1.5.4

## [0.1.0] - 2026-11-10

First public release.

### Mathematics
- Equivariant (orbifold) cohomology of SL(2,Z) and GL(2) with coefficients in Sym^n of the standard representation, over
  F_p and Z/l^k, on a simplicial orbifold model; the central element -I and odd weights handled exactly.
- Hecke operators T_p, T_{p^r}, U_l and the diamond operators; level Gamma_0(N) via Shapiro's lemma, nebentypus.
- Hecke extensions: the cone (boundary) long exact sequence as Hecke modules, Ext computations, Loewy structure, the
  Eisenstein congruences (e.g. 691), Ribet's construction, Kummer and Galois-cohomology checks.
- Hida theory: ordinary projectors, towers over weights mod (l-1)l^(m-1), Lambda-adic computations.
- The integral line over Z: torsion, descent certificates, Hecke structure over Z (HeckeZ records).
- In-house number fields (big integers, cyclotomic fields, LLL, polynomial reduction); no external CAS.

### Software
- One executable (`hacs`) with a console (`--console`, OC scripts), a campaign driver, benchmarks and a validation
  suite of named checks; a golden reference of computed values.
- CPU engine of record; optional CUDA device routes, bit-identical to the CPU route.
- Optional read-only notebook gateway.
- Lean 4 formal checks (not binding).

### Documents
- [C] gl2_contract 1.26.3, [E] hecke_extensions 1.25.2, [U] hida_hecke_algebra 0.7.2, [CLI] cli_manual 1.5.13, [PG] phenomena_gallery 1.5.4

### Data
- The census data sets are **not** in this repository: they are published separately as the companion data repository
  `hacs-data` (CC-BY-4.0): the congruence census edition 4.0, the integral census edition 1, and the integral Hecke
  structure data set (HeckeZ, schema `integral_hecke_dataset/v1`) editions 1 and 2. This tree keeps only the reference
  results and fixtures its validations read (`docs/matrix-engine/results/`, `docs/integral_census/`, `docs/implementation/`).
- The golden reference (`src/hacs/Golden/GoldenReference.inc`) holds 18 385 entries (tier A 12 154, tier B 6 231).

### Not in this release / known limitations
- HeckeZ edition 2: of its 226 records, 140 are final and **86 are pending** (`blocked(U_p)` 67: the Eisenstein U_p
  systems, to be completed in a later release; `refused(export_word)` 20; `refused(k1_route)` 2; three records carry two
  pending words), and 5 records in 4 cells were not computed (`not_computed.tsv` of the data set).
- The census explorer (an interactive browser for the census data) is not part of 0.1.0; it comes in a later release.
- The b2 and L23 addenda of the congruence census are unsigned and are not published.
- `lean/` is work in progress and not part of the binding contract.
