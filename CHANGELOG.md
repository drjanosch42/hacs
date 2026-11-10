<!-- SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Changelog

All notable user-visible changes. The format follows Keep a Changelog; versions follow SemVer (`0.x`: interfaces may still
change).

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
