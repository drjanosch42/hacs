<!-- SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Third-party inventory (HACS 0.2.1)

Own licences: code `LGPL-3.0-only WITH LicenseRef-CUDA-exception`; `lean/` Apache-2.0; `docs/` and `lean/blueprint/` CC-BY-4.0
(`COPYRIGHT.md`, `REUSE.toml`, `LICENSES/`). The `NOTICE` file at the root carries the attributions required by the licences below.
Guard: `docs/licence/tools/check_licences.sh` (run by the validation `LicenceGuardValidation`; it needs a git checkout).

## 1. In this source tree (distributed)

| Item | Version / source | Licence | Where | Obligations |
|---|---|---|---|---|
| Vendored FLT modules (25 `.lean`) | `anthropics/fermats-last-theorem` @ `6e837e75`, 2026-09-24 | Apache-2.0, Copyright 2026 Anthropic, PBC | `lean/Vendor/AnthropicFLT/` | `LICENSE`, `NOTICE`, `ATTRIBUTION.md` kept unchanged; files byte-identical (no modification notice needed), hence **no SPDX headers**; recorded in `MANIFEST.tsv`. Detail: `docs/licence/third_party/anthropic_flt.md` |
| `Def_AbstractHeckeOperator.lean` (inside the above) | adapted by Anthropic from Imperial FLT `FLT/AutomorphicForm/QuaternionAlgebra/HeckeOperators/Abstract.lean` | Apache-2.0, (c) 2025 Kevin Buzzard (authors Buzzard, Yang, Jasper) | `lean/Vendor/AnthropicFLT/Definitions/` | as above; named in `NOTICE` |
| Upstream attribution of Imperial FLT / flt-regular / Mathlib text in the vendored files | per `ATTRIBUTION.md` | Apache-2.0 | same | `ATTRIBUTION.md` kept |
| `ReplayExact.lean` (body of `replayFromImports`) | copied verbatim from the Lean 4 toolchain `LeanChecker.lean`, v4.33.0 | Apache-2.0 (Lean FRO / Microsoft Corp.) | `lean/tools/ReplayExact.lean` | Apache-2.0 like `lean/`; named in `NOTICE`; the file's comment states the origin |
| KaTeX 0.19.0: `katex.min.js`, `katex.min.css`, `LICENSE` | npm `katex` (`PROVENANCE.txt`, `SHA256SUMS`) | MIT, (c) 2013-2020 Khan Academy and other contributors | `notebook/vendor/katex/` | MIT notice kept; in `NOTICE` |
| KaTeX fonts (20 `.woff2`) | same package | SIL OFL 1.1 (Design Science, Inc. 2009-2010; Khan Academy 2014-2018; Reserved Font Names `KaTeX_*`) | `notebook/vendor/katex/fonts/` | notice and licence text accompany the fonts (`FONTS-LICENSE-OFL.txt`, `LICENSES/OFL-1.1.txt`, `NOTICE`); fonts unmodified; no sale in isolation |

No third-party document (paper, book, PDF) is part of this tree; the PDFs under `docs/latex/` are this project's own documents,
built from the `.tex` beside them.

## 2. Not in this tree (build- or run-time dependencies)

| Item | Version | Licence | Used for | Obligations |
|---|---|---|---|---|
| Lean 4 toolchain | `leanprover/lean4:v4.33.0` | Apache-2.0 | `lean/` build | none (not distributed) |
| Mathlib and its Lake packages (Batteries, Aesop, Qq, ProofWidgets, ImportGraph, LeanSearchClient, Plausible, Cli) | pinned by `lean/lake-manifest.json` | Apache-2.0 | `lean/` build, fetched by Lake | none while only fetched |
| NVIDIA CUDA Toolkit (`nvcc`, `libcudart`) | 13.0 or later | proprietary, NVIDIA CUDA Toolkit EULA | the optional GPU engine (`EngineCuda`); the binary links `CUDA::cudart` | not distributed here. LGPL linking is allowed by the CUDA exception in `LICENSE`. A **binary** distribution that ships `libcudart` must follow the EULA's redistribution terms; the CPU-only build (`-DOC_ENABLE_CUDA=OFF`) has no NVIDIA dependency |
| CMake, GCC / clang, GNU make, pdflatex / bibtex, Python 3 (standard library only), bash | host | their own | build, documents, tools | none |

## 3. Code and ideas taken from papers and books

- `NumberField/Lll.{h,cpp}`, `Polred.h`, `Hecke/HeckeNebentypus.cpp`: H. Cohen, *A Course in Computational Algebraic Number Theory*,
  GTM 138, Algorithms 2.2.9, 2.6.3, 2.6.7, 4.4.11, 4.4.12; `NumberField/BigInt.h`: Knuth's Algorithm D. Cited as sources of
  **algorithms**, implemented independently (no text copied).
- `lean/tools/ReplayExact.lean`: verbatim from Lean's `LeanChecker.lean` (Apache-2.0), see section 1.
- No GMP / PARI / Sage / Magma / NTL / FLINT code anywhere.

## 4. Bibliography, figures, data

- `docs/latex/*_refs.bib`: bibliographic metadata of published works; CC-BY-4.0.
- `docs/latex` has no external images.
- Data: the reference results under `docs/matrix-engine/results/` and `docs/integral_census/results/`, the test fixtures, and
  `Golden/*.inc` are computed by this code: CC-BY-4.0 (data) / LGPL (generated source).
