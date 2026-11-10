<!-- SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# HACS — Harder Arithmetic Cohomology System

*High-performance simplicial orbifold cohomology implementation for arithmetic groups (C++ / AVX-512 / CUDA).*

**In memoriam Prof. Dr. Günter Harder (1938-2025).**

This project originated in the winter of 2023–2024, when the author was a guest at the Max Planck Institute for Mathematics in Bonn. In our discussions there, Günter Harder made the point that computing the cohomology of SL_2(Z) together with its refined arithmetic structure, i.e. integral coefficient systems, the boundary long exact sequence, the Hecke action and the orbifold phenomena, is an interesting and worthwhile undertaking.

As is well known from his writings, Harder favored a sheaf-theoretic Čech approach to orbifold cohomology. HACS translates that vision into a topological, Moerdijk--Pronk-style simplicial orbifold architecture engineered from the ground up in C++, AVX-512, and CUDA. In certain cases we deviate from that simplicial structure for performance reasons where the tree models offer a faster pathway.

We see the cohomology of an arithmetic group as a comprehensive 'arithmetic' package. This includes in particular sheaf cohomology, cohomology with compact supports, boundary cohomology, and the corresponding long exact sequence of Hecke modules. HACS computes the structure of the acting Hecke algebras, their localizations, congruence phenomena, e.g. the classic Ramanujan $p = 691$ phenomenon.

HACS is built to explore the arithmetic landscape Harder spent his life illuminating. Though he passed away before seeing the system come to fruition, his mathematical insight and encouragement remain at the core of this codebase.

Computing the cohomology of arithmetic groups is hard. This software is Harder.


---

This is release **0.1.0** (`RELEASE_INFO.txt`). The version follows SemVer; `0.x` means that interfaces (CLI tokens,
console verbs, record schemas) may still change. Changes: `CHANGELOG.md`.

## Documentation

Five PDF documents ship with the tree, in `docs/latex/`; their LaTeX sources sit next to them.

- [`docs/latex/phenomena_gallery.pdf`](docs/latex/phenomena_gallery.pdf): the phenomena gallery, worked examples of what HACS computes, with the commands and their real output (start here);
- [`docs/latex/cli_manual.pdf`](docs/latex/cli_manual.pdf): the user manual (command line, console and its scripting language, campaigns);
- [`docs/latex/gl2_contract.pdf`](docs/latex/gl2_contract.pdf): the mathematical specification of the cohomology and Hecke computations for SL_2(Z) with symmetric-power coefficients;
- [`docs/latex/hecke_extensions.pdf`](docs/latex/hecke_extensions.pdf): extensions of Hecke modules in this cohomology, how their classes are classified, and the congruences they reflect;
- [`docs/latex/hida_hecke_algebra.pdf`](docs/latex/hida_hecke_algebra.pdf): Hida theory and ordinary Hecke algebras at tame level one.

## What is in this tree

| Path | Content |
|---|---|
| `src/ModArithmetic/` | modular arithmetic on `uint32_t`, the dense kernels and the block model (with `Binary/`: the F_2 / Z/2^k / Z/3^k rings) |
| `src/Engine/` | the matrix engine's tape, executor and thread pool |
| `src/Cuda/` | the optional CUDA device code (and its CPU stub) |
| `src/hacs/` | the cohomology layer, Hecke operators, Hecke extensions, the integral line over Z, number fields, the console, the campaign driver, and the validation suite |
| `cmake/`, `CMakeLists.txt`, `CMakePresets.json` | the build |
| `docs/latex/` | the mathematical documents and their PDFs: [C] gl2_contract 1.26.3, [E] hecke_extensions 1.25.2, [U] hida_hecke_algebra 0.7.2, [CLI] cli_manual 1.5.13, [PG] phenomena_gallery 1.5.4 |
| `docs/console/console.md` | the generated console reference |
| `lean/` | formal checks in Lean 4 (in progress; **not part of the binding contract**) |
| `notebook/` | the read-only notebook gateway and page (built with `-DOC_BUILD_NOTEBOOK=ON`), and example console scripts |
| `docs/matrix-engine/results/`, `docs/integral_census/`, `docs/implementation/` | reference results and test fixtures read by the validation suite (only the files it reads) |

## Quick start

Out-of-source builds only (the sources are under `src/`); the program is `build/hacs`.

```bash
cmake -B build -S . -DOC_ENABLE_CUDA=OFF          # CPU only; drop the flag for the CUDA build (the default configuration)
cmake --build build -j8
./build/hacs --version
./build/hacs --list          # every validation, with its tier
./build/hacs --all           # the fast + standard tiers; exit code = number of failures
```

`BUILDING.md` has the four supported targets with their exact commands: (a) x86-64 with CUDA
(`-DOC_CUDA_ARCHITECTURES=<cc>`), (b) CPU only, (c) NVIDIA Jetson Orin (`cmake --preset jetson-orin`), (d) Android /
Termux and small aarch64 devices (`cmake --preset termux`), plus the build types and presets. If the CUDA configure
reports `No CMAKE_CUDA_COMPILER could be found` or `Unsupported gpu architecture`, see `BUILDING.md` (a): `nvcc` outside
`PATH` and the minimum CUDA version per GPU architecture (12.8 for the default `120`). `--all` takes about 25 to 55
minutes in one process; `BUILDING.md`, "Expected run times", shows how to shard it. The user manual is `docs/latex/cli_manual.pdf` (console, CLI, campaign driver).

## Validation

There is no external test framework: every check is a named validation of the executable (`--list`, `<Name>`, `--all`,
`--all-full`, `--all --shard i/N`). `docs/matrix-engine/tools/run_all_sharded.sh <binary> [N]` runs `--all` as N parallel
shards. The golden reference (`src/hacs/Golden/GoldenReference.inc`) is checked by
`GoldenRegressionValidation` (`GOLDEN_STRICT=1` gates both tiers). No build switch and no `ENGINE_*` environment variable
changes a computed value.

## Lean

`lean/` is a Lake project (toolchain in `lean/lean-toolchain`, mathlib pinned by `lean/lake-manifest.json`; `lake build`
fetches mathlib from the network). It records formal checks of statements of the contract; it is work in progress and is
not part of the binding contract.

## Data

The census data sets (congruence census, integral census, the integral Hecke structure data set HeckeZ) are published in a
separate companion repository, `hacs-data`, under CC-BY-4.0; this tree holds only the reference results its validations
read. What is pending or not included in this release: `CHANGELOG.md`, "Not in this release".

## Citation

Please cite HACS using the metadata in `CITATION.cff`. Each release is archived on Zenodo, which assigns it a DOI; see the
release page. The data sets carry their own `CITATION.cff` in `hacs-data`.

## Licence

| Part | Licence |
|---|---|
| C++ / CUDA / scripts | LGPL-3.0-only with the NVIDIA CUDA Toolkit exception (`LICENSE`, `LICENSE.GPL`) |
| `lean/` | Apache-2.0 (`lean/LICENSE`); `lean/Vendor/AnthropicFLT/` keeps its upstream Apache-2.0 notices |
| `docs/`, `lean/blueprint/` | CC-BY-4.0 (`docs/LICENSE`) |

Every file's licence is given by an SPDX header or by `REUSE.toml`; texts are in `LICENSES/`. Third parties: `NOTICE`,
`docs/licence/THIRD_PARTY.md`. Check: `docs/licence/tools/check_licences.sh` (also `hacs LicenceGuardValidation`;
it needs a git checkout, so a tarball without `.git` builds but cannot run the licence guard).

## Acknowledgements

The author thanks the Max Planck Institute for Mathematics in Bonn for its hospitality during the winter of 2023–2024.

We acknowledge the use of two RTX PRO 6000 Blackwell Max-Q Workstation Edition GPUs, awarded to us by the NVIDIA Academic Grant Program for the project 'CUDA-GNFS: GPU-Accelerated General Number Field Sieve for Large-Scale Integer Factorization on NVIDIA HPC Platforms'.

Development of HACS was assisted by Anthropic's Claude AI models, which were used under the author's direction for code, documentation and verification work; all computed results are validated by the project's own test suite and, where stated, by formal Lean proofs.

## Author

Fabian Januszewski, `fabian.januszewski@math.uni-paderborn.de`.
