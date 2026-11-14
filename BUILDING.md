<!-- SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Building HACS 0.2.1

## Requirements

| | needed for | version |
|---|---|---|
| CMake | everything | 3.18 or later (3.21 or later for `CMakePresets.json`); tested: 3.22 and 4.4.5 |
| C++17 compiler | everything | tested: GCC 11.4 and GCC 13.3.0 (x86-64 Linux; aarch64 on a Jetson); clang with libc++ (Termux) is supported but not yet built on a device |
| Python 3 | the guard scripts some validations call (standard library only) | 3.8 or later |
| bash, git | the guard scripts; the licence guard needs a git checkout | any recent |
| CUDA Toolkit (`nvcc`) | **only** the optional GPU engine | tested: 13.0 and 12.9 (x86-64, Blackwell, sm_120; 12.9 on a second workstation, 2026-11-10), 12.6 (Jetson); the minimum depends on the GPU architecture, see (a) |
| pdflatex, bibtex | only `make docs` (`-DOC_BUILD_DOCS=ON`) | TeX Live |

No external computer-algebra system and no big-number library is used.

Builds are **out-of-source only**: the sources are under `src/`, and every build goes to its own directory (`-B <dir>`, or
a preset). An in-source `cmake .` is refused. The program is `build/hacs`; the build also creates `cohomology_app`,
a symbolic link to it (the former name). Four targets are supported; the commands below are the whole recipe for each.

## (a) x86-64 Linux with CUDA (the default configuration)

```bash
cmake -B build -S .                                   # RelWithDebInfo, CUDA on, OC_CUDA_ARCHITECTURES=120
cmake -B build -S . -DOC_CUDA_ARCHITECTURES=89        # instead: the compute capability of your GPU (75, 86, 87, 89, 120, ...)
cmake --build build -j8
./build/hacs --version
./build/hacs --all
```

The architecture is fixed at the **first** configure of a build directory; to change it, use a new directory (or remove
its `CMakeCache.txt` and `CMakeFiles/`). Toolchain of record: GCC 11.4, CMake 3.22 or later, CUDA 13.0 (`nvcc`
V13.0.48), an RTX PRO 6000 Blackwell (compute capability 12.0); CUDA 12.9 was also verified on Blackwell.

**`nvcc` not on `PATH`.** If the configure stops with `No CMAKE_CUDA_COMPILER could be found`, CMake did not find
`nvcc`. Name it explicitly (on a fresh build directory), for example for a toolkit in `/usr/local/cuda` or in your home
directory:

```bash
cmake -B build -S . -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc     # or e.g. $HOME/cuda-12.9/bin/nvcc
# or, equivalently, through the environment:
export CUDA_HOME=$HOME/cuda-12.9                      # the toolkit's root directory
export PATH=$CUDA_HOME/bin:$PATH                      # or: export CUDACXX=$CUDA_HOME/bin/nvcc
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}   # if libcudart is not found at run time
cmake -B build -S .
```

This is the same as the `PATH` line of (c). Without a CUDA toolkit at all, use (b).

**Minimum CUDA version per architecture.** `nvcc` must know the architecture you configure for; an older toolkit stops
the build with `nvcc fatal : Unsupported gpu architecture 'compute_<cc>'` (for example CUDA 12.6 with the default `120`).
The minimums below are NVIDIA's (CUDA release notes); this release itself was built only with the toolkits listed under
Requirements.

| `OC_CUDA_ARCHITECTURES` | GPU family | minimum CUDA |
|---|---|---|
| `120` (default) | Blackwell, RTX 50 series and RTX PRO 6000 Blackwell | 12.8 |
| `90` | Hopper (H100, H200) | 11.8 |
| `89` | Ada Lovelace (RTX 40 series, L4, L40) | 11.8 |
| `87` | Jetson Orin | 11.4 |
| `86` | Ampere, consumer and workstation (RTX 30 series, A40) | 11.1 |
| `80` | Ampere, data centre (A100) | 11.0 |
| `75` | Turing (RTX 20 series, T4) | 10.0 |

Your GPU's compute capability is shown by `nvidia-smi --query-gpu=compute_cap --format=csv`; drop the dot (12.0 ->
`120`). The GPU is **never** used by default: device routes are
opt-in (`ENGINE_DEVICE=...`, see the user manual) and bit-identical to the CPU route. Tested at compute capability 12.0 (Blackwell; earlier versions on two machines): with this release, the full suite
(`--all-full`), the strict golden reference and the determinism digests pass. At 7.5 (Turing) an earlier version (0.1.1) builds cleanly on
an sm_75 laptop (2026-11-10); the `--all` suite passed there on an earlier version (2026-10-30).

## (b) CPU only (no CUDA toolkit)

```bash
cmake -B build-cpu -S . -DOC_ENABLE_CUDA=OFF          # or: cmake --preset cpu-only -B build-cpu
cmake --build build-cpu -j8
./build-cpu/hacs --all
```

`OC_ENABLE_CUDA=OFF` never searches for `nvcc` or `libcudart`: the device layer is a CPU stub, and the device validations
print `[SKIP]`. The CPU path is also the path of record for every computed value. Tested on this release (x86-64,
GCC 11.4): `--all` (the fast and standard tiers), the strict golden reference and the determinism digests; the slow tier
has not been run on a CPU-only build.

## (c) NVIDIA Jetson Orin (aarch64, CUDA compute capability 8.7)

On the Jetson (JetPack 6, i.e. L4T r36, Ubuntu 22.04: GCC 11.4, CMake 3.22; JetPack 5 ships CMake 3.16, too old). JetPack's
CUDA toolkit lives in `/usr/local/cuda` and is usually not on `PATH` (or name `nvcc` with `-DCMAKE_CUDA_COMPILER=...`, as in (a)):

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake --preset jetson-orin -B build                   # = cmake -B build -S . -DOC_CUDA_ARCHITECTURES=87
cmake --build build -j3                               # -j3 on a 7 GB board (tested); -j6 needs about 6 GB free (16/32 GB boards)
./build/hacs --version
GOLDEN_STRICT=1 ./build/hacs GoldenRegressionValidation
```

CPU only on the same board: `cmake --preset jetson-orin-cpu -B build-cpu` (no CUDA toolkit used), or the stripped
`aarch64-cpu` preset of (d). Set the power mode first (`sudo nvpmodel -m 0`, `sudo jetson_clocks`) if timings matter.
On aarch64 the build adds `-ffp-contract=off` (no fused multiply-adds; no value depends on it), and the x86 SIMD paths
are compiled out (the scalar paths run). **What was tested:** on 2026-10-18, a Jetson Orin-series board (6 cores, 7 GB
RAM, L4T R36.4.4, GCC 11.4.0, CMake 3.22.1, nvcc 12.6) built the CUDA configuration at compute capability 8.7 (an
earlier version, 237 validations) and passed the strict golden reference and the engine and
Hecke-extension determinism digests, GPU arms included. `--all` and the CPU-only configuration were not run there, and
this release itself has not been built on a Jetson.

## (d) Android (Termux) and other small aarch64 devices

```bash
pkg install clang cmake make python git               # Termux; clang with libc++ is its default compiler
cmake --preset termux -B build                        # CPU only, Release -O2, no -g, section GC + ICF, stripped (~45 MB)
cmake --build build -j2                               # -j2: phones run short of RAM and thermal budget
./build/hacs --version
GOLDEN_STRICT=1 ./build/hacs GoldenRegressionValidation
```

`aarch64-cpu` is the same preset for an aarch64 Linux host. Why not `cpu-only`: its `RelWithDebInfo` binary is about
800 MB, almost all debug information; `termux` keeps the same `-O2` code and drops the rest (`OC_STRIP_BINARY`,
`OC_SMALL_BINARY`). By hand: `cmake -B build -S . -DOC_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG" -DOC_STRIP_BINARY=ON`, or any build followed by
`cmake --install build --prefix ~/hacs --strip`. A full `--all` (peak about 4.3 GiB) is too heavy for most phones:
run single named validations (the `fast` tier of `--list`, the golden reference, the digests) and limit threads with
`ENGINE_THREADS=2`; under 6 GB of RAM the OS may kill the process silently. **What was tested:** with an earlier version (0.1.x), the
`termux` and `aarch64-cpu` presets configure on x86-64 Linux, and the `termux` preset builds there with GCC 11.4 (a
stripped 39 MB binary that passes the strict golden reference and the determinism digests). The source was checked for
glibc-only constructs, but **no build on an aarch64 device or with Termux's clang has been run yet**. Reports are
welcome.

## Build types

Pass `-DCMAKE_BUILD_TYPE=<type>` on a **fresh** configure (an existing `CMakeCache.txt` in the build directory pins the first choice):

| type | flags | use |
|---|---|---|
| `RelWithDebInfo` (default) | `-O2 -g -DNDEBUG` | normal use; the binary is large (~800 MB) because of the debug info |
| `Release` | `-O3 -DNDEBUG` | |
| `Debug` | `-O0 -g` | |
| `Bench` | `-O3 -march=native -DNDEBUG` | timing runs on the build machine only |
| `Check` | `-O1 -g -fsanitize=address,undefined` | sanitizer runs; set `ENGINE_CUDA_DISABLE=1` for device-probing validations |
| `CheckThread` | `-fsanitize=thread` | data-race checks (`ENGINE_CUDA_DISABLE=1 setarch x86_64 -R ./build/hacs <Name>`) |

All targets build with `-Wall -Wextra -ffp-contract=off`; the floating-point contraction is switched off so that the
certified floating-point filters behave identically on every platform.

## Presets (`cmake --preset <name> -B <dir>`)

| preset | what |
|---|---|
| `minimal` | the defaults of a plain configure |
| `cpu-only` | `OC_ENABLE_CUDA=OFF` |
| `reference` | CUDA sm_120, the LaTeX `docs` targets and the notebook gateway |
| `jetson-orin`, `jetson-orin-cpu` | NVIDIA Jetson Orin (aarch64): CUDA at compute capability 8.7, or CPU only (target (c)) |
| `termux`, `aarch64-cpu` | small aarch64 devices (Android / Termux, aarch64 Linux): CPU only, `Release -O2`, no debug info, section GC and strip (about 45 MB; target (d)) |

## Platforms

- **x86-64 Linux** (reference): targets (a) and (b). Windows users: build inside WSL 2.
- **NVIDIA Jetson Orin**: target (c), the `jetson-orin` / `jetson-orin-cpu` presets.
- **aarch64 Linux / Android (Termux)**: target (d), the `termux` / `aarch64-cpu` presets; the CMake files add
  `-ffp-contract=off` on aarch64 (where the compiler would otherwise fuse multiply-adds).
- The code base uses `unsigned __int128` where the compiler offers it; `NumberField` has a portable fallback
  (`-DNF_NO_INT128`).

## Optional parts

- `-DOC_BUILD_DOCS=ON`: `make docs` rebuilds the PDFs under `docs/latex/` (pdflatex, bibtex).
- `-DOC_BUILD_NOTEBOOK=ON`: `hacs --serve` and the notebook gateway `notebook/gateway/oc_gateway.py` (Python 3, standard
  library); self-test `python3 notebook/gateway/test_gateway.py`.
- `HECKE_PURE_INTEGER=ON`: disables the certified floating-point fast path of the Hecke sign predicates (results are
  bit-identical either way).

## Validating a build

```bash
./build/hacs --list                                    # names and tiers
./build/hacs --all                                     # fast + standard tiers, exit code = failures
docs/matrix-engine/tools/run_all_sharded.sh ./build/hacs 4 --logs <dir>   # the same in 4 parallel shards
./build/hacs --all-full                                # every tier, incl. the slow one (hours)
GOLDEN_STRICT=1 ./build/hacs GoldenRegressionValidation
```

Peak memory (resident, measured on the reference machine): `--all` in one process about 4.3 GiB; as 4 shards
(`run_all_sharded.sh … 4`) each shard below 3 GiB (plan 4 × 3 GiB); `--all-full` in one process about 5.1 GiB (the
slow tier). Plan at least 6 GiB of free memory per process. `ENGINE_THREADS=<k>` limits the worker threads.

## Expected run times

Wall-clock times of the default build (`RelWithDebInfo`, CPU route) on two 16-core x86-64 machines; your times scale
with the core count and clock:

| run | reference machine (16 cores / 32 threads) | a second 16-core workstation (clean clone) |
|---|---|---|
| `--all`, one process | 24 min | 54 min |
| `--all`, 4 shards (`run_all_sharded.sh … 4`) | 7 min 46 s | not measured (one shard of the four, run alone: 12 min 54 s) |
| `--all`, CPU-only build (`-DOC_ENABLE_CUDA=OFF`), one process | not measured | 51 min |
| `--all-full`, one process | about 1 h 45 min (6346 s) | 3 h 41 min |
| `GOLDEN_STRICT=1 … GoldenRegressionValidation` | about 30 s | about 1 min |
| the four determinism checks (`EngineDeterminismValidation`, `HeckeExtDeterminismValidation`, `ConsoleDeterminismValidation`, `HeckeBenchmarkSmoke`) | about 1 min | about 2 min |

Many validations use few threads, so **shard `--all`**: `--all --shard i/N` (1-based) runs the i-th of N disjoint,
time-balanced parts, and the N parts together run exactly `--all`. `docs/matrix-engine/tools/run_all_sharded.sh
./build/hacs 4` starts all four at once and prints one merged `SHARDED_ALL … verdict=OK|FAIL` line. By hand:

```bash
for i in 1 2 3 4; do ./build/hacs --all --shard $i/4 > shard_$i.log 2>&1 & done; wait
tail -n 3 shard_*.log                                        # each shard's summary; exit code = its failures
```

Four shards need about 4 × 3 GiB of memory (above). For a first check of a new build, the strict golden reference and the
four determinism checks (a few minutes) compare the computed values with the recorded ones; `--all` adds the rest of the
suite.
