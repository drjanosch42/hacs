# Vendored: Anthropic FLT, 25 modules (slot 21, L21-V)

**Upstream.** <https://github.com/anthropics/fermats-last-theorem>, commit `6e837e75355538c7f80bab5b956861e86c4eacc2`
(2026-09-24), Lean `v4.33.1`, mathlib `db584cd6d46c92f209a44c0f1c829460d327499d`. Licence: Apache-2.0.
Copyright 2026 Anthropic, PBC; portions from the Imperial College London FLT project (see below and `ATTRIBUTION.md`).

**What.** The import closure of the core set #1 + #2 + #5 of the L20-H port-vs-bump table
(`docs/implementation/slot20/L20-H_flt_harvest.md` §2, §5; owner approval 2026-10-04, slot-21 plan item 4): 25 modules,
2 958 lines. Roots: `Theorems.Thm_HeckeCohomology_heckeH1_{delta0,eq_of_section,natural}`, the six
`Theorems.Thm_groupCohomology_Cores_*`, `Definitions.Def_LocalLanglands_GelfandInvolution` (closure: `HeckePair` (#4),
`AbstractHeckeOperator` (#14)). External imports: `Lean`, `Mathlib.*` only. Per-file sha256, lines, harvest row and
consumer labels: `MANIFEST.tsv`.

**Modifications: none.** Every `.lean` file, `LICENSE`, `NOTICE` and `ATTRIBUTION.md` is byte-identical to upstream
(checked by sha256 against a local clone of the upstream repository at the commit above).
The directory layout below `Vendor/AnthropicFLT/` is upstream's, so module names (`Definitions.*`, `P2M.*`, `Theorems.*`)
and imports are unchanged. Files added by us: `MANIFEST.tsv`, this `PROVENANCE.md`.

**Build.** `lean/lakefile.toml`, `[[lean_lib]] name = "AnthropicFLT"`, `srcDir = "Vendor/AnthropicFLT"`, a default target.
Toolchain stays ours (`leanprover/lean4:v4.33.0`, mathlib `db584cd`); the v4.33.0 → v4.33.1 delta is kernel soundness
fixes only (L20-H §1). The lib carries upstream's package options (its `lakefile.lean`):
`maxHeartbeats 4000000`, `synthInstance.maxHeartbeats 400000`, `backward.isDefEq.respectTransparency.types false`. The
build fails without the last one (`Def_GroupCohomology_Corestriction2.lean:179`, instance synthesis); it is an elaborator
option the kernel never reads, admitted for this lib only by `lean/tools/lean_census.py` (`VENDOR_OPTION_ALLOW`), and every
vendored module is kernel-replayed by `docs/matrix-engine/tools/lean_replay.sh`. A Lake `require` of the upstream
repository was not used: it would fetch the 13M-line tree into the shared `lean-deps/` and needs `lake update`
(forbidden; CLAUDE.md, "Lean lane").

**Third-party notices inside the vendored set.** Of the 25 files, `ATTRIBUTION.md` lists only
`Definitions/Def_AbstractHeckeOperator.lean` (whole file, adapted from Imperial FLT
`FLT/AutomorphicForm/QuaternionAlgebra/HeckeOperators/Abstract.lean`: "Copyright (c) 2025 Kevin Buzzard. All rights
reserved. Released under Apache 2.0 license as described in the file LICENSE. Authors: Kevin Buzzard, Andrew Yang,
Matthew Jasper"). None of the 25 is in its §4 (Mathlib text). The project-level `NOTICE` entry is
`docs/licence/third_party/anthropic_flt.md`.

**Updating.** Re-vendor only by copying from a new upstream commit, recomputing `MANIFEST.tsv`, and stating any edit
(Apache-2.0 §4(b): a modified file must carry a prominent change notice; then `modified=yes` on its row).
