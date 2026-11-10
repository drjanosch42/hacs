# NOTICE entry: vendored Anthropic FLT modules (input for WP-LIC)

Written by slot-21 L21-V for the WP-LIC agent, who folds it into the repository's top-level `NOTICE`. Not the NOTICE itself.

**Proposed NOTICE text:**

```
lean/Vendor/AnthropicFLT/
  25 Lean modules from "Fermat's Last Theorem in Lean 4", https://github.com/anthropics/fermats-last-theorem,
  commit 6e837e75355538c7f80bab5b956861e86c4eacc2. Copyright 2026 Anthropic, PBC.
  Licensed under the Apache License, Version 2.0 (lean/Vendor/AnthropicFLT/LICENSE).
  Upstream's NOTICE and ATTRIBUTION.md are kept unmodified beside the files (lean/Vendor/AnthropicFLT/NOTICE,
  lean/Vendor/AnthropicFLT/ATTRIBUTION.md).
  Definitions/Def_AbstractHeckeOperator.lean is adapted (by Anthropic) from the FLT project, Imperial College London,
  https://github.com/ImperialCollegeLondon/FLT, FLT/AutomorphicForm/QuaternionAlgebra/HeckeOperators/Abstract.lean:
  Copyright (c) 2025 Kevin Buzzard. Released under the Apache License, Version 2.0.
  Authors: Kevin Buzzard, Andrew Yang, Matthew Jasper.
  The files are unmodified; provenance, per-file sha256 and the list of files: lean/Vendor/AnthropicFLT/PROVENANCE.md,
  lean/Vendor/AnthropicFLT/MANIFEST.tsv.
```

**Facts for the inventory**

| Item | Value |
|---|---|
| Path | `lean/Vendor/AnthropicFLT/` (`Definitions/`, `P2M/`, `Theorems/`: 25 `.lean` files, 2 958 lines) |
| Upstream | `anthropics/fermats-last-theorem` @ `6e837e75355538c7f80bab5b956861e86c4eacc2` (2026-09-24) |
| Licence | Apache-2.0; `LICENSE` sha256 `746bc52848eabf8765635fe532e0680756a10b18e7b3fb26e7433ab4edf34034` |
| Upstream `NOTICE` | kept verbatim, sha256 `42b1821f19a6aaa25e6811d22ab8f2fdc6433d4a37bc810b1cfafa5b6dd6b0b6` |
| Upstream `ATTRIBUTION.md` | kept verbatim, sha256 `698fb76c8d80b2ed78ebd48b271f2ee5815b3654fb8ec5c699b21b16ce0d9698` |
| Modified files | none (`MANIFEST.tsv`: `modified=no` on every row) |
| Third-party material inside | one file from Imperial FLT (Apache-2.0), named above; no flt-regular file; no Mathlib-text file (ATTRIBUTION.md §4) |
| SPDX | the vendored files carry no headers upstream and stay byte-identical. **Do not add SPDX headers to them**: an edit would make them modified files under Apache-2.0 §4(b). The SPDX guard should exempt `lean/Vendor/**` or accept `MANIFEST.tsv` as the licence record |
| Build-only dependencies | Lean 4 toolchain, mathlib `db584cd` and its packages: fetched at build time, not distributed |
