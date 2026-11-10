# Contributing

- By contributing you agree that your contribution is licensed as the part of the tree it lands in (see `README.md`, `COPYRIGHT.md`): LGPL-3.0-only with the CUDA exception for code,
  Apache-2.0 under `lean/`, CC-BY-4.0 under `docs/`.
- Every new source file needs an SPDX header (`python3 docs/licence/tools/add_spdx.py --apply`), or must be covered by `REUSE.toml`; `docs/licence/tools/check_licences.sh` must pass.
- Do not add files under `lean/Vendor/**`, `notebook/vendor/**` by hand; vendored files stay byte-identical (provenance files beside them).
- Do not add third-party papers, books or PDFs; cite them in the `.bib` files instead. New third-party code or data goes into `NOTICE` and `docs/licence/THIRD_PARTY.md` in the same commit.
- Build: `BUILDING.md` (build types, presets, CUDA, platform notes).
- Validation: `README.md`, section "Validation". There is no test framework: each check is a named validation of the
  executable (`hacs --list`, `hacs <Name>`, `hacs --all`); a change must keep `--all` green and the golden reference
  (`GOLDEN_STRICT=1 hacs GoldenRegressionValidation`) unchanged unless the change is an intended, explained re-baseline.
- Never set an environment hook to make a validation pass. The emitters (`GOLDEN_EMIT`, `CONSOLE_EMIT_EXPECTED`, the
  `HECKE_DUMP_*` / `CLI_MANUAL_EMIT` dumps) are no-ops unless their variable is set; regenerating a reference file
  (`GoldenReference.inc`, a console `.expected` transcript) is a re-baseline that must explain every changed line.
