/-
Portions Copyright (c) 2023 Kim Morrison. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors of the copied code: Kim Morrison, Sebastian Ullrich (LeanChecker.lean of leanprover/lean4, v4.33.0).

NOTICE: the body of `replayFromImports` below is copied verbatim from the Lean 4 toolchain's `LeanChecker.lean`
(Apache-2.0, https://github.com/leanprover/lean4); the exact-module driver around it is this project's own.
SPDX-License-Identifier: Apache-2.0
-/
/-
ReplayExact -- the exact-module kernel replay of the GL2 oleans (slot-18 L18-5, F3 finding 4).

The body of `replayFromImports` is COPIED VERBATIM from the toolchain's own `LeanChecker.lean`
(<toolchain>/src/lean/LeanChecker.lean, leanprover/lean4:v4.33.0, the `leanchecker` binary): the module's own
constants are replayed through the kernel onto the environment of its imports.  What differs from stock `leanchecker`:
  * the stock tool matches modules by NAME PREFIX and starts one parallel task per match, so `leanchecker GL2` loads
    all 102 modules at once and was OOM-killed (rc 137) on the 62 GB box; this runner takes EXACT module names and
    replays them one after the other, in one process (peak = the largest single module);
  * the imports (Lean core, Mathlib) are NOT replayed -- the same scope as stock `leanchecker GL2.X`.
Usage:  lake env lean --run tools/ReplayExact.lean GL2 GL2.Skeleton ...
Prints one `REPLAY_MODULE <m> ok|FAIL` line per module and `REPLAY_SUMMARY modules=<n> failed=<k>`; exit 1 iff failed > 0
(or no module given).  Never reads as success on an exception.
-/
import Lean.CoreM
import Lean.Replay

open Lean

unsafe def replayFromImports (module : Name) : IO Unit := do
  let mFile ← findOLean module
  unless (← mFile.pathExists) do
    throw <| IO.userError s!"object file '{mFile}' of module {module} does not exist"
  -- Load all module data parts (exported, server, private)
  let mut fnames := #[mFile]
  let sFile := OLeanLevel.server.adjustFileName mFile
  if (← sFile.pathExists) then
    fnames := fnames.push sFile
    let pFile := OLeanLevel.private.adjustFileName mFile
    if (← pFile.pathExists) then
      fnames := fnames.push pFile
  let parts ← readModuleDataParts fnames
  if h : parts.size = 0 then throw <| IO.userError "failed to read module data" else
  let (mod, _) := parts[0]
  let (_, s) ← importModulesCore mod.imports |>.run
  let env ← finalizeImport s mod.imports {} 0 false false (isModule := true)
  let mut newConstants := {}
  -- Collect constants from last ("most private") part, which subsumes all prior ones
  for name in parts[parts.size-1].1.constNames, ci in parts[parts.size-1].1.constants do
    newConstants := newConstants.insert name ci
  let env' ← env.replay newConstants
  env'.freeRegions

unsafe def main (args : List String) : IO UInt32 := do
  initSearchPath (← findSysroot)
  if args.isEmpty then
    IO.eprintln "ReplayExact: no module given"
    return 1
  let mut failed := 0
  for a in args do
    let m := a.toName
    try
      if m.isAnonymous then throw <| IO.userError s!"Could not resolve module: {a}"
      replayFromImports m
      IO.println s!"REPLAY_MODULE {a} ok"
    catch e =>
      failed := failed + 1
      IO.println s!"REPLAY_MODULE {a} FAIL {e}"
  IO.println s!"REPLAY_SUMMARY modules={args.length} failed={failed}"
  return (if failed == 0 then 0 else 1)
