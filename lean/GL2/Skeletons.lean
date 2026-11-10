import GL2.Conventions.SymPowMatrix
import GL2.Conventions.SymPowDet
import GL2.Conventions.SymDual
import GL2.Literature.EichlerShimuraLevelOne
import GL2.Saturation.Basic
import GL2.Saturation.LevelOne
import GL2.Cone.Complex
import GL2.Cone.Pair

/-!
# L16-6's modules (slot 16)

One import line in `GL2.lean` reaches everything L16-6 added, so that the root file is touched once:

* `GL2.Conventions.SymPowMatrix`, `GL2.Conventions.SymPowDet`, `GL2.Conventions.SymDual`: L16-2's three open items
  (`lem:detsym`, the `prop:symconv`(3) bound, `lem:symdual`), now proved;
* `GL2.Literature.EichlerShimuraLevelOne`: the two literature hypotheses G10 rests on;
* `GL2.Saturation.*`: the G10 statement skeleton ([C] `sec:saturation`);
* `GL2.Cone.*`: the G3 statement skeleton ([C] `sec:cone`, `cl:les`).
-/
