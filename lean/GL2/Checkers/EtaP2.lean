import GL2.Checkers.TauP2
import GL2.Checkers.TauBdP2

/-!
# L18-3b: the η checker on real C++ output (kernel-checked)

[GT] `chk:eta-p2`.  `decide +kernel` on `GL2/Checkers/PairData/P2R1.lean` (the `HECKE_DUMP_TAU_PAIR=2:1`
dump), no `native_decide`.

* `etaP2_check`, `etaP2_spec`: the dumped `p = 2`, `ν = 1` pair homotopy satisfies `obl:eta` (i)–(iv)
  (`EtaSpec`) -- over the τ-table of `tauP2_check` and the `τ_∂`-table of `tauBdP2_check`, the same model and
  coset datum by construction; the homotopy identity `∂η + η∂ = j τ_∂ − τ j` on all 21 slots.
* `etaP2_fnv`, `etaP2_fnv_golden`: the Lean FNV-1a-64 (walked as `HeckePairHomotopy::table_fnv64`) equals
  the C++'s, and its `HashScalar64` is the golden object `eta_table_fnv64_p2_r1`.
* `etaP2_tampered_rejected`, `etaP2_tampered_off_golden`: one coefficient of the first non-empty edge slot
  negated is rejected and leaves the golden hash.
-/

namespace GL2.TauChk.Data
open GL2.TauChk

/-- The η-specific clauses, one kernel declaration. -/
theorem etaP2_core : etaCoreCheck etaP2R1 = true := by decide +kernel

theorem etaP2_check : etaCheck etaP2R1 = true := etaCheck_of tauP2_check tauBdP2_check etaP2_core

/-- **Clauses (i)–(iv) of `obl:eta` hold on the real `p = 2` η-table.** -/
theorem etaP2_spec : EtaSpec etaP2R1 := etaCheck_sound _ etaP2_check

theorem etaP2_fnv : etaTableFnv etaP2R1 = etaP2R1.fnv := by decide +kernel

/-- The checked table is the golden one: `HashScalar64(fnv) = eta_table_fnv64_p2_r1 = 0xe03a5d76f6921ca3`. -/
theorem etaP2_fnv_golden : hashScalar64 (etaTableFnv etaP2R1) = goldenEtaP2R1 := by decide +kernel

/-- Negative control: the first term of the first non-empty edge slot negated. -/
def etaP2Tampered : EtaData := { etaP2R1 with slots := negFirstOfDeg 1 etaP2R1.slots }

theorem etaP2_tampered_rejected : etaCoreCheck etaP2Tampered = false := by decide +kernel

theorem etaP2_tampered_off_golden : hashScalar64 (etaTableFnv etaP2Tampered) ≠ goldenEtaP2R1 := by
  decide +kernel

end GL2.TauChk.Data
