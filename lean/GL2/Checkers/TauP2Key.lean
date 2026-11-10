import GL2.Model.KeyTie
import GL2.Checkers.TauP2Data

/-!
# M2-a (L18-1): the real `p = 2` dump is an oriented key model (kernel-checked)

[GT] `chk:tau-p2-key`.  `decide +kernel` on `GL2/Checkers/TauP2Data.lean` (no `native_decide`, no axiom): in each degree
`q = 0, 1, 2` the dumped representatives, transporter lists and vertex correspondences satisfy `TauData.KeyAx`, so
`TauData.toKeyModel` applies and `TauData.canon_eq_canon` identifies the checker's keys with `Ku.orCanon` -- L16-4's
limit 1 is a theorem for the dumped table.
-/

namespace GL2.TauChk.Data
open GL2.TauChk

theorem tauP2_keyModelB0 : tauP2.keyModelB 0 = true := by decide +kernel
theorem tauP2_keyModelB1 : tauP2.keyModelB 1 = true := by decide +kernel
theorem tauP2_keyModelB2 : tauP2.keyModelB 2 = true := by decide +kernel

/-- **`tauP2_keyModelB`**: the `p = 2` dump passes the key-model check in every degree of the model. -/
theorem tauP2_keyModelB : ∀ q : ℤ, 0 ≤ q → q ≤ 2 → tauP2.keyModelB q = true := by
  intro q h0 h2
  obtain rfl | rfl | rfl : q = 0 ∨ q = 1 ∨ q = 2 := by omega
  · exact tauP2_keyModelB0
  · exact tauP2_keyModelB1
  · exact tauP2_keyModelB2

/-- The checker's canonical keys on the dumped table are the key model's oriented canonical keys (degree `q ≤ 2`). -/
theorem tauP2_canon_eq_canon (q : ℕ) (hq : q ≤ 2) (γ : M22) (hγ : γ.det = 1) (σ : tauP2.Vz q) :
    let hK := of_decide_eq_true (tauP2_keyModelB q (by omega) (by omega))
    (tauP2.canon γ q σ.1).1.1 = q ∧
    (tauP2.canon γ q σ.1).1.2.1 = (Ku.orCanon (tauP2.toKeyModel q hK) (M22.toSL γ, σ)).1.1.1 ∧
    M22.lexKeyM (tauP2.canon γ q σ.1).1.2.2 = (Ku.orCanon (tauP2.toKeyModel q hK) (M22.toSL γ, σ)).1.2 ∧
    (tauP2.canon γ q σ.1).2 = ((Ku.orCanon (tauP2.toKeyModel q hK) (M22.toSL γ, σ)).2 : ℤ) :=
  TauData.canon_eq_canon tauP2 q _ γ hγ σ

end GL2.TauChk.Data
