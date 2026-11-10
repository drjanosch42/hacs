import GL2.MP.ACBridge
import GL2.Model.Acyclic

/-!
# MP-L2, open item 1 (slot 22): (AC) for `M₇`, and the unconditional comparison for `SL₂(ℤ)` on `M₇`

[GT] `gt_part_MP.tex`, rows `thm:mp-model7-ac-checks`, `thm:mp-model7-ac`, `thm:mp-model7-comparison-unc`, backing [C]
`thm:mpcomparison` (5) for the row `M₇`.

* `model7_bridge_checks`: the five finite checks of `GL2.MP.ACBridge` hold between the dump `tauP2` (which defines
  `GL2.Model.model7`) and the MP descriptor `model7MP` computed from the same dump (`decide +kernel`).
* `model7_AC`: **(AC) for `M₇` over every commutative ring `k`**: the augmented ordered chain complex `k[Y_•] → k` of the
  vertex scheme of `model7Cover` is acyclic -- from `model7_acyclic` (H4(ii), `GL2.Model.Acyclic`) through the sign
  bridge `Ξ` (ordered chains = `XBulk.chains model7`).
* `model7_comparison_unc`: **`H^n(SL₂(ℤ); A) ≅ H^n(C^•(S(M₇); A))` for every commutative ring `k`, every representation
  `A` and every `n`**, with no hypothesis left (unnormalised MP cochains).
-/

namespace GL2.MP

open GL2.TauChk GL2.TauChk.Data GL2.Model
open scoped MatrixGroups

/-- **The bridge checks for `M₇`** (`decide +kernel`). -/
theorem model7_bridge_checks : ACBridge.Checks tauP2 model7MP := by decide +kernel

/-- **(AC) for `M₇`** over every commutative ring `k` ([C] `thm:mpcomparison` (5), row `M₇`). -/
theorem model7_AC (k : Type) [CommRing k] : (Cover.yCx model7Hyp k).toCx.Acyclic :=
  ACBridge.yCx_acyclic_of_bridge (of_decide_eq_true tauP2_xbulkB) model7_wf model7_rules.2.1 model7_dimB
    model7_bridge_checks model7_acyclic (M7.vx 0) k

/-- **[C] `thm:mpcomparison` for `M₇`, unconditional** (unnormalised MP cochains): for every commutative ring `k`, every
representation `A` of `SL₂(ℤ)` over `k` and every `n`, `H^n(SL₂(ℤ); A) ≅ H^n(C^•(S(M₇); A))`. -/
noncomputable def model7_comparison_unc {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) :
    groupCohomology A n ≅ (model7Cover.mpComplex A).homology n :=
  model7_comparison (model7_AC k) A n

end GL2.MP
