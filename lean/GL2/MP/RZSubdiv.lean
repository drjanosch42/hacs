import GL2.MP.TabMap
import GL2.MP.RZTau
import GL2.MP.ACModels

/-!
# MP-L2, open item 1 (slot 22): (AC) for `R_ℤ` by the subdivision `M₇ ← R_ℤ`; the unconditional comparison on `R_ℤ`

[GT] `gt_part_MP.tex`, rows `def:mp-rz-tau`, `thm:mp-rz-retr`, `thm:mp-rz-ac`, backing [C] `thm:mpcomparison` (5) for the
row `R_ℤ`.

* `tauRZ_xbulkB`: the generated descriptor `RZTau.tauRZ` (every face of every maximal simplex of the C++ dump of `R_ℤ`,
  every transporter) satisfies the `X_bulk` axioms; `rz_bridge_checks`: it is bridged to `rzMP` (the same labels), so its
  oriented chains are the ordered chains of the MP vertex scheme of `rzCover` (`GL2.MP.ACBridge`).
* `rz_retr`: the tables `sdTab` (`M₇ → R_ℤ`, the edge orbit `{4-6, 5-6}` split at `7, 8`), `cTab` (`R_ℤ → M₇`, the collapse
  `7 ↦ 4`, `8 ↦ 5`), `h0Tab`, `h1Tab` (the homotopy) are admissible, chain maps, and `id - sd ∘ c = d h + h d` on `C(R_ℤ)`
  (`decide +kernel`, through `GL2.MP.TabMap.Br.Same`).
* `rzX_acyclic`: H4(ii) for `R_ℤ`, from `model7_acyclic`; `rz_AC`: **(AC) for `R_ℤ` over every commutative ring**;
  `rz_comparison_unc`: **`H^n(SL₂(ℤ); A) ≅ H^n(C^•(S(R_ℤ); A))`** for every ring, every representation and every `n`.
-/

namespace GL2.MP

open GL2.TauChk GL2.TauChk.Data GL2.Model GL2.MP.RZTau TabMap
open scoped MatrixGroups

/-- **The `X_bulk` axioms for `R_ℤ`** (`decide +kernel` on the generated descriptor). -/
theorem tauRZ_xbulkB : tauRZ.xbulkB = true := by decide +kernel

/-- **The bridge checks for `R_ℤ`** (`decide +kernel`). -/
theorem rz_bridge_checks : ACBridge.Checks tauRZ rzMP := by decide +kernel

/-- `M₇`, bridged. -/
def brM7 : Br := ⟨tauP2, model7MP, of_decide_eq_true tauP2_xbulkB, model7_wf, model7_rules.2.1, model7_dimB,
  model7_bridge_checks⟩

/-- `R_ℤ`, bridged. -/
def brRZ : Br := ⟨tauRZ, rzMP, of_decide_eq_true tauRZ_xbulkB, rz_wf, rz_rules.2.1, rz_dimB, rz_bridge_checks⟩

/-- **The subdivision `M₇ ← R_ℤ` is a deformation retraction on chains** (`decide +kernel`, field by field). -/
theorem rz_retr : RetrChk brM7 brRZ sdTab cTab (h0Tab 0) (h1Tab 1) where
  sd0 := by decide +kernel
  sd1 := by decide +kernel
  sd2 := by decide +kernel
  c0 := by decide +kernel
  c1 := by decide +kernel
  c2 := by decide +kernel
  hk0 := by decide +kernel
  hk1 := by decide +kernel
  wf1 := by decide +kernel
  wf2 := by decide +kernel
  wf1' := by decide +kernel
  wf2' := by decide +kernel
  S1 := by decide +kernel
  S2 := by decide +kernel
  C1 := by decide +kernel
  C2 := by decide +kernel
  Ca := by decide +kernel
  H0 := by decide +kernel
  H1 := by decide +kernel
  H2 := by decide +kernel

/-- **H4(ii) for `R_ℤ`**: its `X_bulk` is acyclic (from `model7_acyclic` through the subdivision). -/
theorem rzX_acyclic : XBulk.Acyclic (tauRZ.toXBulk (of_decide_eq_true tauRZ_xbulkB)) :=
  acyclic_of_retr rz_retr model7_acyclic

/-- **(AC) for `R_ℤ`** over every commutative ring `k` ([C] `thm:mpcomparison` (5), row `R_ℤ`). -/
theorem rz_AC (k : Type) [CommRing k] : (Cover.yCx rzHyp k).toCx.Acyclic :=
  ACBridge.yCx_acyclic_of_bridge (of_decide_eq_true tauRZ_xbulkB) rz_wf rz_rules.2.1 rz_dimB rz_bridge_checks rzX_acyclic
    ⟨0, by decide +kernel⟩ k

/-- **[C] `thm:mpcomparison` for `R_ℤ`, unconditional** (unnormalised MP cochains): for every commutative ring `k`, every
representation `A` of `SL₂(ℤ)` over `k` and every `n`, `H^n(SL₂(ℤ); A) ≅ H^n(C^•(S(R_ℤ); A))`. -/
noncomputable def rz_comparison_unc {k : Type} [CommRing k] (A : Rep.{0} k SL(2, ℤ)) (n : ℕ) :
    groupCohomology A n ≅ (rzCover.mpComplex A).homology n :=
  rz_comparison (rz_AC k) A n

end GL2.MP
