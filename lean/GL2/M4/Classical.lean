import GL2.Literature.HeckeClassical
import GL2.M4.Relations

/-!
# M4-f (slot 20, L20-5, branch H): [C] `thm:classical` on the model

[GT] `gt_part_M4.tex`, row `thm:classical-model`. For every `g ∈ Δ`, **every** coset datum and every `τ`, the model's
`T_{Γ g Γ}` on `H^n` is, through `Θ` (`Ψ`, then `model7_rmkOrbifoldL`), `cor ∘ (c_g)_* ∘ res` on `H^n(SL(2, ℤ); V_L)`, for a
transfer `cor` with `cor ∘ res = [Γ : Γ'']`. `proved_mod_lit` over `lit:hecke-classical` (`hcl`), which states it for one
datum (`TdcOf`); the extension to all data is ours (`Tdc_eq_TdcOf`: `heckeTildeR_indep_cd`, `heckeTildeR_indep_tau`).

**Weak form (F3-20 MAJOR 1):** for *some* `R`-linear `cor` with `cor ∘ res = [Γ : Γ'']`; this does not identify `cor` with the
corestriction and is weaker than [C] `thm:classical`, which stays [C]-only.

**Slot 23 (M6-A):** `hcl` is now a theorem (`GL2.M6.heckeClassical_holds`, with `cor` the genuine corestriction), so the statement
holds outright (`GL2.M6.thm_classical_outright`, [GT] `thm:classical-model` is `proved`); [C] `thm:classical` in [C]'s sense is
`GL2.M6.classical_Θ`. This module's `thm_classical` (over `hcl`) is kept unchanged and is no longer linked in [GT].
-/

noncomputable section

namespace GL2.M4

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cosets GL2.Literature

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

/-- [C] `thm:classical` on the model, **weak form**: `T_{Γ g Γ}` has the shape `cor ∘ (c_g)_* ∘ res` through `Θ` for some
`cor` with `cor ∘ res = [Γ : Γ'']` (not identified with the corestriction), any coset datum, any `τ`. -/
def thm_classical_stmt : Prop :=
  ∀ (h6 : IsUnit (6 : R)) (g : GLQ) (cd : CosetData ΓQ ΔQ g) (τ : TauR R g eQ) (n : ℕ),
    ∃ cor : groupCohomology (Rep.res (Spp g).subtype (VL ρ)) n →ₗ[R] groupCohomology (VL ρ) n,
      cor ∘ₗ resTo ρ (Spp g) n = (Spp g).index • LinearMap.id ∧
      Θ ρ h6 n ∘ₗ Tdc ρ cd τ n = cor ∘ₗ conjMap ρ g (mem_ΔQ_of_cd cd) n ∘ₗ resTo ρ (Spp g⁻¹) n ∘ₗ Θ ρ h6 n

/-- **M4-f, [GT] `thm:classical-model`** (`\gtmodlit` over `hcl`). -/
theorem thm_classical (hcl : HeckeClassical ρ) : thm_classical_stmt ρ := by
  intro h6 g cd τ n
  rw [Tdc_eq_TdcOf ρ h6 cd τ n]
  exact hcl.cor_formula h6 g (mem_ΔQ_of_cd cd) n

end GL2.M4

end
