import GL2.M4.Data

/-!
# M4-f (slot 20, L20-5): the objects of [C] `thm:classical` on the model

`T_{Γ g Γ} = cor^Γ_{Γ''} ∘ (c_g)_* ∘ res^Γ_{Γ'}` on `H^n(SL(2, ℤ); V)`, with `Γ'' = Γ ∩ g Γ g⁻¹` (the group of the coset
datum: `Γ = ⊔ Γ'' s_j`), `Γ' = Γ ∩ g⁻¹ Γ g`, and `c_g : H^n(Γ'; V) → H^n(Γ''; V)` induced by `x ↦ g⁻¹ x g : Γ'' → Γ'` and
the coefficient map `ρ(g)` (`ρ(g) ρ(g⁻¹ x g) = ρ(x) ρ(g)`). In degree `0` the composite is `v ↦ ∑_j s_j⁻¹ g v`, the
model's `∑_j ρ(δ_j) v`, `δ_j = s_j⁻¹ g` (`def:cosettransport`).

Here: `VL ρ` (`V` as a representation of `SL(2, ℤ)` through `ρ ∘ e⁻¹`), `Θ` (`H^n` of the model `→ H^n(SL(2, ℤ); V)`: `Ψ`, then
`model7_rmkOrbifoldL`), `Spp g` (`Γ''` in `SL(2, ℤ)`; `Spp g⁻¹` is `Γ'`), `conjHom g`, `conjCoeff ρ g`, `resTo`, `conjMap`.
mathlib `db584cd` has `res` and the conjugation (`groupCohomology.map`) but no corestriction: `cor` is the hypothesised
transfer of `lit:hecke-classical` (`GL2.Literature.HeckeClassical`).

**Slot 23 (M6-A):** the genuine corestriction exists since M5 (`GL2.M5.corH`), and `lit:hecke-classical` is a theorem
(`GL2.M6.heckeClassical_holds`); nothing here is hypothesised any more.
-/

noncomputable section

namespace GL2.M4

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cosets CategoryTheory

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

/-- `V` as a representation of `SL(2, ℤ)`: `t ↦ ρ(e⁻¹ t)`. -/
abbrev VL : Rep R SL(2, ℤ) := Rep.of (ρLRep ΓQ_le_ΔQ ρ eQ)

/-- `Θ : H^n(model) → H^n(SL(2, ℤ); V_L)`: `Ψ` on cohomology, then `model7_rmkOrbifoldL` (`6 ∈ R^×`). -/
def Θ (h6 : IsUnit (6 : R)) (n : ℕ) : HM ρ n →ₗ[R] groupCohomology (VL ρ) n :=
  (model7_rmkOrbifoldL ΓQ_le_ΔQ ρ eQ h6 n).toLinearMap ∘ₗ
    HmapOf (psi model7 ρ eQ (hΓΔ := ΓQ_le_ΔQ)) (psi_isChainMap model7 ρ eQ) n

/-- `Γ'' = Γ ∩ g Γ g⁻¹` as a subgroup of `SL(2, ℤ)`. -/
def Spp (g : GLQ) : Subgroup SL(2, ℤ) := (Γpp (Γ := ΓQ) g).map eQ.toMonoidHom

lemma mem_Spp {g : GLQ} {x : SL(2, ℤ)} : x ∈ Spp g ↔ g⁻¹ * ((eQ.symm x : ΓQ) : GLQ) * g ∈ ΓQ :=
  Subgroup.mem_map_equiv

/-- `x ↦ g⁻¹ x g : Γ'' → Γ'` (`Γ' = Spp g⁻¹ = Γ ∩ g⁻¹ Γ g`). -/
def conjHom (g : GLQ) : Spp g →* Spp g⁻¹ where
  toFun x := ⟨eQ ⟨g⁻¹ * ((eQ.symm x.1 : ΓQ) : GLQ) * g, mem_Spp.1 x.2⟩, by
    rw [mem_Spp, MulEquiv.symm_apply_apply, inv_inv]
    simp only
    have : g * (g⁻¹ * ((eQ.symm x.1 : ΓQ) : GLQ) * g) * g⁻¹ = ((eQ.symm x.1 : ΓQ) : GLQ) := by group
    rw [this]; exact (eQ.symm x.1).2⟩
  map_one' := by
    apply Subtype.ext
    simp only [OneMemClass.coe_one, map_one]
    rw [← map_one eQ]
    congr 1
    apply Subtype.ext
    simp
  map_mul' x y := by
    apply Subtype.ext
    simp only [Subgroup.coe_mul, map_mul]
    rw [← map_mul eQ]
    congr 1
    apply Subtype.ext
    simp only [Subgroup.coe_mul]
    group

/-- The coefficient map `ρ(g) : res(V_L) → res(V_L)` intertwines `x ↦ g⁻¹ x g`: `ρ(g) ρ(g⁻¹ x g) = ρ(x) ρ(g)`. -/
def conjCoeff (g : GLQ) (hg : g ∈ ΔQ) :
    Rep.res (conjHom g) (Rep.res (Spp g⁻¹).subtype (VL ρ)) ⟶ Rep.res (Spp g).subtype (VL ρ) :=
  ConcreteCategory.ofHom
    { toLinearMap := ρ ⟨g, hg⟩
      isIntertwining' := fun x => by
        apply LinearMap.ext; intro v
        show ρ ⟨g, hg⟩ (ρ ⟨_, ΓQ_le_ΔQ (eQ.symm (eQ _))⟩ v) = ρ ⟨_, ΓQ_le_ΔQ (eQ.symm x.1)⟩ (ρ ⟨g, hg⟩ v)
        rw [← Module.End.mul_apply, ← Module.End.mul_apply, ← map_mul, ← map_mul]
        congr 2
        apply Subtype.ext
        simp only [Submonoid.coe_mul, MulEquiv.symm_apply_apply]
        group }

/-- `res^Γ_H : H^n(SL(2, ℤ); V_L) → H^n(H; V_L)`. -/
def resTo (H : Subgroup SL(2, ℤ)) (n : ℕ) :
    groupCohomology (VL ρ) n →ₗ[R] groupCohomology (Rep.res H.subtype (VL ρ)) n :=
  (groupCohomology.map H.subtype (𝟙 _) n).hom

/-- `(c_g)_* : H^n(Γ'; V_L) → H^n(Γ''; V_L)`. -/
def conjMap (g : GLQ) (hg : g ∈ ΔQ) (n : ℕ) :
    groupCohomology (Rep.res (Spp g⁻¹).subtype (VL ρ)) n →ₗ[R] groupCohomology (Rep.res (Spp g).subtype (VL ρ)) n :=
  (groupCohomology.map (conjHom g) (conjCoeff ρ g hg) n).hom

end GL2.M4

end
