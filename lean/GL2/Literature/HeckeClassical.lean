import GL2.M4.ClassicalDefs

/-!
# L1: the double-coset operator as `cor ∘ (c_g)_* ∘ res` (slot 20, L20-5, branch H)

[GT] Part L row `lit:hecke-classical` (L20-0 §3.2, §3.3). Sources: Brown, *Cohomology of groups* (GTM 87), Prop. III.9.5(ii), p. 82
(`cor ∘ res = (G : H)`) and (iii), pp. 82–83 (the double-coset formula; not §10), `c(g)^*` in §8, p. 80; Weibel 6.7.17, p. 194
(a second leg for `cor ∘ res = [G : H]`: "the composite `tr ∘ res` is multiplication by `[G:H]` on `H^*(G;A)`", Weibel's
cohomological `tr` being our `cor`); Ash–Stevens, Crelle 365 (1986), §1.1.

**Locators (R20-L, 2026-10-05).** Brown: verified. **Ash–Stevens: NOT AS CITED.** §1.1, pp. 193–196, gives `T_g` by the
explicit cocycle formula (p. 194) and states res / cor compatibility (Lemma 1.1.3), but never writes
`T_g = cor ∘ c_g ∘ res` on `H^n(Γ; V)`. The identity is standard; it follows from Ash–Stevens' formula and Brown §9 (D), and is
in Kuga–Parry–Sah (Ash–Stevens' [20]), which was **not read**. So `hcl` rests on that derivation, not on a quotation. No
quotation is given.

mathlib `db584cd` has no corestriction, so `cor` is **hypothesised**: the structure asserts a linear map
`cor : H^n(Γ''; V_L) → H^n(SL(2, ℤ); V_L)` with the characterising property `cor ∘ res = [Γ : Γ'']` (Brown Prop. III.9.5(ii); Weibel 6.7.17) that
carries the model's operator: `Θ ∘ T_{Γ g Γ} = cor ∘ (c_g)_* ∘ res ∘ Θ`, where `Θ` is the identification of the model's
`H^n` with `H^n(SL(2, ℤ); V_L)` (`Ψ`, then `model7_rmkOrbifoldL`) and `(c_g)_*`, `res` are mathlib's `groupCohomology.map`.
Stated for the choice-free operator `TdcOf` (one coset datum and one `τ`). **Guards:** `6 ∈ R^×`, `g ∈ Δ`.

**Weak form (F3-20 MAJOR 1).** `cor ∘ res = [Γ : Γ'']` fixes `cor` on `im res` only, so it does not determine `cor`: the
existential asserts that the model's operator has the shape `cor ∘ (c_g)_* ∘ res` for *some* map with this one property of the
transfer. It does **not** identify `cor` with the corestriction, and is weaker than [C] `thm:classical`, which stays [C]-only.

**Slot 23 (M6-A): no longer a hypothesis.** `GL2.M6.heckeClassical_holds : HeckeClassical ρ` proves this structure outright, with
`cor` the genuine corestriction (`GL2.M5.corH`, Brown's transfer by `GL2.M6.corCx_inhomog`); [GT] `lit:hecke-classical` is a
`theorem` row. The structure is kept as the statement.
-/

noncomputable section

namespace GL2.Literature

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cosets GL2.M4

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V]

/-- **`lit:hecke-classical`** (Brown Prop. III.9.5(ii), (iii): verified; Ash–Stevens §1.1: not as cited, derivation from AS p. 194 + Brown §9 (D); Kuga–Parry–Sah not read). For `g ∈ Δ` there is a transfer
`cor : H^n(Γ''; V_L) → H^n(Γ; V_L)` with `cor ∘ res = [Γ : Γ'']` such that the model's `T_{Γ g Γ}` is
`cor ∘ (c_g)_* ∘ res` through `Θ`. Weak form: `cor` is pinned only by `cor ∘ res = [Γ : Γ'']`, not identified with the
corestriction (F3-20 MAJOR 1). -/
structure HeckeClassical (ρ : ΔQ →* Module.End R V) : Prop where
  cor_formula : ∀ (h6 : IsUnit (6 : R)) (g : GLQ) (hg : g ∈ ΔQ) (n : ℕ),
    ∃ cor : groupCohomology (Rep.res (Spp g).subtype (VL ρ)) n →ₗ[R] groupCohomology (VL ρ) n,
      cor ∘ₗ resTo ρ (Spp g) n = (Spp g).index • LinearMap.id ∧
      Θ ρ h6 n ∘ₗ TdcOf ρ h6 g hg n = cor ∘ₗ conjMap ρ g hg n ∘ₗ resTo ρ (Spp g⁻¹) n ∘ₗ Θ ρ h6 n

end GL2.Literature

end
