import GL2.Model.H4
import GL2.Model.Model7
import GL2.Model.Acyclic
import GL2.Model.Bridge
import GL2.Cone.ProjReduction

/-!
# G4 (L18-7): the projective reduction on `X_bulk` and on the 7-vertex model

[GT] `lem:xbulk-permproj`, mirroring [C] v1.17.0 `rmk:orbifold` ("each `R[Γ/Γ_σ]` is projective relative to the
trivial subgroup once `6 ∈ R^×`") and `H3`.

For an `X_bulk` descriptor `D` whose representative stabilisers `S(σ, σ)` have order dividing `12` (H4(iv)), the
permutation representation `R[Simp D q]` on the `q`-simplices of `X_bulk` is projective in `Rep R SL(2, ℤ)` as soon as
`6 ∈ R^×`; for `model7` the orders are `6, 6, 4, 2, …` (`model7_stab`), so `model7_perm_projective` needs nothing else.
With `GL2.Cone.ProjRed.projRedPerm` this is the reduction of `rmk:orbifold`: a resolution of `R` by these permutation
modules computes `H^•(SL(2, ℤ); V)`.

**The bridge on the model** (L19-2, [GT] `thm:xbulk-repaug`, `lem:tauexists-2-model`): with the identification of the
oriented chains `C_q(X_bulk) ⊗ R` with `R[Simp D q]` (`GL2.Model.Bridge`, `XBulk.repCx`) and `model7_acyclic` (H4(ii),
`GL2.Model.Acyclic`), `model7_projRes` is a projective resolution of `k` for `6 ∈ k^×`, and `model7_tauexists_2` is
[C] `lem:tauexists`(2) on the model for chain maps over `k`.  `rmk:orbifold` itself is `GL2.Model.RmkOrbifold`.
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open CategoryTheory GL2.Cone.ProjRed

variable {k : Type} [CommRing k]

/-- `H3` arithmetic: `6 ∈ k^×` makes every divisor of `12` a unit (the stabiliser orders of H4(iv)). -/
lemma isUnit_of_dvd_twelve (h6 : IsUnit (6 : k)) {m : ℕ} (hm : m ∣ 12) : IsUnit (m : k) := by
  have h2 : IsUnit (2 : k) := isUnit_of_mul_isUnit_left (show IsUnit ((2 : k) * 3) by norm_num; exact h6)
  have h12 : IsUnit ((12 : ℕ) : k) := by
    rw [show ((12 : ℕ) : k) = 2 * 6 by norm_num]; exact h2.mul h6
  exact isUnit_natCast_of_dvd hm h12

namespace XBulk

variable (D : XBulkDatum)

/-- The stabiliser of `ι(σ)` has `|S(σ, σ)|` elements. -/
lemma card_stab_mk (q : ℤ) (σ : D.V q) :
    Nat.card (MulAction.stabilizer SL(2, ℤ) (SimpM.mk (D.M q) (1, σ))) = ((D.M q).S σ σ).card := by
  rw [← Nat.card_eq_finsetCard]
  refine Nat.card_congr (Equiv.subtypeEquivRight fun δ => ?_)
  rw [mem_stab_iff D q (1, σ) δ]
  simp

/-- Every simplex stabiliser is finite (H4(i)). -/
lemma finite_stab (q : ℤ) (s : Simp D q) : Finite (MulAction.stabilizer SL(2, ℤ) s) := by
  obtain ⟨γ, σ, -, rfl⟩ := fundDomain D q s
  have := (finiteStab D q (γ, σ)).to_subtype
  rwa [smul_simpMk, mul_one]

/-- With `|S(σ, σ)| ∣ 12` (H4(iv)) and `6 ∈ k^×` (H3), every simplex stabiliser has invertible order. -/
lemma isUnit_card_stab (h6 : IsUnit (6 : k))
    (hdvd : ∀ q (σ : D.V q), (D.M q).rep σ = σ → ((D.M q).S σ σ).card ∣ 12) (q : ℤ) (s : Simp D q) :
    IsUnit (Nat.card (MulAction.stabilizer SL(2, ℤ) s) : k) := by
  have key : Nat.card (MulAction.stabilizer SL(2, ℤ) s) ∣ 12 := by
    obtain ⟨γ, σ, hσ, rfl⟩ := fundDomain D q s
    rw [← Nat.card_congr (MulAction.stabilizerEquivStabilizer (rfl : γ • SimpM.mk (D.M q) (1, σ) = _)).toEquiv,
      card_stab_mk]
    exact hdvd q σ hσ
  exact isUnit_of_dvd_twelve h6 key

/-- **`lem:xbulk-permproj`.** If every representative stabiliser order `|S(σ, σ)|` divides `12` (H4(iv)) and
`6 ∈ k^×` (H3), then the permutation representation `k[Simp D q]` on the `q`-simplices of `X_bulk` is projective. -/
theorem perm_projective (h6 : IsUnit (6 : k))
    (hdvd : ∀ q (σ : D.V q), (D.M q).rep σ = σ → ((D.M q).S σ σ).card ∣ 12) (q : ℤ) :
    Projective (Rep.ofMulAction k SL(2, ℤ) (Simp D q)) :=
  projective_ofMulAction (k := k) (finite_stab D q) (isUnit_card_stab D h6 hdvd q)

end XBulk

/-- H4(iv) on the model: the representative stabiliser orders `stabOrd ∈ {2, 4, 6}` divide `12`. -/
lemma model7_dvd : ∀ q (σ : model7.V q), (model7.M q).rep σ = σ → ((model7.M q).S σ σ).card ∣ 12 := fun q σ _ => by
  rw [model7_stab.2.2.1 q σ]
  unfold GL2.TauChk.Data.stabOrd
  split_ifs <;> decide

/-- **`lem:xbulk-permproj` on the 7-vertex model**: with `6 ∈ k^×`, every `k[Simp model7 q]` is projective in
`Rep k SL(2, ℤ)` (the stabiliser orders `stabOrd ∈ {2, 4, 6}` of `model7_stab` divide `12`). -/
theorem model7_perm_projective (h6 : IsUnit (6 : k)) (q : ℤ) :
    Projective (Rep.ofMulAction k SL(2, ℤ) (XBulk.Simp model7 q)) :=
  XBulk.perm_projective model7 h6 model7_dvd q

/-! ### The bridge on the model: a projective resolution, and `lem:tauexists`(2) (L19-2) -/

variable (k)

/-- **`thm:xbulk-repaug` on the model**: `C_•(model7) ⊗ k → k` is a quasi-isomorphism, over any commutative ring
(`model7_acyclic`, H4(ii), and the vertex `v₀`). -/
theorem model7_repAug_quasiIso : QuasiIso (XBulk.repAug model7 k) :=
  XBulk.repAug_quasiIso k model7_acyclic (M7.vx 0)

variable {k}

/-- **`model7_projRes`**: for `6 ∈ k^×`, the oriented chains of the 7-vertex model, `k[Simp_2] → k[Simp_1] → k[Simp_0]`,
are a projective resolution of the trivial representation `k` of `SL(2, ℤ)` ([C] `rmk:orbifold`). -/
def model7_projRes (h6 : IsUnit (6 : k)) : ProjectiveResolution (Rep.trivial k SL(2, ℤ) k) :=
  haveI := model7_repAug_quasiIso k
  projRes (XBulk.repCx model7 k) (XBulk.repAug model7 k) (fun n => model7_perm_projective h6 n)

lemma model7_projRes_complex (h6 : IsUnit (6 : k)) : (model7_projRes h6).complex = XBulk.repCx model7 k := rfl

/-- **`model7_tauexists_2`** ([C] `lem:tauexists`(2) on the model, `6 ∈ k^×`): for two injective homomorphisms
`f₁, f₂ : H → SL(2, ℤ)` (for a `g`-twisted chain map: `f₁` the inclusion `Γ'' ≤ Γ`, `f₂ = h ↦ g⁻¹ h g`), an
augmentation-compatible `H`-chain map `res_{f₁} C_•(model7) ⊗ k → res_{f₂} C_•(model7) ⊗ k` exists, and any two such are
chain homotopic. -/
theorem model7_tauexists_2 (h6 : IsUnit (6 : k)) {H : Type} [Group H] (f₁ f₂ : H →* SL(2, ℤ))
    (hf₁ : Function.Injective f₁) (hf₂ : Function.Injective f₂) :
    let P₁ := resProjRes f₁ hf₁ (model7_projRes h6) (fun n => XBulk.Simp model7 n) (fun _ => Iso.refl _)
      (fun n => XBulk.finite_stab model7 n) (fun n => XBulk.isUnit_card_stab model7 h6 model7_dvd n)
    let P₂ := resProjRes f₂ hf₂ (model7_projRes h6) (fun n => XBulk.Simp model7 n) (fun _ => Iso.refl _)
      (fun n => XBulk.finite_stab model7 n) (fun n => XBulk.isUnit_card_stab model7 h6 model7_dvd n)
    (∃ τ : P₁.complex ⟶ P₂.complex, τ ≫ P₂.π = P₁.π) ∧
      ∀ τ τ' : P₁.complex ⟶ P₂.complex, τ ≫ P₂.π = P₁.π → τ' ≫ P₂.π = P₁.π → Nonempty (Homotopy τ τ') :=
  twisted_exists_unique_upto_homotopy (model7_projRes h6) (fun n => XBulk.Simp model7 n) (fun _ => Iso.refl _)
    (fun n => XBulk.finite_stab model7 n) (fun n => XBulk.isUnit_card_stab model7 h6 model7_dvd n) f₁ f₂ hf₁ hf₂

end GL2.Model

end
