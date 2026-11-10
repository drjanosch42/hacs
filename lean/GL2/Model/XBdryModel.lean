import GL2.Model.XBdry
import GL2.Model.Acyclic
import GL2.Model.WellDef
import GL2.Checkers.TauBdP2

/-!
# M3 / C2 (L19-4): `X_∂` on the 7-vertex model, and `PairHyp (model7, X_∂)` without hypotheses

[GT] `def:model7bd`, `thm:model7-pairhyp`, `cor:model7-etaexists`, mirroring [C] v1.18.0 `def:model7` (the horocycle
path `v₀ – v₁ – v₂ – v₃` at `∞`), (B1)–(B3), `prop:model7h4`(ii), `lem:etaexists`, `lem:etaunique`, `thm:conewelldef`(b).

* `model7Bd` -- the boundary descriptor read from the dump: `B` is the `τ_∂`-dump's boundary list
  `K_bd = {v₀, v₁, v₂, v₃, e₀₁, e₁₂, e₂₃}` (`tauBdP2R1.bdry`).  Its four fields are discharged on the stored data:
  `Γ`-saturation (`tauP2_bdry_trs`: a stored transporter starting in `K_bd` ends in `K_bd`), face closure and
  dimension `≤ 1` (the `τ_∂` checker's `bdryOk`, `tauBdP2_spec`), central stabilisers (`|S(σ, σ)| = 2` on `K_bd`
  with `±I ∈ S(σ, σ)`, `model7_stab`).
* `model7_pairHyp` -- for every subgroup `H ≤ SL(2, ℤ)` and every `g ∈ SL(2, ℤ)`, and `model7_pairHypQ` -- for
  `Γ = SL(2, ℤ) ⊂ GL(2, ℚ)` (the Hecke setting of `GL2.Model.WellDef`) and **every** `g ∈ GL(2, ℚ)`: [C]'s full pair
  hypothesis, from `model7_acyclic` (H4(ii)) and the descriptor, with no hypothesis left.
* `model7_eta_exists`, `model7_eta_unique`, `model7_conewelldef_b` -- [C] `lem:etaexists`, `lem:etaunique`,
  `thm:conewelldef`(b) on the model, for any `τ`, `τ_∂`.
-/

noncomputable section

namespace GL2.TauChk.Data
open GL2.TauChk

/-- A stored transporter starting at a boundary simplex ends at one (`K_bd` is `Γ`-saturated). -/
theorem tauP2_bdry_trs : ∀ t ∈ tauP2.trs, (t.q, t.s1) ∈ tauBdP2R1.bdry → (t.q, t.s2) ∈ tauBdP2R1.bdry := by
  decide +kernel

/-- The boundary simplices are the vertices `v₀ … v₃` and edges (no `v₄, v₅, v₆`). -/
theorem tauP2_bdry_small : ∀ x ∈ tauBdP2R1.bdry, x.1 = 1 ∨ x.2 ≤ 3 := by decide +kernel

end GL2.TauChk.Data

namespace GL2.Model

open scoped MatrixGroups
open GL2.TauChk GL2.TauChk.Data GL2.Cone

/-- The boundary predicate: `(q, σ) ∈ K_bd`. -/
def model7B (q : ℤ) (σ : model7.V q) : Prop := 0 ≤ q ∧ (q.toNat, σ.1) ∈ tauBdP2R1.bdry

instance (q : ℤ) : DecidablePred (model7B q) := fun _ => by unfold model7B; infer_instance

lemma one_ne_negOne : (1 : SL(2, ℤ)) ≠ -1 := by
  intro h
  have := congrArg (fun A : SL(2, ℤ) => (A : Matrix (Fin 2) (Fin 2) ℤ) 0 0) h
  simp at this

/-- **`def:model7bd`**: the horocycle boundary `X_∂` of the 7-vertex model, from the `τ_∂` dump. -/
def model7Bd : XBdryDatum model7 where
  B := model7B
  closed_S q σ σ' t ht hB := by
    obtain ⟨t0, ht0, -⟩ := (tauP2.mem_Sset).1 ht
    rw [TauData.mem_trSetZ] at ht0
    obtain ⟨hm, hq, h1, h2⟩ := ht0
    refine ⟨hB.1, ?_⟩
    have hq' : t0.q = q.toNat := by omega
    have := tauP2_bdry_trs t0 hm (by rw [hq', h1]; exact hB.2)
    rwa [hq', h2] at this
  closed_face q τ j hB := by
    obtain ⟨hq0, hj⟩ := nf_pos j
    obtain ⟨-, -, hdim, hface⟩ := tauBdP2_spec.closed.2
    have h1 := hdim _ hB.2
    simp only at h1
    have hq : q = 0 := by omega
    subst hq
    refine ⟨le_refl _, ?_⟩
    have := hface _ hB.2 (by simp) j.1 (List.mem_range.2 (by simpa using hj))
    exact this
  dim1 q σ hB := by
    have := (tauBdP2_spec.closed.2).2.2.1 _ hB.2
    simp only at this
    omega
  central q σ hB t ht := by
    have hS := model7_stab
    have hcard : ((model7.M q).S σ σ).card = 2 := by
      rw [hS.2.2.1 q σ]
      unfold stabOrd
      rcases tauP2_bdry_small _ hB.2 with h | h
      · simp only at h
        have : q ≠ 0 := by omega
        simp [this]
      · simp only at h
        split_ifs with h4 h6
        · omega
        · omega
        · rfl
    have hsub : ({1, -1} : Finset SL(2, ℤ)) ⊆ (model7.M q).S σ σ := by
      intro x hx
      simp only [Finset.mem_insert, Finset.mem_singleton] at hx
      rcases hx with rfl | rfl
      · exact (model7.M q).one_mem σ
      · exact hS.2.1 q σ
    have heq := Finset.eq_of_subset_of_card_le hsub (by
      rw [hcard, Finset.card_pair one_ne_negOne])
    rw [← heq] at ht
    simpa using ht

/-- `-I` lies in every stabiliser of the model ([C] (B3) on the bulk side). -/
lemma model7_negI : ∀ q (σ : model7.V q), (-1 : SL(2, ℤ)) ∈ (model7.M q).S σ σ := model7_stab.2.1

/-- `g⁻¹ (±1) g = ±1` in `SL(2, ℤ)`. -/
lemma hc_SL (H : Subgroup SL(2, ℤ)) (g : SL(2, ℤ)) :
    ∀ h h' : H, (h' : SL(2, ℤ)) = g⁻¹ * h * g → (H.subtype h = 1 ∨ H.subtype h = -1) →
      (H.subtype h' = 1 ∨ H.subtype h' = -1) := by
  intro h h' hh hpm
  simp only [Subgroup.coe_subtype] at hpm ⊢
  rw [hh]
  rcases hpm with e | e <;> rw [e]
  · left; group
  · right; rw [mul_neg_one, neg_mul, inv_mul_cancel]

lemma mapGL_negOne : (Matrix.SpecialLinearGroup.mapGL ℚ (-1 : SL(2, ℤ)) : GLQ) = -1 := by
  apply Units.ext
  rw [coe_mapGL, Units.val_neg, Units.val_one]
  ext i j
  fin_cases i <;> fin_cases j <;> simp

/-- In `GL(2, ℚ)`, `±1` is central, so `g⁻¹ (±1) g = ±1` for every `g`. -/
lemma hc_Q (g : GLQ) :
    ∀ h h' : ΓQ, (h' : GLQ) = g⁻¹ * h * g → (eQ.toMonoidHom h = 1 ∨ eQ.toMonoidHom h = -1) →
      (eQ.toMonoidHom h' = 1 ∨ eQ.toMonoidHom h' = -1) := by
  intro h h' hh hpm
  have hcent : (h : GLQ) = 1 ∨ (h : GLQ) = -1 := by
    rw [coe_ΓQ]
    rcases hpm with e | e <;> simp only [MulEquiv.coe_toMonoidHom] at e <;> rw [e]
    · left; rw [map_one]
    · right; exact mapGL_negOne
  have : h' = h := by
    apply Subtype.ext
    rw [hh]
    rcases hcent with e | e <;> rw [e]
    · group
    · rw [mul_neg_one, neg_mul, inv_mul_cancel]
  rw [this]; exact hpm

/-- **`thm:model7-pairhyp`** (`SL(2, ℤ)` form): the pair `(model7, X_∂)` satisfies `PairHyp` for every subgroup `H` and
every `g ∈ SL(2, ℤ)` -- no hypothesis (H4(ii) is `model7_acyclic`, (B1)–(B3) are `model7Bd`'s fields). -/
theorem model7_pairHyp (H : Subgroup SL(2, ℤ)) (g : SL(2, ℤ)) : PairHyp H (model7Bd.pairVia H.subtype) g :=
  model7Bd.pairHyp_via H.subtype g model7_acyclic model7_negI (hc_SL H g)

/-- **`thm:model7-pairhyp`** (the Hecke form): `Γ = SL(2, ℤ) ⊂ GL(2, ℚ)` acting through `eQ`, every `g ∈ GL(2, ℚ)`. -/
theorem model7_pairHypQ (g : GLQ) : PairHyp ΓQ (model7Bd.pairVia eQ.toMonoidHom) g :=
  model7Bd.pairHyp_via eQ.toMonoidHom g model7_acyclic model7_negI (hc_Q g)

/-- The bulk of the Hecke-form pair is `GL2.Model.WellDef`'s `Yc model7 eQ`. -/
lemma model7_pairQ_bulk : (model7Bd.pairVia eQ.toMonoidHom).bulk = XBulk.Yc model7 eQ := rfl

/-- **`cor:model7-etaexists`** ([C] `lem:etaexists` on the model): for every `g ∈ GL(2, ℚ)` and all `g`-twisted chain
data `τ` on `X_bulk`, `τ_∂` on `X_∂`, a pair homotopy exists over `ℤ`. -/
theorem model7_eta_exists (g : GLQ) (τ : TwistedChainDatum ΓQ g (model7Bd.pairVia eQ.toMonoidHom).bulk)
    (τb : TwistedChainDatum ΓQ g (model7Bd.pairVia eQ.toMonoidHom).bdry) :
    ∃ η, IsPairHomotopy ΓQ (model7Bd.pairVia eQ.toMonoidHom) τ τb η :=
  eta_exists _ (model7_pairHypQ g) τ τb (model7Bd.pairVia_jaug _)

/-- [C] `lem:etaunique` on the model: two pair homotopies differ by `∂Λ − Λ∂` for a unique `Λ`. -/
theorem model7_eta_unique (g : GLQ) (τ : TwistedChainDatum ΓQ g (model7Bd.pairVia eQ.toMonoidHom).bulk)
    (τb : TwistedChainDatum ΓQ g (model7Bd.pairVia eQ.toMonoidHom).bdry)
    (η η' : ∀ q, (model7Bd.pairVia eQ.toMonoidHom).bdry.X q →+ (model7Bd.pairVia eQ.toMonoidHom).bulk.X (q + 1))
    (hη : IsPairHomotopy ΓQ _ τ τb η) (hη' : IsPairHomotopy ΓQ _ τ τb η') :
    ∃! Λ : ∀ q, (model7Bd.pairVia eQ.toMonoidHom).bdry.X q →+ (model7Bd.pairVia eQ.toMonoidHom).bulk.X (q + 1 + 1),
      ∀ q (y : (model7Bd.pairVia eQ.toMonoidHom).bdry.X (q + 1)),
        η' (q + 1) y - η (q + 1) y = (model7Bd.pairVia eQ.toMonoidHom).bulk.bd (q + 1 + 1) (Λ (q + 1) y) -
          Λ q ((model7Bd.pairVia eQ.toMonoidHom).bdry.bd q y) :=
  eta_unique _ (model7_pairHypQ g) τ τb η η' hη hη'

/-- [C] `thm:conewelldef`(b) on the model: the cone operator `Φ_g` does not depend on the pair homotopy, up to a
filtration-preserving cone homotopy -- `PairHyp` discharged. -/
theorem model7_conewelldef_b {Δ : Submonoid GLQ} {hΓΔ : ∀ γ : ΓQ, (γ : GLQ) ∈ Δ} {R : Type} [CommRing R] {V : Type}
    [AddCommGroup V] [Module R V] {ρ : Δ →* Module.End R V} {g : GLQ} (cd : CosetData ΓQ Δ g)
    (τ : TwistedChainDatum ΓQ g (model7Bd.pairVia eQ.toMonoidHom).bulk)
    (τb : TwistedChainDatum ΓQ g (model7Bd.pairVia eQ.toMonoidHom).bdry)
    (η η' : ∀ q, (model7Bd.pairVia eQ.toMonoidHom).bdry.X q →+ (model7Bd.pairVia eQ.toMonoidHom).bulk.X (q + 1))
    (hη : IsPairHomotopy ΓQ _ τ τb η) (hη' : IsPairHomotopy ΓQ _ τ τb η') :
    ∃ Υ : ∀ q, (pairCone (hΓΔ := hΓΔ) (ρ := ρ) (model7Bd.pairVia eQ.toMonoidHom)).X (q + 1) →ₗ[R]
        (pairCone (hΓΔ := hΓΔ) (ρ := ρ) (model7Bd.pairVia eQ.toMonoidHom)).X q,
      (∀ q x, (Υ q x).1 = 0) ∧
      ∀ q x, PhiG cd _ τ τb η' hη'.t1 (q + 1) x - PhiG cd _ τ τb η hη.t1 (q + 1) x =
        (pairCone (hΓΔ := hΓΔ) (ρ := ρ) (model7Bd.pairVia eQ.toMonoidHom)).d q (Υ q x) +
          Υ (q + 1) ((pairCone (hΓΔ := hΓΔ) (ρ := ρ) (model7Bd.pairVia eQ.toMonoidHom)).d (q + 1) x) :=
  conewelldef_b cd _ τ τb η η' hη hη' (model7_pairHypQ g)

end GL2.Model

end
