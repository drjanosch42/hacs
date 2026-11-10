import GL2.Model.Psi
import GL2.Model.Model7
import GL2.Cosets.Intake

/-!
# M2-f (L18-2): [C] `thm:welldef` over the `X_bulk` descriptor

[GT] `thm:welldef-lean`, `def:heckeP`, mirroring [C] v1.17.0 `def:hecke` (`eq:heckechain`), `thm:welldef` (the
equivariance / chain-map clause and (a)), `lem:cocycle`, `prop:cosets`(3).

**Generic part.** For any `G ⊇ Γ`, `Δ ⊇ Γ`, `g ∈ G`, coset data `cd` for `Γ g Γ` (`s_1, …, s_m`, `δ_j = s_j⁻¹ g ∈ Δ`) and a
`g`-twisted chain datum `τ` on `C_•(X_bulk)` ((T1), (T2), (T3)),
`heckeTilde cd τ = T̃ : φ ↦ ∑_j ρ(δ_j) φ(τ(s_j ·))` is the cone layer's `thetaG` (`eq:heckechain`):
* `heckeTilde_equivariant` -- `T̃` carries `Γ`-equivariant cochains to `Γ`-equivariant cochains (no hypothesis on `ℓ`);
* `heckeTilde_isChainMap` -- `T̃` commutes with the coboundary ((T2) and additivity); so it descends to `H^q`;
* `heckeTilde_indep_reps` -- (a) at cochain level: with `τ` **held fixed**, replacing `s_j` by `h_j s_j`
  (`h_j ∈ Γ'' = Γ ∩ g Γ g⁻¹`, any re-indexing) gives literally the same cochain map.
(b) needs `lem:tauexists`(2) (G4) and is not here; (c) for transported choices is a tail item.

**The level-`p^ν` instance.** `G = GL(2, ℚ)`, `Γ = SL(2, ℤ)` (as `ΓQ`, the image of `mapGL`), `Δ = ` integral matrices,
`g = diag(1, p^ν)`; `cdP` is the coset datum of L17-3's canonical transporters `s_k = sSL p ν k`, whose `δ_k` is
`deltaMat p ν k` (`cdP_delta`), and whose coset decomposition is `cls` / `lem:cocycle` (`hcoc`) together with
`prop:cosets`(3) in the form `g⁻¹ h g ∈ Γ ⟺ h ∈ Γ₀(p^ν)` (`conj_mem_ΓQ_iff`).  `heckeP` is `T̃` for it.

*Restatement note.* [C] proves equivariance with the cocycle `h_j(γ)` of `lem:cocycle` and the bijection `j ↦ j·γ`; the
cone layer's `transport_equivariant` proves it for *any* coset datum from the transversal property alone, so for `cdP`
the cocycle enters through `cls_spec` / `cls_eq` (the coset decomposition), and `heckeP_summand` records [C]'s
summand identity `ρ(δ_k) φ(τ(s_k γ x)) = ρ(γ) ρ(δ_{k·γ}) φ(τ(s_{k·γ} x))` with `h_k(γ) = hcoc p ν k γ` explicitly.
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open GL2.Ku GL2.Cone GL2.Cosets Matrix CongruenceSubgroup

/-! ### Generic: `T̃` is a cochain map, and (a) -/

section Generic

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ}
  {R : Type*} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : Δ →* Module.End R V) {g : G}

/-- **`thm:welldef`, chain map**, for any `Γ`-complex: `T̃(dφ) = d(T̃φ)` by (T2), term by term. -/
theorem thetaG_isChainMap {Y : GChainCx.{0} Γ} (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g Y) :
    IsChainMap (thetaG (hΓΔ := hΓΔ) (ρ := ρ) cd τ) := by
  intro q
  apply LinearMap.ext; intro φ
  apply Subtype.ext
  show (transport (ρ := ρ) Γ Δ cd (τ.τ q) φ.1).comp (Y.bd q) =
    transport (ρ := ρ) Γ Δ cd (τ.τ (q + 1)) (φ.1.comp (Y.bd q))
  rw [← (transport_bd (ρ := ρ) cd).2 (τ.τ q) φ.1, ← (transport_bd (ρ := ρ) cd).1 (τ.τ (q + 1)) φ.1]
  congr 1
  ext x
  exact (τ.t2 q x).symm

/-- **`thm:welldef`(a)** at cochain level, `τ` held fixed: changing the representatives `s_j ↦ h_j s_j` with
`h_j ∈ Γ''` (and re-indexing by any bijection `π`) does not change `T̃` at all. -/
theorem thetaG_indep_reps {Y : GChainCx.{0} Γ} (cd cd' : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g Y)
    (π : Fin cd.m ≃ Fin cd'.m) (h : Fin cd.m → Γ) (hh : ∀ i, g⁻¹ * (h i : G) * g ∈ Γ)
    (hs : ∀ i, cd'.s (π i) = h i * cd.s i) :
    thetaG (hΓΔ := hΓΔ) (ρ := ρ) cd' τ = thetaG cd τ := by
  funext q
  apply LinearMap.ext; intro φ
  apply Subtype.ext
  show transport (ρ := ρ) Γ Δ cd' (τ.τ q) φ.1 = transport (ρ := ρ) Γ Δ cd (τ.τ q) φ.1
  ext y
  simp only [transport, AddMonoidHom.finsetSum_apply, AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply,
    LinearMap.toAddMonoidHom_coe]
  rw [← Equiv.sum_comp π]
  refine Finset.sum_congr rfl fun i _ => ?_
  have e1 : cd'.s (π i) • y = h i • cd.s i • y := by rw [hs i, mul_smul]
  rw [e1, τ.t1 q (h i) ⟨_, hh i⟩ rfl, φ.2]
  simp only [ρΓ]
  rw [← Module.End.mul_apply, ← map_mul]
  congr 2
  apply Subtype.ext
  simp only [Submonoid.coe_mul]
  rw [hs i]
  simp only [Subgroup.coe_mul]
  group

/-! ### `thm:welldef`(c): transported choices

*Restatement (erratum candidate for [C] v1.17.0 `thm:welldef`, proof of (c)).* [C] takes `{s_j γ₁⁻¹}` as the
representatives of `Γ''₁\Γ` for `g₁ = γ₁ g γ₂` (`Γ''₁ = γ₁ Γ'' γ₁⁻¹`).  That family is not a transversal in general, and
with `τ₁ = γ₂⁻¹ · τ(γ₁⁻¹ · −)` its summands are `τ(γ₁⁻¹ s_j γ₁⁻¹ x)`, not `τ(s_j x)`.  The transported representatives
are `γ₁ s_j`: `⊔_j Γ''₁ γ₁ s_j = γ₁ ⊔_j Γ'' s_j = Γ`, `δ'_j = (γ₁ s_j)⁻¹ g₁ = δ_j γ₂ ∈ Δ`, and then the operator is
reproduced verbatim (`thetaG_transported`), the factor `ρ(γ₂)` cancelling against equivariance of `φ`. -/

section Transported

/-- The transported twisted datum `τ₁ = γ₂⁻¹ · τ(γ₁⁻¹ · −)`, `g₁ = γ₁ g γ₂`. -/
def transportedTau {Y : GChainCx.{0} Γ} (τ : TwistedChainDatum Γ g Y) (γ₁ γ₂ : Γ) :
    TwistedChainDatum Γ ((γ₁ : G) * g * γ₂) Y where
  τ q := (DistribSMul.toAddMonoidHom _ γ₂⁻¹).comp ((τ.τ q).comp (DistribSMul.toAddMonoidHom _ γ₁⁻¹))
  t1 q := by
    intro h h' hh' x
    simp only [AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply]
    have hm : (g⁻¹ * ((γ₁⁻¹ * h * γ₁ : Γ) : G) * g) = (γ₂ * h' * γ₂⁻¹ : Γ) := by
      simp only [Subgroup.coe_mul, Subgroup.coe_inv, hh']; group
    have hmem : g⁻¹ * ((γ₁⁻¹ * h * γ₁ : Γ) : G) * g ∈ Γ := by rw [hm]; exact (γ₂ * h' * γ₂⁻¹).2
    have e1 : γ₁⁻¹ • h • x = (γ₁⁻¹ * h * γ₁) • γ₁⁻¹ • x := by
      rw [← mul_smul, ← mul_smul]; group
    rw [e1, τ.t1 q _ ⟨_, hmem⟩ rfl]
    have e2 : (⟨_, hmem⟩ : Γ) = γ₂ * h' * γ₂⁻¹ := Subtype.ext hm
    rw [e2, ← mul_smul, ← mul_smul]
    congr 1; group
  t2 q x := by
    simp only [AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply]
    rw [Y.bd_equiv, τ.t2, Y.bd_equiv]
  t3 x := by
    simp only [AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply]
    rw [Y.aug_equiv, τ.t3, Y.aug_equiv]

/-- The transported coset datum: representatives `γ₁ s_j`, so `δ'_j = δ_j γ₂`. -/
def transportedCd (hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ) (cd : CosetData Γ Δ g) (γ₁ γ₂ : Γ) :
    CosetData Γ Δ ((γ₁ : G) * g * γ₂) where
  m := cd.m
  s i := γ₁ * cd.s i
  δ_mem i := by
    have : ((γ₁ * cd.s i : Γ) : G)⁻¹ * ((γ₁ : G) * g * γ₂) = ((cd.s i : G)⁻¹ * g) * γ₂ := by
      simp only [Subgroup.coe_mul]; group
    rw [this]; exact Δ.mul_mem (cd.δ_mem i) (hΓΔ γ₂)
  cosets γ := by
    obtain ⟨i, ⟨h, hh, hγ⟩, huniq⟩ := cd.cosets (γ₁⁻¹ * γ)
    refine ⟨i, ⟨γ₁ * h * γ₁⁻¹, ?_, ?_⟩, ?_⟩
    · have : ((γ₁ : G) * g * γ₂)⁻¹ * ((γ₁ * h * γ₁⁻¹ : Γ) : G) * ((γ₁ : G) * g * γ₂) =
          (γ₂ : G)⁻¹ * (g⁻¹ * h * g) * γ₂ := by simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
      rw [this]
      exact Γ.mul_mem (Γ.mul_mem (Γ.inv_mem γ₂.2) hh) γ₂.2
    · have h1 : (γ : G) = γ₁ * (h * cd.s i) := by
        rw [← hγ]; simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
      rw [h1]; simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
    · rintro j ⟨h', hh', hγ'⟩
      apply huniq j
      refine ⟨γ₁⁻¹ * h' * γ₁, ?_, ?_⟩
      · have : g⁻¹ * ((γ₁⁻¹ * h' * γ₁ : Γ) : G) * g =
            (γ₂ : G) * (((γ₁ : G) * g * γ₂)⁻¹ * h' * ((γ₁ : G) * g * γ₂)) * (γ₂ : G)⁻¹ := by
          simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
        rw [this]
        exact Γ.mul_mem (Γ.mul_mem γ₂.2 hh') (Γ.inv_mem γ₂.2)
      · simp only [Subgroup.coe_mul, Subgroup.coe_inv] at hγ' ⊢
        rw [hγ']; group

/-- **`thm:welldef`(c)** for the transported choices, at cochain level, no hypothesis on `ℓ`: the operator of
`(g₁, {γ₁ s_j}, τ₁)` is the operator of `(g, {s_j}, τ)` verbatim. -/
theorem thetaG_transported {Y : GChainCx.{0} Γ} (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g Y) (γ₁ γ₂ : Γ) :
    thetaG (hΓΔ := hΓΔ) (ρ := ρ) (transportedCd hΓΔ cd γ₁ γ₂) (transportedTau τ γ₁ γ₂) = thetaG cd τ := by
  funext q
  apply LinearMap.ext; intro φ
  apply Subtype.ext
  show transport (ρ := ρ) Γ Δ (transportedCd hΓΔ cd γ₁ γ₂) ((transportedTau τ γ₁ γ₂).τ q) φ.1 =
    transport (ρ := ρ) Γ Δ cd (τ.τ q) φ.1
  ext y
  simp only [transport, AddMonoidHom.finsetSum_apply, AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply,
    LinearMap.toAddMonoidHom_coe]
  refine Finset.sum_congr rfl fun i _ => ?_
  have e1 : γ₁⁻¹ • (transportedCd hΓΔ cd γ₁ γ₂).s i • y = cd.s i • y := by
    show γ₁⁻¹ • (γ₁ * cd.s i) • y = _
    rw [← mul_smul, inv_mul_cancel_left]
  simp only [transportedTau, AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply]
  rw [e1, φ.2]
  simp only [ρΓ]
  rw [← Module.End.mul_apply, ← map_mul]
  congr 2
  apply Subtype.ext
  show ((γ₁ * cd.s i : Γ) : G)⁻¹ * ((γ₁ : G) * g * γ₂) * ((γ₂⁻¹ : Γ) : G) = (cd.s i : G)⁻¹ * g
  simp only [Subgroup.coe_mul, Subgroup.coe_inv]
  group

end Transported

end Generic

/-! ### `T̃` on `X_bulk` -/

namespace XBulk

variable (D : XBulkDatum) {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ}
  {R : Type*} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : Δ →* Module.End R V) (e : Γ ≃* SL(2, ℤ))
  {g : G}

/-- **`def:hecke`** (`eq:heckechain`) on `X_bulk`: `(T̃φ)(x) = ∑_j ρ(δ_j) φ(τ(s_j x))`. -/
def heckeTilde (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc D e)) :
    GMap (EqC D ρ e (hΓΔ := hΓΔ)) (EqC D ρ e (hΓΔ := hΓΔ)) :=
  thetaG cd τ

lemma heckeTilde_apply (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ)
    (φ : (EqC D ρ e (hΓΔ := hΓΔ)).X q) (y : (Yc D e).X q) :
    (heckeTilde D ρ e cd τ q φ).1 y =
      ∑ i, ρ ⟨(cd.s i : G)⁻¹ * g, cd.δ_mem i⟩ (φ.1 (τ.τ q (cd.s i • y))) := by
  show transport (ρ := ρ) Γ Δ cd (τ.τ q) φ.1 y = _
  rw [transport, AddMonoidHom.finsetSum_apply]
  rfl

/-- **`thm:welldef`, equivariance** (no hypothesis on `ℓ`): `T̃` carries `Γ`-equivariant cochains to `Γ`-equivariant
cochains, i.e. `(T̃φ)(γ x) = ρ(γ)(T̃φ)(x)` for `γ ∈ Γ`; in `SL(2, ℤ)` terms `T̃ φ` is `ρ_L`-equivariant. -/
theorem heckeTilde_equivariant (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc D e)) (q : ℤ)
    (φ : (Yc D e).X q →+ V) (hφ : IsEqv D hΓΔ ρ e φ) :
    IsEqv D hΓΔ ρ e (transport (ρ := ρ) Γ Δ cd (A := (Yc D e).X q) (τ.τ q) φ) :=
  (mem_eqHom_iff D ρ e _).1
    (transport_equivariant (hΓΔ := hΓΔ) cd (τ.τ q) (τ.t1 q) φ ((mem_eqHom_iff D ρ e φ).2 hφ)).1

/-- **`thm:welldef`, chain map** (no hypothesis on `ℓ`): `T̃` commutes with the coboundary. -/
theorem heckeTilde_isChainMap (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc D e)) :
    IsChainMap (heckeTilde D ρ e (hΓΔ := hΓΔ) cd τ) :=
  thetaG_isChainMap ρ cd τ

/-- **`thm:welldef`(a)** at cochain level, `τ` fixed (no hypothesis on `ℓ`). -/
theorem heckeTilde_indep_reps (cd cd' : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc D e))
    (π : Fin cd.m ≃ Fin cd'.m) (h : Fin cd.m → Γ) (hh : ∀ i, g⁻¹ * (h i : G) * g ∈ Γ)
    (hs : ∀ i, cd'.s (π i) = h i * cd.s i) :
    heckeTilde D ρ e (hΓΔ := hΓΔ) cd' τ = heckeTilde D ρ e cd τ :=
  thetaG_indep_reps ρ cd cd' τ π h hh hs

/-- `T̃` transported to the relative complex by `Ψ` (the operator the implementation computes on `C^•_eq`); a cochain
map because `Ψ`, `Φ` and `T̃` are. -/
def heckeRel (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc D e)) :
    GMap (relCx D hΓΔ ρ e) (relCx D hΓΔ ρ e) :=
  fun q => (psi D ρ e q).comp ((heckeTilde D ρ e cd τ q).comp (phi D ρ e q))

theorem heckeRel_isChainMap (cd : CosetData Γ Δ g) (τ : TwistedChainDatum Γ g (Yc D e)) :
    IsChainMap (heckeRel D ρ e (hΓΔ := hΓΔ) cd τ) := by
  intro q
  apply LinearMap.ext; intro f
  simp only [heckeRel, LinearMap.comp_apply]
  rw [(psi_isChainMap D ρ e).apply, (heckeTilde_isChainMap D ρ e cd τ).apply, (phi_isChainMap D ρ e).apply]

end XBulk

/-! ### The level-`p^ν` instance: `G = GL(2, ℚ)`, `g = diag(1, p^ν)` -/

section LevelP

/-- `GL(2, ℚ)`. -/
abbrev GLQ := GL (Fin 2) ℚ

/-- Entrywise `ℤ → ℚ` on `2 × 2` matrices. -/
abbrev κ : Mat2 ℤ →+* Matrix (Fin 2) (Fin 2) ℚ := (Int.castRingHom ℚ).mapMatrix

lemma κ_injective : Function.Injective κ := fun A B h => by
  ext i j; have := congrFun (congrFun h i) j
  simp only [RingHom.mapMatrix_apply, Matrix.map_apply, eq_intCast, Int.cast_inj] at this; exact this

lemma coe_mapGL (γ : SL(2, ℤ)) : ((Matrix.SpecialLinearGroup.mapGL ℚ γ : GLQ) : Matrix (Fin 2) (Fin 2) ℚ) = κ γ := by
  rw [Matrix.SpecialLinearGroup.mapGL_coe_matrix]; rfl

/-- `Γ = SL(2, ℤ)` inside `GL(2, ℚ)`. -/
def ΓQ : Subgroup GLQ := (Matrix.SpecialLinearGroup.mapGL ℚ : SL(2, ℤ) →* GLQ).range

lemma mapGL_inj : Function.Injective (Matrix.SpecialLinearGroup.mapGL ℚ : SL(2, ℤ) →* GLQ) :=
  Matrix.SpecialLinearGroup.mapGL_injective

/-- `ΓQ ≅ SL(2, ℤ)`. -/
def eQ : ΓQ ≃* SL(2, ℤ) := (MonoidHom.ofInjective mapGL_inj).symm

lemma coe_eQ_symm (t : SL(2, ℤ)) : ((eQ.symm t : ΓQ) : GLQ) = Matrix.SpecialLinearGroup.mapGL ℚ t := by
  exact MonoidHom.ofInjective_apply mapGL_inj

lemma coe_ΓQ (γ : ΓQ) : (γ : GLQ) = Matrix.SpecialLinearGroup.mapGL ℚ (eQ γ) := by
  rw [← coe_eQ_symm, MulEquiv.symm_apply_apply]

/-- `Δ`: the integral matrices of `GL(2, ℚ)`. -/
def ΔQ : Submonoid GLQ where
  carrier := {A | ∃ M : Mat2 ℤ, (A : Matrix (Fin 2) (Fin 2) ℚ) = κ M}
  mul_mem' := by
    rintro A B ⟨M, hM⟩ ⟨M', hM'⟩
    exact ⟨M * M', by rw [Units.val_mul, hM, hM', map_mul]⟩
  one_mem' := ⟨1, by rw [Units.val_one, map_one]⟩

lemma ΓQ_le_ΔQ (γ : ΓQ) : (γ : GLQ) ∈ ΔQ := ⟨eQ γ, by rw [coe_ΓQ, coe_mapGL]⟩

/-- `g = diag(1, N)` in `GL(2, ℚ)`. -/
def gQ (N : ℤ) (hN : N ≠ 0) : GLQ :=
  Matrix.GeneralLinearGroup.mkOfDetNeZero (κ (gMat N)) (by
    rw [← RingHom.map_det, gMat_det, eq_intCast]; exact_mod_cast hN)

lemma coe_gQ (N : ℤ) (hN : N ≠ 0) : ((gQ N hN : GLQ) : Matrix (Fin 2) (Fin 2) ℚ) = κ (gMat N) := rfl

/-- **[C] `prop:cosets`(3)** in `GL(2, ℚ)`: `g⁻¹ h g ∈ Γ ⟺ N ∣ c(h)`, i.e. `Γ'' = Γ ∩ g Γ g⁻¹ = Γ₀(N)`. -/
theorem conj_mem_ΓQ_iff (N : ℤ) (hN : N ≠ 0) (h : SL(2, ℤ)) :
    (gQ N hN)⁻¹ * Matrix.SpecialLinearGroup.mapGL ℚ h * gQ N hN ∈ ΓQ ↔ N ∣ h 1 0 := by
  rw [← conj_Gamma0_iff]
  constructor
  · rintro ⟨B, hB⟩
    refine ⟨B, κ_injective ?_⟩
    have h1 : gQ N hN * Matrix.SpecialLinearGroup.mapGL ℚ B = Matrix.SpecialLinearGroup.mapGL ℚ h * gQ N hN := by
      rw [hB]; group
    have h2 := congrArg (fun u : GLQ => (u : Matrix (Fin 2) (Fin 2) ℚ)) h1
    simp only [Units.val_mul, coe_mapGL, coe_gQ] at h2
    rw [map_mul, map_mul]; exact h2.symm
  · rintro ⟨B, hB⟩
    refine ⟨B, ?_⟩
    have h1 : Matrix.SpecialLinearGroup.mapGL ℚ h * gQ N hN = gQ N hN * Matrix.SpecialLinearGroup.mapGL ℚ B := by
      apply Units.ext
      simp only [Units.val_mul, coe_mapGL, coe_gQ]
      rw [← map_mul, ← map_mul, hB]
    rw [mul_assoc, h1]; group

variable (p ν : ℕ)

lemma pν_ne (hp : p.Prime) : ((p : ℤ) ^ ν) ≠ 0 := pow_ne_zero _ (by exact_mod_cast Nat.Prime.ne_zero hp)

/-- `g = diag(1, p^ν)`. -/
def gP (hp : p.Prime) : GLQ := gQ ((p : ℤ) ^ ν) (pν_ne p ν hp)

lemma delta_mem (hp : p.Prime) (k : ℕ) : ((eQ.symm (sSL p ν k) : ΓQ) : GLQ)⁻¹ * gP p ν hp ∈ ΔQ :=
  ⟨deltaMat p ν k, by
    rw [Units.val_mul, coe_eQ_symm, ← map_inv, coe_mapGL, gP, coe_gQ, deltaMat, map_mul]⟩

lemma mem_Gamma0_pν (γ : SL(2, ℤ)) : γ ∈ Gamma0 (p ^ ν) ↔ (p : ℤ) ^ ν ∣ γ 1 0 := by
  rw [mem_Gamma0_iff_dvd]; push_cast; rfl

/-- **The coset datum of the canonical transporters** `s_k = sSL p ν k` ([C] `def:cosetreps`), `δ_k = deltaMat p ν k`;
the coset decomposition is `cls` (`lem:cocycle`) and `conj_mem_ΓQ_iff` (`prop:cosets`(3)). -/
def cdP (hp : p.Prime) (hν : 1 ≤ ν) : CosetData ΓQ ΔQ (gP p ν hp) where
  m := mIdx p ν
  s k := eQ.symm (sSL p ν k)
  δ_mem k := delta_mem p ν hp k
  cosets γ := by
    set c := eQ γ
    have hc := cls_spec hp hν c
    refine ⟨⟨cls p ν c, hc.1⟩, ⟨eQ.symm (c * (sSL p ν (cls p ν c))⁻¹), ?_, ?_⟩, ?_⟩
    · rw [coe_eQ_symm, gP, conj_mem_ΓQ_iff, ← mem_Gamma0_pν]; exact hc.2
    · rw [← Subgroup.coe_mul, ← map_mul, inv_mul_cancel_right, MulEquiv.symm_apply_apply]
    · rintro i ⟨h, hconj, hγ⟩
      have hγ' : γ = h * eQ.symm (sSL p ν i) := Subtype.ext hγ
      have hmem : eQ h ∈ Gamma0 (p ^ ν) := by
        rw [mem_Gamma0_pν, ← conj_mem_ΓQ_iff _ (pν_ne p ν hp), ← coe_ΓQ]; exact hconj
      have hc' : c * (sSL p ν i)⁻¹ = eQ h := by
        simp only [c, hγ', map_mul, MulEquiv.apply_symm_apply, mul_inv_cancel_right]
      apply Fin.ext
      exact (cls_eq hp hν i.2 (hc' ▸ hmem)).symm

/-- `δ_k = s_k⁻¹ g` is `deltaMat p ν k` ([C] `prop:cosets`(5)). -/
theorem cdP_delta (hp : p.Prime) (hν : 1 ≤ ν) (k : Fin (cdP p ν hp hν).m) :
    (((((cdP p ν hp hν).s k : ΓQ) : GLQ)⁻¹ * gP p ν hp : GLQ) : Matrix (Fin 2) (Fin 2) ℚ) = κ (deltaMat p ν k) := by
  show (((((eQ.symm (sSL p ν k)) : ΓQ) : GLQ)⁻¹ * gP p ν hp : GLQ) : Matrix (Fin 2) (Fin 2) ℚ) = _
  rw [Units.val_mul, coe_eQ_symm, ← map_inv, coe_mapGL, deltaMat, map_mul]
  rfl

variable (D : XBulkDatum) {R : Type*} [CommRing R] {V : Type} [AddCommGroup V] [Module R V]
  (ρ : ΔQ →* Module.End R V)

/-- **`T_{p^ν}` at cochain level on `X_bulk`** for a `g`-twisted chain datum `τ`: `heckeTilde` of `cdP`. -/
def heckeP (hp : p.Prime) (hν : 1 ≤ ν) (τ : TwistedChainDatum ΓQ (gP p ν hp) (XBulk.Yc D eQ)) :=
  XBulk.heckeTilde D (hΓΔ := ΓQ_le_ΔQ) ρ eQ (cdP p ν hp hν) τ

theorem heckeP_isChainMap (hp : p.Prime) (hν : 1 ≤ ν) (τ : TwistedChainDatum ΓQ (gP p ν hp) (XBulk.Yc D eQ)) :
    IsChainMap (heckeP p ν D ρ hp hν τ) :=
  XBulk.heckeTilde_isChainMap D ρ eQ _ τ

/-- [C]'s summand identity in `thm:welldef`'s proof, with `lem:cocycle`'s `h_k(γ) = hcoc p ν k γ` explicit:
`ρ(δ_k) φ(τ(s_k γ x)) = ρ(γ) ρ(δ_{k·γ}) φ(τ(s_{k·γ} x))` (`k·γ = act p ν k γ`). -/
theorem heckeP_summand (hp : p.Prime) (hν : 1 ≤ ν) (τ : TwistedChainDatum ΓQ (gP p ν hp) (XBulk.Yc D eQ)) (q : ℤ)
    (φ : (XBulk.EqC D (hΓΔ := ΓQ_le_ΔQ) ρ eQ).X q) (k : ℕ) (γ : SL(2, ℤ)) (x : (XBulk.Yc D eQ).X q) :
    ρ ⟨_, delta_mem p ν hp k⟩ (φ.1 (τ.τ q (eQ.symm (sSL p ν k * γ) • x))) =
      XBulk.ρL ΓQ_le_ΔQ ρ eQ γ
        (ρ ⟨_, delta_mem p ν hp (act p ν k γ)⟩ (φ.1 (τ.τ q (eQ.symm (sSL p ν (act p ν k γ)) • x)))) := by
  have hmem : (gP p ν hp)⁻¹ * ((eQ.symm (hcoc p ν k γ) : ΓQ) : GLQ) * gP p ν hp ∈ ΓQ := by
    rw [coe_eQ_symm, gP, conj_mem_ΓQ_iff, ← mem_Gamma0_pν]; exact hcoc_mem hp hν k γ
  have e1 : eQ.symm (sSL p ν k * γ) • x = eQ.symm (hcoc p ν k γ) • eQ.symm (sSL p ν (act p ν k γ)) • x := by
    rw [← mul_smul, ← map_mul, ← hcoc_eq]
  rw [e1, τ.t1 q _ ⟨_, hmem⟩ rfl, φ.2]
  simp only [ρΓ, XBulk.ρL]
  rw [← Module.End.mul_apply, ← Module.End.mul_apply, ← map_mul, ← map_mul]
  congr 2
  apply Subtype.ext
  simp only [Submonoid.coe_mul, coe_eQ_symm]
  have hcG := congrArg (Matrix.SpecialLinearGroup.mapGL ℚ) (hcoc_eq (p := p) (ν := ν) k γ)
  simp only [map_mul] at hcG
  have : (Matrix.SpecialLinearGroup.mapGL ℚ (hcoc p ν k γ) : GLQ) =
      Matrix.SpecialLinearGroup.mapGL ℚ (sSL p ν k) * Matrix.SpecialLinearGroup.mapGL ℚ γ *
        (Matrix.SpecialLinearGroup.mapGL ℚ (sSL p ν (act p ν k γ)))⁻¹ := by
    rw [hcG]; group
  rw [this]; group

end LevelP

/-! ### `Ψ` at `model7` -/

/-- **`thm:explicit_complex_equiv`(1) at the seven-vertex model**, `G = Γ = SL(2, ℤ)`, `Δ = SL(2, ℤ)`, for any
`SL(2, ℤ)`-module `V` over any commutative ring. -/
def psiEquivModel7 {R : Type*} [CommRing R] {V : Type} [AddCommGroup V] [Module R V]
    (ρ : (⊤ : Submonoid SL(2, ℤ)) →* Module.End R V) (q : ℤ) :=
  XBulk.psiEquiv model7 (Γ := ⊤) (hΓΔ := fun _ => Submonoid.mem_top _) ρ Subgroup.topEquiv q

end GL2.Model

end
