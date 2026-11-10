import GL2.M4.Data
import GL2.M5.Smith

/-!
# M5-E on the model (slot 22): `T_{ΓgΓ}` depends only on the double coset; [C] `eq:fullop` = `eq:fullassembly`

[GT] `gt_part_M5.tex`, rows `prop:m5-tdc-dc` and `thm:m5-fullop`.

* `cdDC`, `tauDC`, `heckeTildeR_DC` (generic `(G, Γ, Δ)`): from a coset datum `s_i` and a `τ` for `g`, the datum
  `s'_i = γ₁ s_i γ₁⁻¹` and `τ' = γ₂⁻¹ · τ(γ₁⁻¹ · –)` serve `g' = γ₁ g γ₂` (`γ₁, γ₂ ∈ Γ`), and `T̃_{g'} = T̃_g` **at cochain
  level** (no hypothesis on `ℓ`): `ρ(s'^{-1}_i g') ρ(γ₂⁻¹) = ρ(γ₁) ρ(s_i⁻¹ g)` and the `Γ`-equivariance of `T̃_g φ`.
* `TdcOf_DC` (`6 ∈ R^×`): `TdcOf (γ₁ g γ₂) = TdcOf g` -- the operator of a double coset is well defined.
* `fullop_eq_Tfull`: for every complete irredundant system of representatives `g_1, …, g_k ∈ Δ` of the `Γ` double cosets
  of determinant `p^ν`, `∑_i T_{Γ g_i Γ} = Tfull` -- [C] `eq:fullop` (Shimura's `T(p^ν)`, the sum over **all** double cosets
  of determinant `p^ν`) equals [GT]'s full operator, defined by the assembled form `eq:fullassembly` (`def:m4-hecke`).
  `fullop_system`: such a system exists (`g_j = p^j diag(1, p^{ν-2j})`, `0 ≤ 2j ≤ ν`). Route: the enumeration
  `GL2.M5.dc_enum`, `TdcOf_DC`, then `fullassembly` (`ν - 2j ≥ 1`) and `diamond` (`ν = 2j`).

[C] `eq:fullop` keeps **no** `\LEAN` marker (owner ruling Q5-4): it stays [C]'s definition; these rows back it.
-/

noncomputable section

namespace GL2.M5

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cosets GL2.M4 CategoryTheory MonoidAlgebra

section Generic

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G}
  {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] {g : G} (γ₁ γ₂ : Γ)

variable (hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ) in
/-- **A coset datum for `γ₁ g γ₂`** from one for `g`: `s'_i = γ₁ s_i γ₁⁻¹`, `δ'_i = γ₁ δ_i γ₂`. -/
def cdDC (cd : CosetData Γ Δ g) : CosetData Γ Δ ((γ₁ : G) * g * γ₂) where
  m := cd.m
  s i := γ₁ * cd.s i * γ₁⁻¹
  δ_mem i := by
    have : (((γ₁ * cd.s i * γ₁⁻¹ : Γ) : G))⁻¹ * ((γ₁ : G) * g * γ₂) =
        (γ₁ : G) * (((cd.s i : G))⁻¹ * g) * γ₂ := by
      simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
    rw [this]; exact Δ.mul_mem (Δ.mul_mem (hΓΔ γ₁) (cd.δ_mem i)) (hΓΔ γ₂)
  cosets γ := by
    obtain ⟨i, ⟨h, hh, he⟩, hu⟩ := cd.cosets (γ₁⁻¹ * γ * γ₁)
    refine ⟨i, ⟨γ₁ * h * γ₁⁻¹, ?_, ?_⟩, ?_⟩
    · have : ((γ₁ : G) * g * γ₂)⁻¹ * ((γ₁ * h * γ₁⁻¹ : Γ) : G) * ((γ₁ : G) * g * γ₂) =
          ((γ₂⁻¹ : Γ) : G) * (g⁻¹ * (h : G) * g) * (γ₂ : G) := by
        simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
      rw [this]; exact Γ.mul_mem (Γ.mul_mem (γ₂⁻¹).2 hh) γ₂.2
    · have he' : ((γ₁⁻¹ * γ * γ₁ : Γ) : G) = (h : G) * (cd.s i : G) := he
      simp only [Subgroup.coe_mul, Subgroup.coe_inv] at he' ⊢
      calc (γ : G) = (γ₁ : G) * ((γ₁ : G)⁻¹ * γ * γ₁) * (γ₁ : G)⁻¹ := by group
        _ = _ := by rw [he']; group
    · rintro i' ⟨h', hh', he'⟩
      refine hu i' ⟨γ₁⁻¹ * h' * γ₁, ?_, ?_⟩
      · have : g⁻¹ * ((γ₁⁻¹ * h' * γ₁ : Γ) : G) * g =
            (γ₂ : G) * (((γ₁ : G) * g * γ₂)⁻¹ * (h' : G) * ((γ₁ : G) * g * γ₂)) * ((γ₂⁻¹ : Γ) : G) := by
          simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
        rw [this]; exact Γ.mul_mem (Γ.mul_mem γ₂.2 hh') (γ₂⁻¹).2
      · simp only [Subgroup.coe_mul, Subgroup.coe_inv] at he' ⊢
        rw [he']; group

variable (e : Γ ≃* SL(2, ℤ))

/-- `h ∈ Γ''(γ₁ g γ₂) ↦ γ₁⁻¹ h γ₁ ∈ Γ''(g)`. -/
def ppDC (h : Γpp (Γ := Γ) ((γ₁ : G) * g * γ₂)) : Γpp (Γ := Γ) g :=
  ⟨γ₁⁻¹ * h.1 * γ₁, by
    show g⁻¹ * ((γ₁⁻¹ * h.1 * γ₁ : Γ) : G) * g ∈ Γ
    have hm : ((γ₁ : G) * g * γ₂)⁻¹ * ((h.1 : Γ) : G) * ((γ₁ : G) * g * γ₂) ∈ Γ := h.2
    have : g⁻¹ * ((γ₁⁻¹ * h.1 * γ₁ : Γ) : G) * g =
        (γ₂ : G) * (((γ₁ : G) * g * γ₂)⁻¹ * ((h.1 : Γ) : G) * ((γ₁ : G) * g * γ₂)) * ((γ₂⁻¹ : Γ) : G) := by
      simp only [Subgroup.coe_mul, Subgroup.coe_inv]; group
    rw [this]; exact Γ.mul_mem (Γ.mul_mem γ₂.2 hm) (γ₂⁻¹).2⟩

/-- The permutation action of `SL(2, ℤ)` on `R[Simp_n]`. -/
abbrev act (n : ℕ) (t : SL(2, ℤ)) : Module.End R (R[Simp model7 n]) :=
  Representation.ofMulAction R SL(2, ℤ) (Simp model7 n) t

lemma act_mul_apply (n : ℕ) (s t : SL(2, ℤ)) (x : R[Simp model7 n]) :
    act (R := R) n s (act n t x) = act n (s * t) x := by
  unfold act; rw [map_mul]; rfl

lemma act_repD (n : ℕ) (t : SL(2, ℤ)) (y : R[Simp model7 (n + 1 : ℕ)]) :
    (repD model7 R n).hom (act (n + 1) t y) = act n t ((repD model7 R n).hom y) :=
  Rep.hom_comm_apply (repD model7 R n) t y

lemma act_repEps (t : SL(2, ℤ)) (y : R[Simp model7 0]) :
    (repEps model7 R).hom (act 0 t y) = (repEps model7 R).hom y := by
  have := Rep.hom_comm_apply (repEps model7 R) t y
  simpa [act] using this

variable (R) in
/-- **A `τ` for `γ₁ g γ₂`** from one for `g`: `τ'(x) = γ₂⁻¹ · τ(γ₁⁻¹ · x)`. -/
def tauDC (τ : TauR R g e) : TauR R ((γ₁ : G) * g * γ₂) e where
  f n := Rep.ofHom ⟨act n (e γ₂⁻¹) ∘ₗ (τ.f n).hom.toLinearMap ∘ₗ act n (e γ₁⁻¹), fun h => by
    refine LinearMap.ext fun x => ?_
    show act n (e γ₂⁻¹) ((τ.f n).hom (act n (e γ₁⁻¹) (act n (fIncl _ e h) x))) =
      act n (fConj _ e h) (act n (e γ₂⁻¹) ((τ.f n).hom (act n (e γ₁⁻¹) x)))
    have h1 : act n (e γ₁⁻¹) (act n (fIncl _ e h) x) =
        act n (fIncl g e (ppDC γ₁ γ₂ h)) (act (R := R) n (e γ₁⁻¹) x) := by
      rw [act_mul_apply, act_mul_apply]
      congr 2
      show e γ₁⁻¹ * e h.1 = e (γ₁⁻¹ * h.1 * γ₁) * e γ₁⁻¹
      rw [← map_mul, ← map_mul]; congr 1; group
    have h2 := τ.twist n (ppDC γ₁ γ₂ h) (act (R := R) n (e γ₁⁻¹) x)
    rw [h1, h2, act_mul_apply, act_mul_apply]
    congr 2
    show e γ₂⁻¹ * e (conjHom g (ppDC γ₁ γ₂ h)) = e (conjHom ((γ₁ : G) * g * γ₂) h) * e γ₂⁻¹
    rw [← map_mul, ← map_mul]
    congr 1
    apply Subtype.ext
    simp only [Subgroup.coe_mul, Subgroup.coe_inv, coe_conjHom, ppDC]
    group⟩
  t2 n x := by
    show (repD model7 R n).hom (act (n + 1) (e γ₂⁻¹) ((τ.f (n + 1)).hom (act (n + 1) (e γ₁⁻¹) x))) =
      act n (e γ₂⁻¹) ((τ.f n).hom (act n (e γ₁⁻¹) ((repD model7 R n).hom x)))
    rw [act_repD, τ.t2, act_repD]
  t3 x := by
    show (repEps model7 R).hom (act 0 (e γ₂⁻¹) ((τ.f 0).hom (act 0 (e γ₁⁻¹) x))) = (repEps model7 R).hom x
    rw [act_repEps, τ.t3, act_repEps]

variable {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ} (ρ : Δ →* Module.End R V)

/-- **`T̃` depends only on the double coset, at cochain level**: `T̃_{γ₁ g γ₂}` on `cdDC`, `tauDC` is `T̃_g`. -/
theorem heckeTildeR_DC (cd : CosetData Γ Δ g) (τ : TauR R g e) :
    heckeTildeR (hΓΔ := hΓΔ) ρ e (cdDC γ₁ γ₂ hΓΔ cd) (tauDC R γ₁ γ₂ e τ) = heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ := by
  funext q
  apply LinearMap.ext; intro φ
  apply Subtype.ext; apply AddMonoidHom.ext; intro y
  have heq := (heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ q φ).2 γ₁ (γ₁⁻¹ • y)
  rw [smul_inv_smul] at heq
  rw [heq, heckeTildeR_apply, heckeTildeR_apply, map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hρ : ρ ⟨((cdDC γ₁ γ₂ hΓΔ cd).s i : G)⁻¹ * ((γ₁ : G) * g * γ₂), (cdDC γ₁ γ₂ hΓΔ cd).δ_mem i⟩ *
      ρΓ Γ Δ hΓΔ ρ γ₂⁻¹ = ρΓ Γ Δ hΓΔ ρ γ₁ * ρ ⟨(cd.s i : G)⁻¹ * g, cd.δ_mem i⟩ := by
    simp only [ρΓ]
    rw [← map_mul, ← map_mul]
    congr 1
    apply Subtype.ext
    simp only [Submonoid.coe_mul, cdDC, Subgroup.coe_mul, Subgroup.coe_inv]
    group
  rcases q with n | k
  · have hsm : ∀ (s : Γ) (z : (Yc model7 e).X (Int.ofNat n)),
        embR (R := R) (D := model7) (n : ℤ) (s • z) = act n (e s) (embR (n : ℤ) z) :=
      fun s z => embR_smul (D := model7) (e s) z
    have hFR : FR (tauDC R γ₁ γ₂ e τ) (Int.ofNat n) ((cdDC γ₁ γ₂ hΓΔ cd).s i • y) =
        act n (e γ₂⁻¹) (FR τ (Int.ofNat n) (cd.s i • γ₁⁻¹ • y)) := by
      show act n (e γ₂⁻¹) ((τ.f n).hom (act n (e γ₁⁻¹) (embR (R := R) (n : ℤ) ((cdDC γ₁ γ₂ hΓΔ cd).s i • y)))) =
        act n (e γ₂⁻¹) ((τ.f n).hom (embR (R := R) (n : ℤ) (cd.s i • γ₁⁻¹ • y)))
      rw [hsm, hsm, hsm, act_mul_apply, act_mul_apply]
      congr 4
      show e γ₁⁻¹ * e (γ₁ * cd.s i * γ₁⁻¹) = e (cd.s i) * e γ₁⁻¹
      rw [← map_mul, ← map_mul]; congr 1; group
    rw [hFR]
    have hl := linR_mem ρ φ γ₂⁻¹ (FR τ (Int.ofNat n) (cd.s i • γ₁⁻¹ • y))
    change linR (R := R) φ.1 (act n (e γ₂⁻¹) _) = ρΓ Γ Δ hΓΔ ρ γ₂⁻¹ (linR (R := R) φ.1 _) at hl
    rw [hl, ← Module.End.mul_apply, hρ, Module.End.mul_apply]
  · have h0 : ∀ z, FR (tauDC R γ₁ γ₂ e τ) (Int.negSucc k) z = 0 := fun _ => rfl
    have h0' : ∀ z, FR τ (Int.negSucc k) z = 0 := fun _ => rfl
    rw [h0, h0']
    show ρ _ (linR (R := R) φ.1 0) = ρΓ Γ Δ hΓΔ ρ γ₁ (ρ _ (linR (R := R) φ.1 0))
    rw [map_zero, map_zero, map_zero, map_zero]

end Generic

/-! ### On the model: `TdcOf` is a function of the double coset -/

section Model

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

/-- **M5-E (4): `T_{Γ g Γ}` depends only on `Γ g Γ`** (`6 ∈ R^×`): `TdcOf g' = TdcOf g` whenever `g' = γ₁ g γ₂`,
`γ₁, γ₂ ∈ Γ`. -/
theorem TdcOf_DC (h6 : IsUnit (6 : R)) {g g' : GLQ} (hg : g ∈ ΔQ) (hg' : g' ∈ ΔQ) (γ₁ γ₂ : ΓQ)
    (hgg : g' = (γ₁ : GLQ) * g * γ₂) (n : ℕ) : TdcOf ρ h6 g' hg' n = TdcOf ρ h6 g hg n := by
  subst hgg
  obtain ⟨cd⟩ := cosetData_nonempty g hg
  obtain ⟨τ⟩ := tauR_nonempty R h6 g
  rw [TdcOf_eq ρ h6 _ hg' (cdDC γ₁ γ₂ ΓQ_le_ΔQ cd) (tauDC R γ₁ γ₂ eQ τ), TdcOf_eq ρ h6 g hg cd τ]
  unfold Tdc
  exact HmapOf_congr (heckeTildeR_DC γ₁ γ₂ eQ ρ cd τ) _ _ _

/-- `p^j diag(1, p^μ)` in `GL(2, ℚ)`. -/
def gD (p j μ : ℕ) (hp : p.Prime) : GLQ :=
  Matrix.GeneralLinearGroup.mkOfDetNeZero (κ (dMat p j μ)) (by
    rw [← RingHom.map_det, dMat_det, eq_intCast]
    exact_mod_cast pow_ne_zero _ hp.ne_zero)

lemma coe_gD (p j μ : ℕ) (hp : p.Prime) : ((gD p j μ hp : GLQ) : Matrix (Fin 2) (Fin 2) ℚ) = κ (dMat p j μ) := rfl

lemma gD_mem (p j μ : ℕ) (hp : p.Prime) : gD p j μ hp ∈ ΔQ := ⟨dMat p j μ, rfl⟩

/-- A double-coset relation in `GL(2, ℚ)` between integral matrices is one in `M_2(ℤ)`. -/
lemma DCRel_of_GL {A B : GLQ} {M N : Mat2 ℤ} (hA : (A : Matrix (Fin 2) (Fin 2) ℚ) = κ M)
    (hB : (B : Matrix (Fin 2) (Fin 2) ℚ) = κ N) {γ₁ γ₂ : ΓQ} (h : A = (γ₁ : GLQ) * B * γ₂) : DCRel M N := by
  refine ⟨eQ γ₁, eQ γ₂, κ_injective ?_⟩
  have := congrArg (fun u : GLQ => (u : Matrix (Fin 2) (Fin 2) ℚ)) h
  simp only [Units.val_mul, coe_ΓQ, coe_mapGL] at this
  rw [map_mul, map_mul, ← hA, ← hB, this]

/-- The converse: a double-coset relation in `M_2(ℤ)` gives one in `GL(2, ℚ)`. -/
lemma GL_of_DCRel {A B : GLQ} {M N : Mat2 ℤ} (hA : (A : Matrix (Fin 2) (Fin 2) ℚ) = κ M)
    (hB : (B : Matrix (Fin 2) (Fin 2) ℚ) = κ N) (h : DCRel M N) :
    ∃ γ₁ γ₂ : ΓQ, A = (γ₁ : GLQ) * B * γ₂ := by
  obtain ⟨t₁, t₂, ht⟩ := h
  refine ⟨eQ.symm t₁, eQ.symm t₂, Units.ext ?_⟩
  simp only [Units.val_mul, coe_eQ_symm, coe_mapGL]
  rw [hA, hB, ht, map_mul, map_mul]

/-- An element of `Δ` with determinant `p^ν` is an integral matrix of determinant `p^ν`. -/
lemma int_det {δ : GLQ} {M : Mat2 ℤ} (hM : (δ : Matrix (Fin 2) (Fin 2) ℚ) = κ M) {p ν : ℕ}
    (hdet : (δ : Matrix (Fin 2) (Fin 2) ℚ).det = (p : ℚ) ^ ν) : M.det = (p : ℤ) ^ ν := by
  rw [hM, ← RingHom.map_det, eq_intCast] at hdet
  exact_mod_cast hdet

/-- **`T_{Γ p^j diag(1, p^μ) Γ} = c^j T_{Δ^prim_{p^μ}}`** in the choice-free form (`fullassembly` for `μ ≥ 1`, `diamond`
for `μ = 0`), `c` the scalar of `ρ(pI)`. -/
theorem TdcOf_gD (h6 : IsUnit (6 : R)) (p : ℕ) (hp : p.Prime) (z : GLQ) (hz : z ∈ ΔQ) (c : R)
    (hzs : IsScalarQ z p) (hρ : ρ ⟨z, hz⟩ = c • 1) (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (j μ n : ℕ) :
    TdcOf ρ h6 (gD p j μ hp) (gD_mem p j μ hp) n = c ^ j • Tprim ρ p hp τ n μ := by
  obtain ⟨cd⟩ := cosetData_nonempty _ (gD_mem p j μ hp)
  obtain ⟨τ'⟩ := tauR_nonempty R h6 (gD p j μ hp)
  rw [TdcOf_eq ρ h6 _ _ cd τ']
  by_cases hμ : 1 ≤ μ
  · rw [Tprim, dif_pos hμ]
    exact fullassembly ρ p hp z hz c hzs hρ h6 j μ hμ _ (coe_gD p j μ hp) cd τ' (τ μ) n
  · have hμ0 : μ = 0 := by omega
    subst hμ0
    rw [Tprim, dif_neg hμ]
    have hgz : gD p j 0 hp = z ^ j := by
      apply Units.ext
      rw [coe_gD, Units.val_pow_eq_pow_val, hzs, ← map_pow, dMat, smul_pow, one_pow, pow_zero]
      congr 1
      ext a b; fin_cases a <;> fin_cases b <;> simp [gMat]
    have hsc : IsScalarQ (gD p j 0 hp) ((p : ℤ) ^ j) := by
      show _ = _
      rw [coe_gD, dMat, pow_zero]
      congr 1
      ext a b; fin_cases a <;> fin_cases b <;> simp [gMat]
    have hρj : ρ ⟨gD p j 0 hp, gD_mem p j 0 hp⟩ = c ^ j • 1 := by
      have : (⟨gD p j 0 hp, gD_mem p j 0 hp⟩ : ΔQ) = ⟨z, hz⟩ ^ j := Subtype.ext (by simp only [SubmonoidClass.coe_pow]; exact hgz)
      rw [this, map_pow, hρ, smul_pow, one_pow]
    exact diamond ρ _ _ _ _ hsc hρj h6 cd τ' n

/-- **M5-E (5), [C] `eq:fullop` = [GT]'s `Tfull`**: for every complete irredundant system `g_1, …, g_k ∈ Δ` of
representatives of the `Γ` double cosets of determinant `p^ν` (every `δ ∈ Δ` with `det δ = p^ν` lies in exactly one
`Γ g_i Γ`), `∑_i T_{Γ g_i Γ} = Tfull` (`eq:fullassembly`, `T(p, p)` acting by the scalar `c` of `ρ(pI)`). -/
theorem fullop_eq_Tfull (h6 : IsUnit (6 : R)) (p : ℕ) (hp : p.Prime) (z : GLQ) (hz : z ∈ ΔQ) (c : R)
    (hzs : IsScalarQ z p) (hρ : ρ ⟨z, hz⟩ = c • 1) (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (ν n : ℕ)
    {k : ℕ} (gs : Fin k → GLQ) (hgs : ∀ i, gs i ∈ ΔQ)
    (hdet : ∀ i, ((gs i : GLQ) : Matrix (Fin 2) (Fin 2) ℚ).det = (p : ℚ) ^ ν)
    (hcomplete : ∀ δ : GLQ, δ ∈ ΔQ → (δ : Matrix (Fin 2) (Fin 2) ℚ).det = (p : ℚ) ^ ν →
      ∃! i, ∃ γ₁ γ₂ : ΓQ, δ = (γ₁ : GLQ) * gs i * γ₂) :
    ∑ i, TdcOf ρ h6 (gs i) (hgs i) n = Tfull ρ p hp τ c n ν := by
  -- the integral matrices of the representatives and their elementary-divisor index
  have hM : ∀ i, ∃ M : Mat2 ℤ, ((gs i : GLQ) : Matrix (Fin 2) (Fin 2) ℚ) = κ M := fun i => hgs i
  choose M hMi using hM
  have hcl : ∀ i, ∃ j, 2 * j ≤ ν ∧ DCRel (M i) (dMat p j (ν - 2 * j)) :=
    fun i => dc_classify hp ν (M i) (int_det (hMi i) (hdet i))
  choose f hf hfr using hcl
  have hTi : ∀ i, TdcOf ρ h6 (gs i) (hgs i) n = c ^ (f i) • Tprim ρ p hp τ n (ν - 2 * f i) := by
    intro i
    obtain ⟨γ₁, γ₂, hγ⟩ := GL_of_DCRel (hMi i) (coe_gD p (f i) (ν - 2 * f i) hp) (hfr i)
    rw [TdcOf_DC ρ h6 (gD_mem _ _ _ hp) (hgs i) γ₁ γ₂ hγ, TdcOf_gD ρ h6 p hp z hz c hzs hρ τ]
  simp only [hTi, Tfull]
  refine Finset.sum_nbij f (fun i _ => ?_) (fun i _ i' _ hii => ?_) (fun j hj => ?_) (fun i _ => rfl)
  · have := hf i; simp only [Finset.mem_range]; omega
  · -- injective: `g_i` and `g_{i'}` lie in one double coset, so `i = i'` by irredundancy
    have hr : DCRel (M i) (M i') := (hfr i).trans (hii ▸ (hfr i').symm)
    obtain ⟨γ₁, γ₂, hγ⟩ := GL_of_DCRel (hMi i) (hMi i') hr
    exact (hcomplete (gs i) (hgs i) (hdet i)).unique ⟨1, 1, by simp⟩ ⟨γ₁, γ₂, hγ⟩
  · -- surjective onto `{j | 2j ≤ ν}`: the representative `p^j diag(1, p^{ν-2j})` lies in some `Γ g_i Γ`
    have hj' : 2 * j ≤ ν := by simp only [Finset.coe_range, Set.mem_Iio] at hj; omega
    have hdj : ((gD p j (ν - 2 * j) hp : GLQ) : Matrix (Fin 2) (Fin 2) ℚ).det = (p : ℚ) ^ ν := by
      rw [coe_gD, ← RingHom.map_det, (dc_enum hp ν).2 j hj', eq_intCast]; push_cast; rfl
    obtain ⟨i, ⟨γ₁, γ₂, hγ⟩, -⟩ := hcomplete _ (gD_mem p j (ν - 2 * j) hp) hdj
    have hr : DCRel (dMat p j (ν - 2 * j)) (dMat p (f i) (ν - 2 * f i)) :=
      (DCRel_of_GL (coe_gD p j _ hp) (hMi i) hγ).trans (hfr i)
    exact ⟨i, Finset.mem_coe.2 (Finset.mem_univ i), ((dc_unique hp hr).1).symm⟩

/-- **The canonical system** `g_j = p^j diag(1, p^{ν-2j})`, `0 ≤ 2j ≤ ν`, is complete and irredundant (so the sum
over all double cosets of determinant `p^ν` has exactly `⌊ν/2⌋ + 1` terms). -/
theorem fullop_system (p : ℕ) (hp : p.Prime) (ν : ℕ) :
    (∀ j : Fin (ν / 2 + 1), gD p j (ν - 2 * j) hp ∈ ΔQ) ∧
    (∀ j : Fin (ν / 2 + 1), ((gD p j (ν - 2 * j) hp : GLQ) : Matrix (Fin 2) (Fin 2) ℚ).det = (p : ℚ) ^ ν) ∧
    (∀ δ : GLQ, δ ∈ ΔQ → (δ : Matrix (Fin 2) (Fin 2) ℚ).det = (p : ℚ) ^ ν →
      ∃! j : Fin (ν / 2 + 1), ∃ γ₁ γ₂ : ΓQ, δ = (γ₁ : GLQ) * gD p j (ν - 2 * j) hp * γ₂) := by
  have hle : ∀ j : Fin (ν / 2 + 1), 2 * (j : ℕ) ≤ ν := fun j => by have := j.2; omega
  refine ⟨fun j => gD_mem _ _ _ hp, fun j => ?_, fun δ hδ hdet => ?_⟩
  · rw [coe_gD, ← RingHom.map_det, (dc_enum hp ν).2 j (hle j), eq_intCast]; push_cast; rfl
  · obtain ⟨N, hN⟩ := hδ
    obtain ⟨j, hj, hr⟩ := dc_classify hp ν N (int_det hN hdet)
    refine ⟨⟨j, by omega⟩, GL_of_DCRel hN (coe_gD p j _ hp) hr, fun j' ⟨γ₁, γ₂, hγ⟩ => ?_⟩
    have hr' := DCRel_of_GL hN (coe_gD p j' _ hp) hγ
    exact Fin.ext (dc_unique hp (hr'.symm.trans hr)).1

/-- [C] `eq:fullop` on the model through the canonical system: `∑_{0 ≤ 2j ≤ ν} T_{Γ p^j diag(1, p^{ν-2j}) Γ} = Tfull`. -/
theorem fullop_canonical (h6 : IsUnit (6 : R)) (p : ℕ) (hp : p.Prime) (z : GLQ) (hz : z ∈ ΔQ) (c : R)
    (hzs : IsScalarQ z p) (hρ : ρ ⟨z, hz⟩ = c • 1) (τ : ∀ μ : ℕ, TauR R (gP p μ hp) eQ) (ν n : ℕ) :
    ∑ j : Fin (ν / 2 + 1), TdcOf ρ h6 (gD p j (ν - 2 * j) hp) (gD_mem _ _ _ hp) n = Tfull ρ p hp τ c n ν :=
  fullop_eq_Tfull ρ h6 p hp z hz c hzs hρ τ ν n _ (fun _ => gD_mem _ _ _ hp) (fullop_system p hp ν).2.1
    (fullop_system p hp ν).2.2

end Model

end GL2.M5

end
