import GL2.M4.Skeleton

/-!
# M4-a, M4-b (slot 20, L20-1): the data exist; the central operator; the primitive-to-full assembly

[GT] `gt_part_M4.tex`, rows `lem:m4-tauR-exists`, `lem:m4-cosetdata`, `prop:diamond-model`, `prop:fullassembly-model`;
the frozen statements are the `_stmt` definitions of `GL2.M4.Skeleton`.

* `tauR_nonempty` (`6 ∈ R^×`): `model7_tauexists_2` at `fIncl g`, `fConj g`, read as a `TauR`.
* `cosetData_nonempty` (no hypothesis): `Γ'' = Γ ∩ g Γ g⁻¹` has finite index (mathlib `exists_Gamma_le_conj`), and a
  transversal is chosen (L20-0 §4; not the Smith form, not `def:cosetrepsN`).
* `Tp`, `TdcOf`: the choice-free operators, with `Tp_eq`, `TdcOf_eq` (independence of the choices).
* `heckeTildeR_indep_cd` ([C] `thm:welldef`(a) for `τ` over `R`, cochain level) and the **scaling identity**
  `heckeTildeR_scale`: for `z` central in `Δ` with `ρ(z) = c` and `g = z g'`, the coset datum and `τ` of `g'` serve `g`
  unchanged (`cdScale`, `tauScale`), and `T̃_g = c T̃_{g'}` at cochain level.
* `diamond_id` (`τ = id`, no hypothesis on `ℓ`: [C]'s scope), `diamond` (any `τ`, `6 ∈ R^×`), from `g = g · 1` and
  `T̃_{Γ 1 Γ} = id` (`heckeTildeR_one`).
* `fullassembly` (`6 ∈ R^×`): `g = (pI)^j diag(1, p^μ)`, the scaling identity with `z = (pI)^j`, `ρ(z) = c^j`.
-/

noncomputable section

namespace GL2.M4

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cosets CategoryTheory

section Generic

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ}
  {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : Δ →* Module.End R V) {g : G}

/-- `thm:welldef`(a) for the coset transport of any `F` with (T1) and any equivariant `φ`. -/
theorem transport_indep_cd (cd cd' : CosetData Γ Δ g) {Y Y' : GChainCx.{0} Γ} {a b : ℤ} (F : Y.X a →+ Y'.X b)
    (hF : T1 Γ g F) (φ : Y'.X b →+ V) (hφ : φ ∈ eqHom Γ Δ hΓΔ ρ Y' b) :
    transport (ρ := ρ) Γ Δ cd' F φ = transport (ρ := ρ) Γ Δ cd F φ := by
  obtain ⟨π, h, hh, hs⟩ := CosetData.reindex cd cd'
  ext y
  simp only [transport, AddMonoidHom.finsetSum_apply, AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply,
    LinearMap.toAddMonoidHom_coe]
  rw [← Equiv.sum_comp π]
  refine Finset.sum_congr rfl fun i _ => ?_
  have e1 : cd'.s (π i) • y = h i • cd.s i • y := by rw [hs i, mul_smul]
  rw [e1, hF (h i) ⟨_, hh i⟩ rfl, hφ]
  simp only [ρΓ]
  rw [← Module.End.mul_apply, ← map_mul]
  congr 2
  apply Subtype.ext
  simp only [Submonoid.coe_mul]
  rw [hs i]
  simp only [Subgroup.coe_mul]
  group

variable (e : Γ ≃* SL(2, ℤ))

/-- `thm:welldef`(a) for `τ` over `R`, at cochain level, no hypothesis on `ℓ`. -/
theorem heckeTildeR_indep_cd (cd cd' : CosetData Γ Δ g) (τ : TauR R g e) :
    heckeTildeR (hΓΔ := hΓΔ) ρ e cd' τ = heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ := by
  funext q
  apply LinearMap.ext; intro φ
  apply Subtype.ext
  exact transport_indep_cd ρ cd cd' (FR τ q) (FR_t1 τ q) _ (linR_mem ρ φ)

/-! ### Scaling by a central element -/

variable {z g' : G} (hzc : ∀ x : G, z * x = x * z) (hgz : g = z * g')
include hzc hgz

lemma conj_central (x : G) : g⁻¹ * x * g = g'⁻¹ * x * g' := by
  have h1 : z⁻¹ * x * z = x := by rw [mul_assoc, ← hzc x, ← mul_assoc, inv_mul_cancel, one_mul]
  rw [hgz, mul_inv_rev]
  calc g'⁻¹ * z⁻¹ * x * (z * g') = g'⁻¹ * (z⁻¹ * x * z) * g' := by group
    _ = g'⁻¹ * x * g' := by rw [h1]

variable (hz : z ∈ Δ) in
/-- A coset datum for `g'` is one for `g = z g'`, `z` central in `Δ`: the same `s_j`, `δ_j = z δ'_j`. -/
def cdScale (cd : CosetData Γ Δ g') : CosetData Γ Δ g where
  m := cd.m
  s := cd.s
  δ_mem i := by
    have : ((cd.s i : G))⁻¹ * g = z * (((cd.s i : G))⁻¹ * g') := by
      rw [hgz, ← mul_assoc, ← mul_assoc, hzc]
    rw [this]; exact Δ.mul_mem hz (cd.δ_mem i)
  cosets γ := by
    simp only [conj_central hzc hgz]
    exact cd.cosets γ

/-- The element of `Γ''(g')` with the same underlying element as `h ∈ Γ''(g)`. -/
def ppScale (h : Γpp (Γ := Γ) g) : Γpp (Γ := Γ) g' :=
  ⟨h.1, by
    show g'⁻¹ * ((h.1 : Γ) : G) * g' ∈ Γ
    rw [← conj_central hzc hgz]; exact h.2⟩

lemma fIncl_ppScale (h : Γpp (Γ := Γ) g) : fIncl g' e (ppScale hzc hgz h) = fIncl g e h := rfl

lemma fConj_ppScale (h : Γpp (Γ := Γ) g) : fConj g' e (ppScale hzc hgz h) = fConj g e h := by
  show e (conjHom g' (ppScale hzc hgz h)) = e (conjHom g h)
  congr 1
  apply Subtype.ext
  rw [coe_conjHom, coe_conjHom]
  exact (conj_central hzc hgz _).symm

variable (R) in
/-- A `τ` over `R` for `g'` is one for `g = z g'`, `z` central (the same maps). -/
def tauScale (τ : TauR R g' e) : TauR R g e where
  f n := Rep.ofHom ⟨(τ.f n).hom.toLinearMap, fun h => by
    refine LinearMap.ext fun x => ?_
    have := τ.twist n (ppScale hzc hgz h) x
    rw [fIncl_ppScale, fConj_ppScale] at this
    exact this⟩
  t2 n x := τ.t2 n x
  t3 x := τ.t3 x

lemma FR_tauScale (τ : TauR R g' e) (q : ℤ) : FR (tauScale R e hzc hgz τ) q = FR τ q := by
  rcases q with n | k
  · rfl
  · rfl

omit hzc hgz in
/-- `HmapOf` of a map that is pointwise `c •` another is `c •` its `HmapOf`. -/
lemma HmapOf_smul_of {A B : CochainCx.{0} R} {f f' : GMap A B} (hf : IsChainMap f) (hf' : IsChainMap f') (c : R)
    (q : ℤ) (h : ∀ x, f q x = c • f' q x) : HmapOf f hf q = c • HmapOf f' hf' q := by
  apply LinearMap.ext; intro x
  induction x using Submodule.Quotient.induction_on with
  | _ z =>
    rw [HmapOf_mk, LinearMap.smul_apply, HmapOf_mk, ← Submodule.Quotient.mk_smul]
    congr 1
    exact Subtype.ext (h z.1)

variable (hz : z ∈ Δ) {c : R} (hρz : ρ ⟨z, hz⟩ = c • 1)
include hρz

/-- **The scaling identity** at cochain level: `T̃_{z g'} = c T̃_{g'}` for `z` central with `ρ(z) = c`. -/
theorem heckeTildeR_scale (cd : CosetData Γ Δ g') (τ : TauR R g' e) (q : ℤ) (φ) :
    heckeTildeR (hΓΔ := hΓΔ) ρ e (cdScale hzc hgz hz cd) (tauScale R e hzc hgz τ) q φ =
      c • heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ q φ := by
  apply Subtype.ext; apply AddMonoidHom.ext; intro y
  rw [heckeTildeR_apply, FR_tauScale]
  show _ = c • (heckeTildeR (hΓΔ := hΓΔ) ρ e cd τ q φ).1 y
  rw [heckeTildeR_apply, Finset.smul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hδ : (⟨((cd.s i : G))⁻¹ * g, (cdScale hzc hgz hz cd).δ_mem i⟩ : Δ) =
      ⟨z, hz⟩ * ⟨((cd.s i : G))⁻¹ * g', cd.δ_mem i⟩ := by
    apply Subtype.ext
    simp only [Submonoid.coe_mul]
    rw [hgz, ← mul_assoc, ← mul_assoc, hzc]
  show ρ ⟨((cd.s i : G))⁻¹ * g, (cdScale hzc hgz hz cd).δ_mem i⟩ _ = _
  rw [hδ, map_mul, hρz, Module.End.mul_apply, LinearMap.smul_apply, Module.End.one_apply]
  rfl

end Generic

section Unit

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ}
  {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : Δ →* Module.End R V)
  (e : Γ ≃* SL(2, ℤ))

/-- The coset datum of `Γ 1 Γ = Γ`: one coset, `s = 1`. -/
def cdOne : CosetData Γ Δ (1 : G) where
  m := 1
  s _ := 1
  δ_mem _ := by simp only [OneMemClass.coe_one, inv_one, mul_one]; exact Δ.one_mem
  cosets γ := ⟨0, ⟨γ, by simp only [inv_one, one_mul, mul_one]; exact γ.2, by simp⟩,
    fun _ _ => Subsingleton.elim _ _⟩

lemma fConj_one (h : Γpp (Γ := Γ) (1 : G)) : fConj (1 : G) e h = fIncl (1 : G) e h := by
  show e (conjHom 1 h) = e h
  congr 1
  apply Subtype.ext
  rw [coe_conjHom, inv_one, one_mul, mul_one]

variable (R) in
/-- `τ = id` for `g = 1`. -/
def tauOne : TauR R (1 : G) e where
  f n := Rep.ofHom ⟨LinearMap.id, fun h => by
    refine LinearMap.ext fun x => ?_
    show (Representation.ofMulAction R SL(2, ℤ) (Simp model7 n) (fIncl (1 : G) e h)) x =
      (Representation.ofMulAction R SL(2, ℤ) (Simp model7 n) (fConj (1 : G) e h)) x
    rw [fConj_one]⟩
  t2 _ _ := rfl
  t3 _ := rfl

/-- `T̃_{Γ 1 Γ}` with `τ = id` is the identity on the cochains of degree `n ≥ 0`. -/
theorem heckeTildeR_one (n : ℕ) (φ : (EqC model7 (hΓΔ := hΓΔ) ρ e).X (n : ℤ)) :
    heckeTildeR (hΓΔ := hΓΔ) ρ e cdOne (tauOne R e) (n : ℤ) φ = φ := by
  apply Subtype.ext; apply AddMonoidHom.ext; intro y
  rw [heckeTildeR_apply]
  have h1 : ∀ i, (⟨((cdOne (Δ := Δ) (Γ := Γ)).s i : G)⁻¹ * 1, (cdOne (Δ := Δ) (Γ := Γ)).δ_mem i⟩ : Δ) = 1 :=
    fun i => by apply Subtype.ext; simp [cdOne]
  have h2 : ∀ i, (cdOne (Δ := Δ) (Γ := Γ)).s i = 1 := fun _ => rfl
  refine (Finset.sum_congr rfl fun i _ => ?_).trans (?_ : ∑ _i : Fin 1, linR (R := R) φ.1 (embR (n : ℤ) y) = _)
  · rw [h1 i, map_one, Module.End.one_apply]
    show linR (R := R) φ.1 (embR (n : ℤ) ((cdOne (Δ := Δ) (Γ := Γ)).s i • y)) = _
    rw [h2 i, one_smul]
  · rw [Fin.sum_univ_one]
    exact linR_embR (R := R) (D := model7) φ.1 y

lemma HmapOf_one (n : ℕ) :
    HmapOf (heckeTildeR (hΓΔ := hΓΔ) ρ e cdOne (tauOne R e)) (heckeTildeR_isChainMap ρ cdOne (tauOne R e)) n = 1 := by
  apply LinearMap.ext; intro x
  induction x using Submodule.Quotient.induction_on with
  | _ z =>
    rw [HmapOf_mk, Module.End.one_apply]
    congr 1
    exact Subtype.ext (heckeTildeR_one ρ e n z.1)

end Unit


/-! ### M4-a: existence of the data -/

/-- **M4-a, [GT] `lem:m4-tauR-exists`** ([C] `lem:tauexists`(2), existence over `R`, `6 ∈ R^×`): the augmentation-compatible
chain map `res_{f₁}(C_•(model7) ⊗ R) → res_{f₂}(C_•(model7) ⊗ R)` of `model7_tauexists_2` (at `f₁ = fIncl g`,
`f₂ = fConj g`) is a `TauR`: its components are the maps, `comm` gives (T2), the augmentation gives (T3). -/
theorem tauR_nonempty : tauR_nonempty_stmt := by
  intro R _ h6 g
  have H := (model7_tauexists_2 h6 (fIncl g eQ) (fConj g eQ) (fIncl_injective g eQ) (fConj_injective g eQ)).1
  change ∃ τ : (P₁ g eQ h6).complex ⟶ (P₂ g eQ h6).complex, τ ≫ (P₂ g eQ h6).π = (P₁ g eQ h6).π at H
  obtain ⟨τ, hτ⟩ := H
  refine ⟨⟨fun n => τ.f n, fun n x => ?_, fun x => ?_⟩⟩
  · have hc := τ.comm (n + 1) n
    rw [P₁_d, P₂_d] at hc
    have := congrArg (fun φ => (Rep.Hom.hom φ) x) hc
    simp only [Rep.resFunctor] at this
    exact this
  · have h0 := congrArg (fun φ => HomologicalComplex.Hom.f φ 0) hτ
    simp only [HomologicalComplex.comp_f] at h0
    have h2 := congrArg (fun φ => (Rep.Hom.hom φ) x) h0
    simp only [Rep.hom_comp] at h2
    have e2 : ∀ y, (Rep.Hom.hom ((P₂ g eQ h6).π.f 0)) y = (Rep.Hom.hom ((model7_projRes h6).π.f 0)) y :=
      fun y => rfl
    have e1 : ∀ y, (Rep.Hom.hom ((P₁ g eQ h6).π.f 0)) y = (Rep.Hom.hom ((model7_projRes h6).π.f 0)) y :=
      fun y => rfl
    have h3 := (e2 _).symm.trans (h2.trans (e1 x))
    have e3 : (model7_projRes h6).π.f 0 = repEps model7 R := repAug_f_zero model7 R
    rw [e3] at h3
    exact h3

/-- `Γ'' = Γ ∩ g Γ g⁻¹` has finite index in `Γ = SL(2, ℤ)` for every `g ∈ GL(2, ℚ)` (mathlib `exists_Gamma_le_conj`). -/
theorem Γpp_finiteIndex (g : GLQ) : (Γpp (Γ := ΓQ) g).FiniteIndex := by
  obtain ⟨N, hN, hconj⟩ := CongruenceSubgroup.exists_Gamma_le_conj g⁻¹ 1
  have : NeZero N := ⟨hN⟩
  have hle : CongruenceSubgroup.Gamma N ≤ (Γpp (Γ := ΓQ) g).comap eQ.symm.toMonoidHom := by
    intro x hx
    obtain ⟨y, -, hy⟩ := hconj x hx
    show g⁻¹ * ((eQ.symm x : ΓQ) : GLQ) * g ∈ ΓQ
    rw [coe_eQ_symm]
    rw [inv_inv] at hy
    exact ⟨y, hy⟩
  constructor
  rw [← Subgroup.index_comap_of_surjective _ eQ.symm.surjective]
  exact (Subgroup.finiteIndex_of_le hle).index_ne_zero

/-- **M4-a, [GT] `lem:m4-cosetdata`**: every `g ∈ Δ` has a coset datum (L20-0 §4, the mathlib route: `Γ''` has finite
index by `Γpp_finiteIndex`; `s_i` are the inverses of representatives of the left cosets `Γ/Γ''`, enumerated by
`Fintype.equivFin`; `δ_i = s_i⁻¹ g ∈ Δ` since `Γ ⊆ Δ`). Non-constructive; the operators do not depend on the datum. -/
theorem cosetData_nonempty : cosetData_nonempty_stmt := by
  intro g hg
  classical
  have := Γpp_finiteIndex g
  have : Finite (ΓQ ⧸ Γpp (Γ := ΓQ) g) := Subgroup.finite_quotient_of_finiteIndex
  let _ := Fintype.ofFinite (ΓQ ⧸ Γpp (Γ := ΓQ) g)
  let E := Fintype.equivFin (ΓQ ⧸ Γpp (Γ := ΓQ) g)
  refine ⟨⟨Fintype.card (ΓQ ⧸ Γpp (Γ := ΓQ) g), fun i => (Quotient.out (E.symm i))⁻¹, fun i => ?_, fun γ => ?_⟩⟩
  · rw [Subgroup.coe_inv, inv_inv]
    exact ΔQ.mul_mem (ΓQ_le_ΔQ _) hg
  · obtain ⟨h0, hh0⟩ := QuotientGroup.mk_out_eq_mul (Γpp (Γ := ΓQ) g) γ⁻¹
    refine ⟨E (QuotientGroup.mk γ⁻¹), ⟨γ * Quotient.out (E.symm (E (QuotientGroup.mk γ⁻¹))), ?_, ?_⟩, ?_⟩
    · rw [Equiv.symm_apply_apply]
      have hmem : γ * Quotient.out (QuotientGroup.mk (s := Γpp (Γ := ΓQ) g) γ⁻¹) ∈ Γpp (Γ := ΓQ) g := by
        rw [hh0, mul_inv_cancel_left]; exact h0.2
      exact hmem
    · simp only [Subgroup.coe_mul, Subgroup.coe_inv, mul_inv_cancel_right]
    · rintro j ⟨h, hconj, hγ⟩
      have hmem : h ∈ Γpp (Γ := ΓQ) g := hconj
      have hγ' : γ = h * (Quotient.out (E.symm j))⁻¹ := Subtype.ext (by simpa using hγ)
      apply E.symm.injective
      rw [Equiv.symm_apply_apply, ← QuotientGroup.out_eq' (E.symm j)]
      apply QuotientGroup.eq.mpr
      rw [hγ', mul_inv_rev, inv_inv, ← mul_assoc, inv_mul_cancel, one_mul]
      exact Subgroup.inv_mem _ hmem

section Model

variable {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)

/-- A scalar matrix is central in `GL(2, ℚ)`. -/
lemma central_of_scalar {z : GLQ} {N : ℤ} (hs : IsScalarQ z N) (x : GLQ) : z * x = x * z := by
  apply Units.ext
  have hk : (z : Matrix (Fin 2) (Fin 2) ℚ) = (N : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
    rw [hs, map_zsmul, map_one, ← Int.cast_smul_eq_zsmul ℚ]
  rw [Units.val_mul, Units.val_mul, hk, Matrix.smul_mul, Matrix.mul_smul, Matrix.one_mul, Matrix.mul_one]

/-- `τ = id` for a scalar `g = N · I` (`Γ'' = Γ`, `g⁻¹ h g = h`). -/
def tauScalar {g : GLQ} {N : ℤ} (hs : IsScalarQ g N) : TauR R g eQ :=
  tauScale R eQ (central_of_scalar hs) (mul_one g).symm (tauOne R eQ)

/-- **`prop:diamond` for `τ = id`** ([C]'s scope: no hypothesis on `ℓ`): `g = N · I` with `ρ(g) = c` acts on `H^n` by `c`,
for every coset datum. -/
theorem diamond_id (g : GLQ) (hg : g ∈ ΔQ) (N : ℤ) (c : R) (hs : IsScalarQ g N) (hρ : ρ ⟨g, hg⟩ = c • 1)
    (cd : CosetData ΓQ ΔQ g) (n : ℕ) : Tdc ρ cd (tauScalar hs) n = c • 1 := by
  have hgz : g = g * 1 := (mul_one g).symm
  unfold Tdc
  rw [HmapOf_congr (heckeTildeR_indep_cd ρ eQ (cdScale (central_of_scalar hs) hgz hg cdOne) cd (tauScalar hs)) _
    (heckeTildeR_isChainMap ρ _ _)]
  unfold tauScalar
  rw [HmapOf_smul_of _ (heckeTildeR_isChainMap ρ cdOne (tauOne R eQ)) c (n : ℤ)
    (heckeTildeR_scale ρ eQ (central_of_scalar hs) hgz hg hρ cdOne (tauOne R eQ) (n : ℤ)), HmapOf_one]

/-- **[C] `prop:diamond` on the model** ([GT] `prop:diamond-model`). -/
theorem diamond : diamond_stmt ρ := by
  intro g hg N c hs hρ h6 cd τ n
  unfold Tdc
  rw [heckeTildeR_indep_tau ρ h6 cd τ (tauScalar hs) n]
  exact diamond_id ρ g hg N c hs hρ cd n

/-- **[C] `prop:fullassembly` on the model, the summand** ([GT] `prop:fullassembly-model`). -/
theorem fullassembly : fullassembly_stmt ρ := by
  intro p hp z hz c hs hρ h6 j μ hμ g hgm cd τ τ' n
  have hzj : z ^ j ∈ ΔQ := pow_mem hz j
  have hρj : ρ ⟨z ^ j, hzj⟩ = c ^ j • 1 := by
    have : (⟨z ^ j, hzj⟩ : ΔQ) = ⟨z, hz⟩ ^ j := Subtype.ext (by simp)
    rw [this, map_pow, hρ, smul_pow, one_pow]
  have hzc : ∀ x : GLQ, z ^ j * x = x * z ^ j := fun x => ((Commute.pow_left (central_of_scalar hs x) j)).eq
  have hgz : g = z ^ j * gP p μ hp := by
    apply Units.ext
    rw [Units.val_mul, Units.val_pow_eq_pow_val, hs, hgm, gP, coe_gQ, ← map_pow, ← map_mul]
    congr 1
    rw [smul_pow, one_pow, Matrix.smul_mul, Matrix.one_mul]
  unfold Tdc
  rw [HmapOf_congr (heckeTildeR_indep_cd ρ eQ (cdScale hzc hgz hzj (cdP p μ hp hμ)) cd τ) _
    (heckeTildeR_isChainMap ρ _ _)]
  rw [heckeTildeR_indep_tau ρ h6 _ τ (tauScale R eQ hzc hgz τ') n]
  exact HmapOf_smul_of _ _ (c ^ j) (n : ℤ) (heckeTildeR_scale ρ eQ hzc hgz hzj hρj (cdP p μ hp hμ) τ' (n : ℤ))


/-- **M4-a: `T_p`, choice-free** (`T_{Γ diag(1,p) Γ}` on `cdP p 1` and a chosen `τ`; by `Tp_eq` independent of the choice). -/
def Tp (h6 : IsUnit (6 : R)) (p : ℕ) (hp : p.Prime) (n : ℕ) : Module.End R (HM ρ n) :=
  Tdc ρ (cdP p 1 hp le_rfl) (Classical.choice (tauR_nonempty R h6 (gP p 1 hp))) n

/-- `Tp` is `Tdc` at any `τ` (`heckeTildeR_indep_tau`). -/
theorem Tp_eq (h6 : IsUnit (6 : R)) (p : ℕ) (hp : p.Prime) (τ : TauR R (gP p 1 hp) eQ) (n : ℕ) :
    Tp ρ h6 p hp n = Tdc ρ (cdP p 1 hp le_rfl) τ n :=
  heckeTildeR_indep_tau ρ h6 _ _ τ n

/-- **M4-a: `T_{Γ g Γ}` for `g ∈ Δ`, choice-free** (a chosen coset datum and `τ`; by `TdcOf_eq` independent of both). -/
def TdcOf (h6 : IsUnit (6 : R)) (g : GLQ) (hg : g ∈ ΔQ) (n : ℕ) : Module.End R (HM ρ n) :=
  Tdc ρ (Classical.choice (cosetData_nonempty g hg)) (Classical.choice (tauR_nonempty R h6 g)) n

/-- `TdcOf` is `Tdc` at any coset datum and any `τ` (`heckeTildeR_indep_cd`, `heckeTildeR_indep_tau`). -/
theorem TdcOf_eq (h6 : IsUnit (6 : R)) (g : GLQ) (hg : g ∈ ΔQ) (cd : CosetData ΓQ ΔQ g) (τ : TauR R g eQ) (n : ℕ) :
    TdcOf ρ h6 g hg n = Tdc ρ cd τ n := by
  unfold TdcOf Tdc
  rw [HmapOf_congr (heckeTildeR_indep_cd ρ eQ cd (Classical.choice (cosetData_nonempty g hg)) _) _
    (heckeTildeR_isChainMap ρ _ _)]
  exact heckeTildeR_indep_tau ρ h6 _ _ τ n

end Model

end GL2.M4

end
