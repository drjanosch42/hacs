import GL2.Model.H4

/-!
# M2-e (L18-2): `Ψ`, [C] `thm:explicit_complex_equiv`(1) over the `X_bulk` descriptor

[GT] `def:relcx`, `thm:psi`, mirroring [C] v1.17.0 `def:naivecochains`, `def:eq_cochains`, `lem:eqgroupoid`,
`lemma:d_maps_C_eq`, `thm:explicit_complex_equiv`(1), `eq:psiinverse`.

**Setting.** `G` any group, `Γ ≤ G` a subgroup identified with `SL(2, ℤ)` by `e : Γ ≃* SL(2, ℤ)`, `Δ ⊇ Γ` a
submonoid acting on the coefficients `V` through `ρ : Δ →* End_R(V)` (the cone layer's conventions, so that the same
`eqCochains` carries the Hecke operator of `GL2.Model.WellDef`).  `toGChainCxVia D e` is `X_bulk`'s chain complex with
`Γ` acting through `e`; `toGChainCxOn H` is the case `e = H.subtype` (`toGChainCxOn_eq`).  The matrix local system is
`L = (V, ρ_L)`, `ρ_L(t) = ρ(e⁻¹ t)`.

**The relative complex** `relCx` (`C^q_eq(K, Rel; L)` of [C]): functions `f : V_q → V` on the model simplices with
`f(σ) = sgn(σ, σ', t) · ρ_L(t) f(σ')` for **every** transporter `t ∈ S(σ, σ')` (the descriptor's transporter sets are the
full coincidence groupoid, so [C]'s FD4 / `lem:eqgroupoid` is built in; the oriented sign is [C]'s stored orientation),
with `(d f)(τ) = ∑_j ε_j ρ_L(h_j) f(σ_j)` for the signed faces `∂_j ι(τ) = ε_j h_j ι(σ_j)` of the descriptor.

*Restatement note.* [C] `eq:naivecoboundary` reads `∑_t (-1)^t f(∂_t σ)` because there every face of a simplex of `K` is
again in `K` (`h_j = 1`); the descriptor allows a face to be a translate `h_j ι(σ_j)` of a model simplex, so the
coboundary carries `ρ_L(h_j)`.  For `model7` all `h_j = 1` and the formula is [C]'s.  `d` preserving the constraint
([C] `lemma:d_maps_C_eq`) and `d² = 0` are obtained here *through* `Ψ` (`dF_mem`, `dF_dF`).

**Theorem** (`psiEquiv`, `psi_isChainMap`, `phi_psi`, `psi_phi`, `psi_natural`): `Ψ φ = (σ ↦ φ(ι σ))` is an isomorphism
of cochain complexes `Hom_Γ(C_•(X_bulk), V) ≅ relCx`, natural in `L`, with inverse `Φ f (γ ι σ) = ρ_L(γ) f(σ)`
(`eq:psiinverse`).  No freeness, contractibility or field hypothesis.
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open GL2.Ku GL2.Cone

namespace XBulk

variable (D : XBulkDatum)

/-- `X_bulk`'s chain complex with `Γ` acting through a homomorphism `e : Γ → SL(2, ℤ)`. -/
def toGChainCxVia {Γ : Type} [Group Γ] (e : Γ →* SL(2, ℤ)) : GChainCx.{0} Γ where
  X q := chains D q
  act _ := DistribMulAction.compHom _ e
  bd q := (bd D q).toAddMonoidHom
  bd_equiv q γ x := bd_equiv D q (e γ) x
  bd_bd q x := bd_bd D q x
  aug := (aug D).toAddMonoidHom
  aug_bd x := aug_bd D x
  aug_equiv γ x := aug_equiv D (e γ) x

lemma toGChainCxOn_eq (H : Subgroup SL(2, ℤ)) : toGChainCxOn D H = toGChainCxVia D H.subtype := rfl

/-- Two additive maps out of the chains agree if they agree on the generators `genQ (γ, σ)`. -/
lemma chains_hom_ext {q : ℤ} {A : Type*} [AddCommGroup A] {u v : chains D q →+ A}
    (h : ∀ a : SL(2, ℤ) × D.V q, u (genQ D q a) = v (genQ D q a)) : u = v := by
  ext y
  induction y using Submodule.Quotient.induction_on with
  | _ F =>
    induction F using Finsupp.induction_linear with
    | zero => rw [Submodule.Quotient.mk_zero, map_zero, map_zero]
    | add f g hf hg => rw [Submodule.Quotient.mk_add, map_add, map_add, hf, hg]
    | single a c =>
      have : (Submodule.Quotient.mk (Finsupp.single a c) : chains D q) = c • genQ D q a := by
        rw [genQ, ← Submodule.Quotient.mk_smul]; congr 1; simp [gen, Finsupp.smul_single]
      rw [this, map_zsmul, map_zsmul, h]

/-- `[γ, σ, ε] = ε · γ · [1, σ]`. -/
lemma mk_gen_eq_smul {q : ℤ} (x : OrPt (D.V q)) :
    (Submodule.Quotient.mk (gen x) : chains D q) = ((x.2.2 : ℤˣ) : ℤ) • (x.1 • genQ D q (1, x.2.1)) := by
  rw [genQ, smul_mk, actF_gen, ← Submodule.Quotient.mk_smul]
  congr 1
  simp [gen, OrPt.lt, Finsupp.smul_single]

lemma genQ_eq_smul {q : ℤ} (a : SL(2, ℤ) × D.V q) : genQ D q a = a.1 • genQ D q (1, a.2) := by
  rw [smul_genQ, mul_one]

section Psi

variable {G : Type} [Group G] {Γ : Subgroup G} {Δ : Submonoid G} {hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ}
  {R : Type*} [CommRing R] {V : Type} [AddCommGroup V] [Module R V]
  (ρ : Δ →* Module.End R V) (e : Γ ≃* SL(2, ℤ))

/-- The `Γ`-complex `C_•(X_bulk)` with `Γ` acting through `e`. -/
abbrev Yc : GChainCx.{0} Γ := toGChainCxVia D e.toMonoidHom

variable (hΓΔ) in
/-- The matrix local system's representation `ρ_L(t) = ρ(e⁻¹ t)`. -/
def ρL (t : SL(2, ℤ)) : Module.End R V := ρΓ Γ Δ hΓΔ ρ (e.symm t)

omit [CommRing R] in
lemma ρΓ_mul' {R : Type*} [CommRing R] [Module R V] (ρ : Δ →* Module.End R V) (a b : Γ) :
    ρΓ Γ Δ hΓΔ ρ (a * b) = ρΓ Γ Δ hΓΔ ρ a * ρΓ Γ Δ hΓΔ ρ b := by
  unfold ρΓ; rw [← map_mul]; rfl

lemma ρΓ_one' : ρΓ Γ Δ hΓΔ ρ 1 = 1 := by
  unfold ρΓ; rw [← map_one ρ]; rfl

lemma ρL_mul (s t : SL(2, ℤ)) : ρL hΓΔ ρ e (s * t) = ρL hΓΔ ρ e s * ρL hΓΔ ρ e t := by
  unfold ρL; rw [map_mul, ρΓ_mul']

lemma ρL_one : ρL hΓΔ ρ e 1 = 1 := by
  unfold ρL; rw [map_one, ρΓ_one']

lemma ρL_apply_e (γ : Γ) : ρL hΓΔ ρ e (e γ) = ρΓ Γ Δ hΓΔ ρ γ := by
  unfold ρL; rw [MulEquiv.symm_apply_apply]

/-- The equivariant cochains `Hom_Γ(C_q(X_bulk), V)`. -/
abbrev EqC := eqCochains Γ Δ hΓΔ ρ (Yc D e)

variable (hΓΔ) in
/-- `ρ_L`-equivariance of an additive map out of the chains. -/
def IsEqv {q : ℤ} (φ : chains D q →+ V) : Prop := ∀ (t : SL(2, ℤ)) (y : chains D q), φ (t • y) = ρL hΓΔ ρ e t (φ y)

/-- Membership in `Hom_Γ` is `ρ_L`-equivariance for the `SL(2, ℤ)`-action. -/
lemma mem_eqHom_iff {q : ℤ} (φ : chains D q →+ V) :
    φ ∈ eqHom Γ Δ hΓΔ ρ (Yc D e) q ↔ IsEqv D hΓΔ ρ e φ := by
  constructor
  · intro h t y
    have h1 := h (e.symm t) y
    have hs : t • y = e.toMonoidHom (e.symm t) • y := by simp
    exact (congrArg φ hs).trans h1
  · intro h γ y
    have h1 := h (e γ) y
    rw [ρL_apply_e] at h1
    exact h1

/-- `φ[γ, σ, ε] = ε ρ_L(γ) φ[1, σ]`. -/
lemma isEqv_gen {q : ℤ} {φ : chains D q →+ V} (hφ : IsEqv D hΓΔ ρ e φ) (x : OrPt (D.V q)) :
    φ (Submodule.Quotient.mk (gen x)) = ((x.2.2 : ℤˣ) : ℤ) • ρL hΓΔ ρ e x.1 (φ (genQ D q (1, x.2.1))) := by
  rw [mk_gen_eq_smul, map_zsmul, hφ]

/-! ### The relative complex -/

variable (hΓΔ) in
/-- **`def:eq_cochains`** (`eq:equivariance`): the functions `f : V_q → V` with `f σ = sgn · ρ_L(t) f σ'` for every
transporter `t ∈ S(σ, σ')`. -/
def relSubm (q : ℤ) : Submodule R (D.V q → V) where
  carrier := {f | ∀ σ σ' t, t ∈ (D.M q).S σ σ' → f σ = (((D.M q).sgn σ σ' t : ℤˣ) : ℤ) • ρL hΓΔ ρ e t (f σ')}
  zero_mem' := by intro σ σ' t _; simp
  add_mem' := by
    intro f f' hf hf' σ σ' t ht
    simp only [Pi.add_apply, map_add, smul_add, ← hf σ σ' t ht, ← hf' σ σ' t ht]
  smul_mem' := by
    intro c f hf σ σ' t ht
    simp only [Pi.smul_apply, map_smul, hf σ σ' t ht]
    rw [smul_comm]

variable (hΓΔ) in
/-- The coboundary `(d f)(τ) = ∑_j ε_j ρ_L(h_j) f(σ_j)` on all functions `V_q → V`. -/
def dF (q : ℤ) : (D.V q → V) →ₗ[R] (D.V (q + 1) → V) where
  toFun f τ := ∑ j, (((D.face q τ j).2.2 : ℤˣ) : ℤ) • ρL hΓΔ ρ e (D.face q τ j).1 (f (D.face q τ j).2.1)
  map_add' f f' := by ext τ; simp [Finset.sum_add_distrib, smul_add]
  map_smul' c f := by
    ext τ
    simp only [Pi.smul_apply, map_smul, RingHom.id_apply, Finset.smul_sum]
    exact Finset.sum_congr rfl fun j _ => smul_comm _ _ _

lemma dF_apply (q : ℤ) (f : D.V q → V) (τ : D.V (q + 1)) :
    dF D hΓΔ ρ e q f τ = ∑ j, (((D.face q τ j).2.2 : ℤˣ) : ℤ) • ρL hΓΔ ρ e (D.face q τ j).1 (f (D.face q τ j).2.1) :=
  rfl

/-! ### `Ψ` and `Φ` on functions -/

/-- `(Ψ φ)(σ) = φ(ι σ)`. -/
def psiF {q : ℤ} (φ : chains D q →+ V) : D.V q → V := fun σ => φ (genQ D q (1, σ))

/-- **`Ψ` lands in the relative cochains** ([C] `thm:explicit_complex_equiv`(1), first step). -/
theorem psiF_mem {q : ℤ} {φ : chains D q →+ V} (hφ : IsEqv D hΓΔ ρ e φ) : psiF D φ ∈ relSubm D hΓΔ ρ e q := by
  intro σ σ' t ht
  have hr : OrRel (D.M q) ((1, σ, 1) : OrPt (D.V q)) (t, σ', (D.M q).sgn σ σ' t) := by
    refine ⟨by simpa using ht, ?_⟩
    simp only [inv_one, one_mul]
    rw [sgn_sq]
  have := mk_gen_eq (D.M q) hr
  rw [Submodule.mkQ_apply, Submodule.mkQ_apply] at this
  unfold psiF
  rw [genQ, this, isEqv_gen D ρ e hφ]

/-- **`Ψ` commutes with the coboundaries**: `d (Ψ φ) = Ψ (φ ∘ ∂)`. -/
theorem psiF_d {q : ℤ} {φ : chains D q →+ V} (hφ : IsEqv D hΓΔ ρ e φ) :
    dF D hΓΔ ρ e q (psiF D φ) = psiF D (φ.comp (bd D q).toAddMonoidHom) := by
  ext τ
  rw [dF_apply]
  show _ = φ (bd D q (Submodule.Quotient.mk (gen ((1, τ, 1) : OrPt (D.V (q + 1))))))
  rw [bd_mk, bdF_gen, Units.val_one, one_smul, quot_mk_sum, map_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [isEqv_gen D ρ e hφ]
  simp only [OrPt.lt, one_mul]
  rfl

/-- `Φ f` on the free module: `(γ, σ) ↦ ρ_L(γ) f(σ)` (`eq:psiinverse`). -/
def phiFree {q : ℤ} (f : D.V q → V) : (SL(2, ℤ) × D.V q →₀ ℤ) →ₗ[ℤ] V :=
  Finsupp.linearCombination ℤ (fun x => ρL hΓΔ ρ e x.1 (f x.2))

lemma phiFree_gen {q : ℤ} (f : D.V q → V) (x : OrPt (D.V q)) :
    phiFree D ρ e (hΓΔ := hΓΔ) f (gen x) = ((x.2.2 : ℤˣ) : ℤ) • ρL hΓΔ ρ e x.1 (f x.2.1) := by
  simp [phiFree, gen, Finsupp.linearCombination_single]

/-- **`Φ` is well defined** ([C]: "this step is the whole content of the theorem"): `Φ f` kills the oriented relations,
because `f` satisfies the constraint for every transporter. -/
lemma relSub_le_phiFree {q : ℤ} {f : D.V q → V} (hf : f ∈ relSubm D hΓΔ ρ e q) :
    relSub (D.M q) ≤ LinearMap.ker (phiFree D ρ e (hΓΔ := hΓΔ) f) := by
  rw [relSub, Submodule.span_le]
  rintro _ ⟨x, y, h, rfl⟩
  simp only [SetLike.mem_coe, LinearMap.mem_ker, map_sub, phiFree_gen, sub_eq_zero]
  rw [hf x.2.1 y.2.1 _ h.1, map_zsmul, ← Module.End.mul_apply, ← ρL_mul, mul_inv_cancel_left, smul_smul,
    h.2, Units.val_mul]
  congr 1
  rw [mul_comm, ← mul_assoc, ← Units.val_mul, sgn_sq, Units.val_one, one_mul]

/-- `Φ f : C_q(X_bulk) → V`. -/
def phiF {q : ℤ} {f : D.V q → V} (hf : f ∈ relSubm D hΓΔ ρ e q) : chains D q →+ V :=
  ((relSub (D.M q)).liftQ (phiFree D ρ e (hΓΔ := hΓΔ) f) (relSub_le_phiFree D ρ e hf)).toAddMonoidHom

lemma phiF_genQ {q : ℤ} {f : D.V q → V} (hf : f ∈ relSubm D hΓΔ ρ e q) (a : SL(2, ℤ) × D.V q) :
    phiF D ρ e hf (genQ D q a) = ρL hΓΔ ρ e a.1 (f a.2) := by
  show phiFree D ρ e (hΓΔ := hΓΔ) f (gen _) = _
  rw [phiFree_gen, Units.val_one, one_smul]

/-- **`Φ f` is equivariant.** -/
theorem phiF_isEqv {q : ℤ} {f : D.V q → V} (hf : f ∈ relSubm D hΓΔ ρ e q) : IsEqv D hΓΔ ρ e (phiF D ρ e hf) := by
  intro t y
  have := chains_hom_ext D (u := (phiF D ρ e hf).comp (DistribSMul.toAddMonoidHom _ t))
    (v := (ρL hΓΔ ρ e t).toAddMonoidHom.comp (phiF D ρ e hf)) (fun a => by
      simp only [AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply, LinearMap.toAddMonoidHom_coe,
        smul_genQ, phiF_genQ, ρL_mul, Module.End.mul_apply])
  exact DFunLike.congr_fun this y

/-- `Ψ Φ = id`. -/
theorem psiF_phiF {q : ℤ} {f : D.V q → V} (hf : f ∈ relSubm D hΓΔ ρ e q) : psiF D (phiF D ρ e hf) = f := by
  ext σ
  rw [psiF, phiF_genQ, ρL_one, Module.End.one_apply]

/-- `Φ Ψ = id`. -/
theorem phiF_psiF {q : ℤ} {φ : chains D q →+ V} (hφ : IsEqv D hΓΔ ρ e φ) :
    phiF D ρ e (psiF_mem D ρ e hφ) = φ := by
  apply chains_hom_ext
  intro a
  rw [phiF_genQ, genQ_eq_smul D a, hφ]
  rfl

lemma isEqv_comp_bd {q : ℤ} {φ : chains D q →+ V} (hφ : IsEqv D hΓΔ ρ e φ) :
    IsEqv D hΓΔ ρ e (φ.comp (bd D q).toAddMonoidHom) := by
  intro t y
  show φ (bd D q (t • y)) = _
  rw [bd_equiv, hφ]
  rfl

/-- **[C] `lemma:d_maps_C_eq`**, here a consequence of `Ψ`: `d` preserves the relative cochains. -/
theorem dF_mem {q : ℤ} {f : D.V q → V} (hf : f ∈ relSubm D hΓΔ ρ e q) :
    dF D hΓΔ ρ e q f ∈ relSubm D hΓΔ ρ e (q + 1) := by
  have h := phiF_isEqv D ρ e hf
  rw [← psiF_phiF D ρ e hf, psiF_d D ρ e h]
  exact psiF_mem D ρ e (isEqv_comp_bd D ρ e h)

/-- **`d² = 0` on the relative cochains** (through `Ψ`). -/
theorem dF_dF {q : ℤ} {f : D.V q → V} (hf : f ∈ relSubm D hΓΔ ρ e q) :
    dF D hΓΔ ρ e (q + 1) (dF D hΓΔ ρ e q f) = 0 := by
  have h := phiF_isEqv D ρ e hf
  rw [← psiF_phiF D ρ e hf, psiF_d D ρ e h, psiF_d D ρ e (isEqv_comp_bd D ρ e h)]
  ext τ
  simp only [psiF, AddMonoidHom.comp_apply, LinearMap.toAddMonoidHom_coe, bd_bd, map_zero, Pi.zero_apply]

/-! ### The cochain complexes and the isomorphism -/

variable (hΓΔ) in
/-- **The relative complex** `C^•_eq(K, Rel; L)` over the descriptor. -/
def relCx : CochainCx.{0} R where
  X q := relSubm D hΓΔ ρ e q
  d q := (dF D hΓΔ ρ e q).restrict fun _ hf => dF_mem D ρ e hf
  d_comp_d q := by
    ext ⟨f, hf⟩ : 1
    exact Subtype.ext (dF_dF D ρ e hf)

lemma eqC_isEqv {q : ℤ} (φ : (EqC D ρ e (hΓΔ := hΓΔ)).X q) : IsEqv D hΓΔ ρ e φ.1 :=
  (mem_eqHom_iff D ρ e φ.1).1 φ.2

/-- **`Ψ` lands in `C^q_eq`** for every equivariant cochain (the L18-0 name of `psiF_mem`). -/
theorem psi_mem {q : ℤ} (φ : (EqC D ρ e (hΓΔ := hΓΔ)).X q) : psiF D φ.1 ∈ relSubm D hΓΔ ρ e q :=
  psiF_mem D ρ e (eqC_isEqv D ρ e φ)

/-- **`Ψ`** as a graded `R`-linear map `Hom_Γ(C_q(X_bulk), V) → C^q_eq`. -/
def psi : GMap (EqC D ρ e (hΓΔ := hΓΔ)) (relCx D hΓΔ ρ e) := fun _ =>
  { toFun := fun φ => ⟨psiF D φ.1, psiF_mem D ρ e (eqC_isEqv D ρ e φ)⟩
    map_add' := fun _ _ => rfl
    map_smul' := fun _ _ => rfl }

/-- **`Φ`** (`eq:psiinverse`) as a graded `R`-linear map. -/
def phi : GMap (relCx D hΓΔ ρ e) (EqC D ρ e (hΓΔ := hΓΔ)) := fun q =>
  { toFun := fun f => ⟨phiF D ρ e f.2, (mem_eqHom_iff D ρ e _).2 (phiF_isEqv D ρ e f.2)⟩
    map_add' := fun f f' => by
      apply Subtype.ext
      apply chains_hom_ext
      intro a
      show phiF D ρ e (f + f').2 (genQ D q a) = phiF D ρ e f.2 (genQ D q a) + phiF D ρ e f'.2 (genQ D q a)
      simp only [phiF_genQ]
      exact map_add (ρL hΓΔ ρ e a.1) _ _
    map_smul' := fun c f => by
      apply Subtype.ext
      apply chains_hom_ext
      intro a
      show phiF D ρ e (c • f).2 (genQ D q a) = c • phiF D ρ e f.2 (genQ D q a)
      simp only [phiF_genQ]
      exact map_smul (ρL hΓΔ ρ e a.1) c _ }

/-- **`Ψ` is a cochain map.** -/
theorem psi_isChainMap : IsChainMap (psi D ρ e (hΓΔ := hΓΔ)) := by
  intro q
  apply LinearMap.ext; intro φ
  exact Subtype.ext (psiF_d D ρ e (eqC_isEqv D ρ e φ))

theorem psi_phi (q : ℤ) : (psi D ρ e (hΓΔ := hΓΔ) q).comp (phi D ρ e q) = LinearMap.id := by
  apply LinearMap.ext; intro f
  exact Subtype.ext (psiF_phiF D ρ e f.2)

theorem phi_psi (q : ℤ) : (phi D ρ e (hΓΔ := hΓΔ) q).comp (psi D ρ e q) = LinearMap.id := by
  apply LinearMap.ext; intro φ
  exact Subtype.ext (phiF_psiF D ρ e (eqC_isEqv D ρ e φ))

/-- **[C] `thm:explicit_complex_equiv`(1)**: `Ψ` is an isomorphism in every degree. -/
def psiEquiv (q : ℤ) : (EqC D ρ e (hΓΔ := hΓΔ)).X q ≃ₗ[R] (relCx D hΓΔ ρ e).X q :=
  LinearEquiv.ofLinearMap (psi D ρ e q) (phi D ρ e q) (psi_phi D ρ e q) (phi_psi D ρ e q)

/-- `Φ` is a cochain map too (the inverse of a cochain isomorphism). -/
theorem phi_isChainMap : IsChainMap (phi D ρ e (hΓΔ := hΓΔ)) := by
  intro q
  apply LinearMap.ext; intro f
  apply (psiEquiv D ρ e (q + 1)).injective
  have h1 := LinearMap.congr_fun (psi_phi D ρ e (hΓΔ := hΓΔ) q) f
  have h2 := LinearMap.congr_fun (psi_phi D ρ e (hΓΔ := hΓΔ) (q + 1)) ((relCx D hΓΔ ρ e).d q f)
  simp only [LinearMap.comp_apply, LinearMap.id_apply] at h1 h2
  show psi D ρ e (q + 1) _ = psi D ρ e (q + 1) (phi D ρ e (q + 1) _)
  rw [h2, LinearMap.comp_apply, ← (psi_isChainMap D ρ e).apply, h1]

/-! ### Naturality in `L` -/

section Natural

variable {V' : Type} [AddCommGroup V'] [Module R V'] (ρ' : Δ →* Module.End R V') (u : V →ₗ[R] V')

variable (hΓΔ) in
/-- A morphism of local systems: `u ρ(γ) = ρ'(γ) u` for `γ ∈ Γ`. -/
def IsLSHom : Prop := ∀ γ : Γ, u ∘ₗ ρΓ Γ Δ hΓΔ ρ γ = ρΓ Γ Δ hΓΔ ρ' γ ∘ₗ u

variable {ρ ρ' u}

lemma IsLSHom.ρL_comm (hu : IsLSHom hΓΔ ρ ρ' u) (t : SL(2, ℤ)) (v : V) :
    u (ρL hΓΔ ρ e t v) = ρL hΓΔ ρ' e t (u v) :=
  LinearMap.congr_fun (hu (e.symm t)) v

/-- `u_*` on the equivariant cochains. -/
def eqPost (hu : IsLSHom hΓΔ ρ ρ' u) : GMap (EqC D ρ e (hΓΔ := hΓΔ)) (EqC D ρ' e (hΓΔ := hΓΔ)) := fun q =>
  { toFun := fun φ => ⟨u.toAddMonoidHom.comp φ.1, fun γ y => by
      simp only [AddMonoidHom.comp_apply, LinearMap.toAddMonoidHom_coe]
      rw [φ.2 γ y]; exact LinearMap.congr_fun (hu γ) _⟩
    map_add' := fun φ ψ => by apply Subtype.ext; ext y; exact map_add u _ _
    map_smul' := fun c φ => by apply Subtype.ext; ext y; exact map_smul u c _ }

/-- `u_*` on the relative cochains. -/
def relPost (hu : IsLSHom hΓΔ ρ ρ' u) : GMap (relCx D hΓΔ ρ e) (relCx D hΓΔ ρ' e) := fun q =>
  { toFun := fun f => ⟨u ∘ f.1, fun σ σ' t ht => by
      simp only [Function.comp_apply]
      rw [f.2 σ σ' t ht, map_zsmul, hu.ρL_comm]⟩
    map_add' := fun f f' => by apply Subtype.ext; ext σ; exact map_add u _ _
    map_smul' := fun c f => by apply Subtype.ext; ext σ; exact map_smul u c _ }

theorem eqPost_isChainMap (hu : IsLSHom hΓΔ ρ ρ' u) : IsChainMap (eqPost D e hu) := fun _ => rfl

theorem relPost_isChainMap (hu : IsLSHom hΓΔ ρ ρ' u) : IsChainMap (relPost D e hu) := by
  intro q
  apply LinearMap.ext; intro f
  apply Subtype.ext; ext τ
  show (dF D hΓΔ ρ' e q (u ∘ f.1)) τ = u (dF D hΓΔ ρ e q f.1 τ)
  rw [dF_apply, dF_apply, map_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [map_zsmul, hu.ρL_comm]
  rfl

/-- **`Ψ` is natural in `L`.** -/
theorem psi_natural (hu : IsLSHom hΓΔ ρ ρ' u) (q : ℤ) :
    (psi D ρ' e q).comp (eqPost D e hu q) = (relPost D e hu q).comp (psi D ρ e q) := rfl

end Natural

end Psi

end XBulk

end GL2.Model

end
