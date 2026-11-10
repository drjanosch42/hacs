import GL2.Model.H4
import GL2.Model.Psi
import Mathlib.Algebra.Homology.QuasiIso
import Mathlib.Algebra.Homology.SingleHomology
import Mathlib.Algebra.Homology.ShortComplex.ModuleCat
import Mathlib.RepresentationTheory.Rep.Basic

/-!
# M3-c (L19-2): the bridge `C_•(X_bulk) ⊗ R → R`, as a complex of permutation representations

[GT] `lem:xbulk-orient`, `def:xbulk-repcx`, `thm:xbulk-repaug`, mirroring [C] v1.18.0 `rmk:orbifold` ("`C_q(X) ⊗ R` is a
direct sum of the `R[Γ/Γ_σ]`") and `H4`(i)--(ii).

**Orientation** (`orient`, `chainsEquivPerm`).  Every `q`-simplex `s` of `X_bulk` is `γ ι(σ̂)` with `σ̂` an orbit
representative; its oriented generator `orient s = [γ t, σ̂]` (`t` a fixed transporter `σ → σ̂`) is well defined because a
stabiliser element of `ι(σ̂)` preserves the orientation (H4(i), `noInv`).  So the oriented chains are the permutation
module: `chainsEquivPerm : C_q(X_bulk) ≃ ℤ[Simp_q]`, `SL(2, ℤ)`-equivariant, with `[γ, σ] ↦ sgn(σ, σ̂, t) · e_{γ ι σ}`
(the sign is read off the oriented relation, not chosen).

**The complex** (`repCx`).  Over any commutative ring `R`, `repCx D R` is the chain complex of representations
`R[Simp_2] → R[Simp_1] → R[Simp_0]` (and `R[Simp_q] = 0` for `q ≥ 3`), with the boundary transported from `bd` through
`chainsEquivPerm` and base-changed along `ℤ → R`; its terms are the permutation representations by construction
(`repCx_iso_perm` is `Iso.refl`).  `repAug : repCx → R` is the augmentation.

**Quasi-isomorphism** (`repAug_quasiIso`).  If `X_bulk` is acyclic (`Acyclic D`, H4(ii)) and has a vertex, then the
augmented `ℤ`-complex of free modules is exact, hence split: `contractZ` builds a contracting homotopy `z, s₀, s₁`
degree by degree from projectivity of the free modules.  Its base change along `ℤ → R` is a contracting homotopy of the
augmented `R`-complex, so `repAug` is a quasi-isomorphism in `Rep R SL(2, ℤ)` (exactness is checked on underlying modules).
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open GL2.Ku

namespace XBulk

variable (D : XBulkDatum)

/-! ### The orientation of the orbits -/

section Orient

variable (q : ℤ)

/-- A fixed transporter `t_σ ∈ S(σ, σ̂)` to the orbit representative. -/
def tRep (σ : D.V q) : SL(2, ℤ) := ((D.M q).rep_mem σ).choose

lemma tRep_mem (σ : D.V q) : tRep D q σ ∈ (D.M q).S σ ((D.M q).rep σ) := ((D.M q).rep_mem σ).choose_spec

lemma rep_rep (σ : D.V q) : (D.M q).rep ((D.M q).rep σ) = (D.M q).rep σ :=
  ((D.M q).rep_eq ((D.M q).rep_mem σ)).symm

/-- The oriented generator of the simplex of `(γ, σ)`: `[γ t_σ, σ̂]`. -/
def orientF (x : SL(2, ℤ) × D.V q) : chains D q := genQ D q (x.1 * tRep D q x.2, (D.M q).rep x.2)

lemma orientF_rel {x y : SL(2, ℤ) × D.V q} (h : Ku.Rel (D.M q).toKeyModel x y) : orientF D q x = orientF D q y := by
  have hr : (D.M q).rep x.2 = (D.M q).rep y.2 := (D.M q).rep_eq ⟨_, h⟩
  have hu : (tRep D q x.2)⁻¹ * (x.1⁻¹ * y.1) * tRep D q y.2 ∈ (D.M q).S ((D.M q).rep x.2) ((D.M q).rep y.2) :=
    (D.M q).mul_mem ((D.M q).mul_mem ((D.M q).inv_mem (tRep_mem D q x.2)) h) (tRep_mem D q y.2)
  have e : (x.1 * tRep D q x.2)⁻¹ * (y.1 * tRep D q y.2) = (tRep D q x.2)⁻¹ * (x.1⁻¹ * y.1) * tRep D q y.2 := by
    group
  unfold orientF genQ
  have := mk_gen_eq (D.M q) (x := (x.1 * tRep D q x.2, (D.M q).rep x.2, 1))
    (y := (y.1 * tRep D q y.2, (D.M q).rep y.2, 1)) ⟨by simp only; rw [e]; exact hu, by
      simp only
      rw [e, mul_one]
      revert hu
      rw [hr]
      intro hu
      rw [(D.M q).noInv hu]⟩
  simpa only [Submodule.mkQ_apply] using this

/-- **The orientation** `orient s`: the oriented generator of the simplex `s` (well defined by H4(i)). -/
def orient (s : Simp D q) : chains D q := Quot.lift (orientF D q) (fun _ _ h => orientF_rel D q h) s

lemma orient_mk (x : SL(2, ℤ) × D.V q) : orient D q (SimpM.mk (D.M q) x) = orientF D q x := rfl

lemma orient_smul (γ : SL(2, ℤ)) (s : Simp D q) : orient D q (γ • s) = γ • orient D q s := by
  induction s using Quot.ind with
  | _ x =>
    show orient D q (SimpM.mk _ (γ * x.1, x.2)) = γ • orientF D q x
    rw [orient_mk, orientF, orientF, smul_genQ, mul_assoc]

/-- The orientation sign `sgn(σ, σ̂, t_σ)`. -/
def sgnRep (σ : D.V q) : ℤ := (((D.M q).sgn σ ((D.M q).rep σ) (tRep D q σ) : ℤˣ) : ℤ)

lemma sgnRep_sq (σ : D.V q) : sgnRep D q σ * sgnRep D q σ = 1 := by
  unfold sgnRep; rw [← Units.val_mul, sgn_sq, Units.val_one]

lemma sgnRep_rep (σ : D.V q) : sgnRep D q ((D.M q).rep σ) = 1 := by
  have h := tRep_mem D q ((D.M q).rep σ)
  rw [rep_rep] at h
  unfold sgnRep
  have h2 := (D.M q).noInv h
  rw [rep_rep] at *
  rw [h2, Units.val_one]

/-- **The oriented generator is `±` the orientation of its simplex**: `[γ, σ] = sgn(σ, σ̂, t_σ) · orient(γ ι σ)`. -/
lemma genQ_eq_orient (x : SL(2, ℤ) × D.V q) : genQ D q x = sgnRep D q x.2 • orient D q (SimpM.mk (D.M q) x) := by
  rw [orient_mk, orientF]
  set s := (D.M q).sgn x.2 ((D.M q).rep x.2) (tRep D q x.2)
  have hr : OrRel (D.M q) ((x.1, x.2, 1) : OrPt (D.V q)) (x.1 * tRep D q x.2, (D.M q).rep x.2, s) := by
    refine ⟨by simpa using tRep_mem D q x.2, ?_⟩
    simp only [inv_mul_cancel_left]
    rw [sgn_sq]
  have := mk_gen_eq (D.M q) hr
  rw [Submodule.mkQ_apply, Submodule.mkQ_apply] at this
  rw [genQ, this, genQ, ← Submodule.Quotient.mk_smul]
  congr 1
  simp [gen, Finsupp.smul_single, sgnRep, s]

/-! ### The oriented chains are the permutation module -/

/-- `ℤ[Simp_q] → C_q`, `e_s ↦ orient s`. -/
def toChains : (Simp D q →₀ ℤ) →ₗ[ℤ] chains D q := Finsupp.linearCombination ℤ (orient D q)

lemma toChains_single (s : Simp D q) (c : ℤ) : toChains D q (Finsupp.single s c) = c • orient D q s := by
  simp [toChains, Finsupp.linearCombination_single]

/-- `C_q → ℤ[Simp_q]` on the free module: `e_(γ, σ) ↦ sgn(σ, σ̂, t_σ) e_{γ ι σ}`. -/
def ofFree : (SL(2, ℤ) × D.V q →₀ ℤ) →ₗ[ℤ] (Simp D q →₀ ℤ) :=
  Finsupp.linearCombination ℤ (fun x => sgnRep D q x.2 • Finsupp.single (SimpM.mk (D.M q) x) 1)

lemma ofFree_single (x : SL(2, ℤ) × D.V q) (c : ℤ) :
    ofFree D q (Finsupp.single x c) = (c * sgnRep D q x.2) • Finsupp.single (SimpM.mk (D.M q) x) 1 := by
  simp [ofFree, Finsupp.linearCombination_single, mul_smul]

/-- The sign identity behind `ofFree`: `sgn(σ, σ̂, t_σ) = sgn(σ, σ', u) · sgn(σ', σ̂, t_σ')` for `u ∈ S(σ, σ')`. -/
lemma sgnRep_transport {σ σ' : D.V q} {u : SL(2, ℤ)} (hu : u ∈ (D.M q).S σ σ') :
    (D.M q).sgn σ ((D.M q).rep σ) (tRep D q σ) =
      (D.M q).sgn σ σ' u * (D.M q).sgn σ' ((D.M q).rep σ') (tRep D q σ') := by
  have hr : (D.M q).rep σ = (D.M q).rep σ' := (D.M q).rep_eq ⟨_, hu⟩
  have h1 := (D.M q).sgn_mul hu (tRep_mem D q σ')
  rw [← h1]
  have ht := tRep_mem D q σ
  have hv : (tRep D q σ)⁻¹ * (u * tRep D q σ') ∈ (D.M q).S ((D.M q).rep σ) ((D.M q).rep σ') :=
    (D.M q).mul_mem ((D.M q).inv_mem ht) ((D.M q).mul_mem hu (tRep_mem D q σ'))
  have h2 := (D.M q).sgn_mul ht hv
  rw [mul_inv_cancel_left] at h2
  rw [h2]
  revert hv
  rw [← hr]
  intro hv
  rw [(D.M q).noInv hv, mul_one]

lemma relSub_le_ofFree : relSub (D.M q) ≤ LinearMap.ker (ofFree D q) := by
  rw [relSub, Submodule.span_le]
  rintro _ ⟨x, y, h, rfl⟩
  simp only [SetLike.mem_coe, LinearMap.mem_ker, map_sub, sub_eq_zero, gen, ofFree_single]
  have hm : SimpM.mk (D.M q) (x.1, x.2.1) = SimpM.mk (D.M q) (y.1, y.2.1) := Quot.sound h.1
  rw [hm]
  congr 1
  unfold sgnRep
  rw [h.2, sgnRep_transport D q h.1]
  simp only [Units.val_mul]
  have := sgn_sq ((D.M q).sgn x.2.1 y.2.1 (x.1⁻¹ * y.1))
  have h' : (((D.M q).sgn x.2.1 y.2.1 (x.1⁻¹ * y.1) : ℤˣ) : ℤ) * ((D.M q).sgn x.2.1 y.2.1 (x.1⁻¹ * y.1) : ℤˣ) = 1 := by
    rw [← Units.val_mul, this, Units.val_one]
  linear_combination (((y.2.2 : ℤˣ) : ℤ) * (((D.M q).sgn y.2.1 ((D.M q).rep y.2.1) (tRep D q y.2.1) : ℤˣ) : ℤ)) * h'

/-- `C_q → ℤ[Simp_q]`. -/
def ofChains : chains D q →ₗ[ℤ] (Simp D q →₀ ℤ) := (relSub (D.M q)).liftQ (ofFree D q) (relSub_le_ofFree D q)

lemma ofChains_genQ (x : SL(2, ℤ) × D.V q) :
    ofChains D q (genQ D q x) = sgnRep D q x.2 • Finsupp.single (SimpM.mk (D.M q) x) 1 := by
  show ofFree D q (gen _) = _
  simp only [gen, Units.val_one, ofFree_single, one_mul]

lemma ofChains_orient (s : Simp D q) : ofChains D q (orient D q s) = Finsupp.single s 1 := by
  induction s using Quot.ind with
  | _ x =>
    show ofChains D q (orientF D q x) = _
    rw [orientF, ofChains_genQ, sgnRep_rep, one_smul]
    congr 1
    show SimpM.mk _ _ = SimpM.mk _ _
    exact Quot.sound (by unfold Ku.Rel; simpa using (D.M q).inv_mem (tRep_mem D q x.2))

lemma toChains_ofChains (y : chains D q) : toChains D q (ofChains D q y) = y := by
  have := chains_hom_ext D (u := ((toChains D q).comp (ofChains D q)).toAddMonoidHom)
    (v := AddMonoidHom.id _) (fun a => by
      simp only [LinearMap.toAddMonoidHom_coe, LinearMap.coe_comp, Function.comp_apply, AddMonoidHom.id_apply,
        ofChains_genQ, map_zsmul, toChains_single, one_smul]
      exact (genQ_eq_orient D q a).symm)
  exact DFunLike.congr_fun this y

lemma ofChains_toChains (f : Simp D q →₀ ℤ) : ofChains D q (toChains D q f) = f := by
  induction f using Finsupp.induction_linear with
  | zero => simp
  | add f g hf hg => rw [map_add, map_add, hf, hg]
  | single s c => rw [toChains_single, map_zsmul, ofChains_orient, Finsupp.smul_single, smul_eq_mul, mul_one]

/-- **`chainsEquivPerm`: the oriented chains are the permutation module** `C_q(X_bulk) ≃ ℤ[Simp_q]`. -/
def chainsEquivPerm : chains D q ≃ₗ[ℤ] (Simp D q →₀ ℤ) :=
  LinearEquiv.ofLinearMap (ofChains D q) (toChains D q)
    (LinearMap.ext fun f => ofChains_toChains D q f) (LinearMap.ext fun y => toChains_ofChains D q y)

lemma chainsEquivPerm_symm_single (s : Simp D q) :
    (chainsEquivPerm D q).symm (Finsupp.single s 1) = orient D q s := by
  show toChains D q _ = _; rw [toChains_single, one_smul]

/-- `toChains` is equivariant. -/
lemma toChains_smul (γ : SL(2, ℤ)) (f : Simp D q →₀ ℤ) :
    toChains D q (Finsupp.mapDomain (γ • ·) f) = γ • toChains D q f := by
  induction f using Finsupp.induction_linear with
  | zero => simp
  | add f g hf hg => rw [Finsupp.mapDomain_add, map_add, map_add, hf, hg, smul_add]
  | single s c =>
    rw [Finsupp.mapDomain_single, toChains_single, toChains_single, orient_smul, smul_comm]

/-- **`chainsEquivPerm` is `SL(2, ℤ)`-equivariant.** -/
lemma ofChains_smul (γ : SL(2, ℤ)) (y : chains D q) :
    ofChains D q (γ • y) = Finsupp.mapDomain (γ • ·) (ofChains D q y) := by
  conv_lhs => rw [← toChains_ofChains D q y, ← toChains_smul]
  rw [ofChains_toChains]

end Orient


/-! ### The boundary and the augmentation on the permutation modules; exactness over `ℤ` -/

section Perm

/-- The boundary on `ℤ[Simp_•]`: `∂` transported through `chainsEquivPerm`. -/
def dZ (q : ℤ) : (Simp D (q + 1) →₀ ℤ) →ₗ[ℤ] (Simp D q →₀ ℤ) :=
  ofChains D q ∘ₗ bd D q ∘ₗ toChains D (q + 1)

/-- The augmentation on `ℤ[Simp_0]`. -/
def augZ : (Simp D 0 →₀ ℤ) →ₗ[ℤ] ℤ := aug D ∘ₗ toChains D 0

lemma dZ_dZ (q : ℤ) (f : Simp D (q + 1 + 1) →₀ ℤ) : dZ D q (dZ D (q + 1) f) = 0 := by
  simp only [dZ, LinearMap.coe_comp, Function.comp_apply, toChains_ofChains, bd_bd, map_zero]

lemma augZ_dZ (f : Simp D (0 + 1) →₀ ℤ) : augZ D (dZ D 0 f) = 0 := by
  simp only [augZ, dZ, LinearMap.coe_comp, Function.comp_apply, toChains_ofChains, aug_bd]

lemma dZ_smul (q : ℤ) (γ : SL(2, ℤ)) (f : Simp D (q + 1) →₀ ℤ) :
    dZ D q (Finsupp.mapDomain (γ • ·) f) = Finsupp.mapDomain (γ • ·) (dZ D q f) := by
  simp only [dZ, LinearMap.coe_comp, Function.comp_apply, toChains_smul, bd_equiv, ofChains_smul]

lemma augZ_smul (γ : SL(2, ℤ)) (f : Simp D 0 →₀ ℤ) : augZ D (Finsupp.mapDomain (γ • ·) f) = augZ D f := by
  simp only [augZ, LinearMap.coe_comp, Function.comp_apply, toChains_smul, aug_equiv]

lemma aug_genQ (x : SL(2, ℤ) × D.V 0) : aug D (genQ D 0 x) = 1 := by
  show augF D (gen _) = 1
  rw [augF_gen]; rfl

lemma augZ_single (s : Simp D 0) : augZ D (Finsupp.single s 1) = 1 := by
  rw [augZ, LinearMap.comp_apply, toChains_single, one_smul]
  induction s using Quot.ind with
  | _ x => exact aug_genQ D _

variable {D}

lemma exactZ0 (hA : Acyclic D) (f : Simp D 0 →₀ ℤ) (hf : augZ D f = 0) : ∃ g, dZ D 0 g = f := by
  obtain ⟨y, hy⟩ := hA.contract0 (toChains D 0 f) hf
  exact ⟨ofChains D (0 + 1) y, by simp only [dZ, LinearMap.coe_comp, Function.comp_apply, toChains_ofChains, hy,
    ofChains_toChains]⟩

lemma exactZ1 (hA : Acyclic D) (f : Simp D (0 + 1) →₀ ℤ) (hf : dZ D 0 f = 0) : ∃ g, dZ D (0 + 1) g = f := by
  have h0 : bd D 0 (toChains D (0 + 1) f) = 0 := by
    have := congrArg (toChains D 0) hf
    simpa only [dZ, LinearMap.coe_comp, Function.comp_apply, toChains_ofChains, map_zero] using this
  obtain ⟨y, hy⟩ := hA.contract1 _ h0
  exact ⟨ofChains D (0 + 1 + 1) y, by simp only [dZ, LinearMap.coe_comp, Function.comp_apply, toChains_ofChains, hy,
    ofChains_toChains]⟩

lemma exactZ2 (hA : Acyclic D) (f : Simp D (0 + 1 + 1) →₀ ℤ) (hf : dZ D (0 + 1) f = 0) : f = 0 := by
  have h0 : bd D (0 + 1) (toChains D (0 + 1 + 1) f) = 0 := by
    have := congrArg (toChains D (0 + 1)) hf
    simpa only [dZ, LinearMap.coe_comp, Function.comp_apply, toChains_ofChains, map_zero] using this
  have := hA.contract2 _ h0
  rw [← ofChains_toChains D _ f, this, map_zero]

/-- **The contracting homotopy over `ℤ`** of the augmented complex `ℤ[Simp_2] → ℤ[Simp_1] → ℤ[Simp_0] → ℤ` of an
acyclic `X_bulk` with a vertex `v`: `z = e_v`, `s₀`, `s₁` with `ε z = 1`, `∂ s₀ = 1 - z ε`, `∂ s₁ = 1 - s₀ ∂`,
`s₁ ∂ = 1`; built degree-wise from the projectivity of the free modules. -/
structure ContractZ (D : XBulkDatum) where
  z : Simp D 0 →₀ ℤ
  s0 : (Simp D 0 →₀ ℤ) →ₗ[ℤ] (Simp D (0 + 1) →₀ ℤ)
  s1 : (Simp D (0 + 1) →₀ ℤ) →ₗ[ℤ] (Simp D (0 + 1 + 1) →₀ ℤ)
  aug_z : augZ D z = 1
  d_s0 : dZ D 0 ∘ₗ s0 = LinearMap.id - (augZ D).smulRight z
  d_s1 : dZ D (0 + 1) ∘ₗ s1 = LinearMap.id - s0 ∘ₗ dZ D 0
  s1_d : s1 ∘ₗ dZ D (0 + 1) = LinearMap.id

/-- `contractZ`: an acyclic `X_bulk` with a vertex has a contracting homotopy over `ℤ`. -/
theorem contractZ (hA : Acyclic D) (v : D.V 0) : Nonempty (ContractZ D) := by
  set z : Simp D 0 →₀ ℤ := Finsupp.single (SimpM.mk (D.M 0) (1, v)) 1
  have hz : augZ D z = 1 := augZ_single D _
  -- s₀
  have hr0 : ∀ x, x - augZ D x • z ∈ LinearMap.range (dZ D 0) := by
    intro x
    obtain ⟨g, hg⟩ := exactZ0 hA (x - augZ D x • z) (by simp [hz])
    exact ⟨g, hg⟩
  obtain ⟨s0', hs0'⟩ := Module.projective_lifting_property (dZ D 0).rangeRestrict
    (LinearMap.codRestrict _ (LinearMap.id - (augZ D).smulRight z) (fun x => by simpa using hr0 x))
    (LinearMap.surjective_rangeRestrict _)
  have hs0 : dZ D 0 ∘ₗ s0' = LinearMap.id - (augZ D).smulRight z := by
    refine LinearMap.ext fun x => ?_
    have := congrArg Subtype.val (LinearMap.congr_fun hs0' x)
    simpa using this
  -- s₁
  have hr1 : ∀ x, x - s0' (dZ D 0 x) ∈ LinearMap.range (dZ D (0 + 1)) := by
    intro x
    have hk : dZ D 0 (x - s0' (dZ D 0 x)) = 0 := by
      have := LinearMap.congr_fun hs0 (dZ D 0 x)
      simp only [LinearMap.coe_comp, Function.comp_apply, LinearMap.sub_apply, LinearMap.id_apply,
        LinearMap.smulRight_apply, augZ_dZ, zero_smul, sub_zero] at this
      rw [map_sub, this, sub_self]
    obtain ⟨g, hg⟩ := exactZ1 hA _ hk
    exact ⟨g, hg⟩
  obtain ⟨s1', hs1'⟩ := Module.projective_lifting_property (dZ D (0 + 1)).rangeRestrict
    (LinearMap.codRestrict _ (LinearMap.id - s0' ∘ₗ dZ D 0) (fun x => by simpa using hr1 x))
    (LinearMap.surjective_rangeRestrict _)
  have hs1 : dZ D (0 + 1) ∘ₗ s1' = LinearMap.id - s0' ∘ₗ dZ D 0 := by
    refine LinearMap.ext fun x => ?_
    have := congrArg Subtype.val (LinearMap.congr_fun hs1' x)
    simpa using this
  refine ⟨⟨z, s0', s1', hz, hs0, hs1, ?_⟩⟩
  refine LinearMap.ext fun x => ?_
  apply sub_eq_zero.1
  apply exactZ2 hA
  have := LinearMap.congr_fun hs1 (dZ D (0 + 1) x)
  simp only [LinearMap.coe_comp, Function.comp_apply, LinearMap.sub_apply, LinearMap.id_apply, dZ_dZ, map_zero,
    sub_zero] at this
  simp only [LinearMap.coe_comp, Function.comp_apply, LinearMap.id_apply, map_sub, this, sub_self]

end Perm

/-! ### Base change `ℤ[S] → R[S]` of `ℤ`-linear maps between permutation modules -/

section BaseChange

variable {R : Type} [CommRing R] {S T U : Type}

/-- `ℤ[S] → R[S]`, coefficientwise. -/
def castF : (S →₀ ℤ) →+ (S →₀ R) := Finsupp.mapRange.addMonoidHom (Int.castAddHom R)

lemma castF_single (s : S) (n : ℤ) : castF (R := R) (Finsupp.single s n) = Finsupp.single s (n : R) := by
  simp [castF]

lemma castF_mapDomain (h : S → T) (x : S →₀ ℤ) :
    castF (R := R) (Finsupp.mapDomain h x) = Finsupp.mapDomain h (castF x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add, Finsupp.mapDomain_add]
  | single s n => rw [Finsupp.mapDomain_single, castF_single, castF_single, Finsupp.mapDomain_single]

variable (R) in
/-- **The base change** of a `ℤ`-linear map `ℤ[S] → ℤ[T]` to `R[S] → R[T]`. -/
def bcF (f : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)) : (S →₀ R) →ₗ[R] (T →₀ R) :=
  Finsupp.linearCombination R (fun s => castF (f (Finsupp.single s 1)))

lemma bcF_single (f : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)) (s : S) (r : R) :
    bcF R f (Finsupp.single s r) = r • castF (f (Finsupp.single s 1)) := by
  simp [bcF, Finsupp.linearCombination_single]

lemma castF_zsmul (n : ℤ) (x : S →₀ ℤ) : castF (R := R) (n • x) = (n : R) • castF x := by
  rw [map_zsmul, Int.cast_smul_eq_zsmul]

lemma bcF_castF (f : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)) (x : S →₀ ℤ) : bcF R f (castF x) = castF (f x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, map_add, map_add]
  | single s n =>
    rw [castF_single, bcF_single]
    have : Finsupp.single s n = n • Finsupp.single s (1 : ℤ) := by simp [Finsupp.smul_single]
    rw [this, map_zsmul, castF_zsmul]

lemma bcF_comp (f : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)) (g : (T →₀ ℤ) →ₗ[ℤ] (U →₀ ℤ)) :
    bcF R (g ∘ₗ f) = bcF R g ∘ₗ bcF R f := by
  apply Finsupp.lhom_ext; intro s r
  simp only [LinearMap.coe_comp, Function.comp_apply, bcF_single, map_smul, bcF_castF]

lemma bcF_id : bcF R (LinearMap.id : (S →₀ ℤ) →ₗ[ℤ] (S →₀ ℤ)) = LinearMap.id := by
  apply Finsupp.lhom_ext; intro s r
  simp [bcF_single, castF_single, Finsupp.smul_single]

lemma bcF_sub (f g : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)) : bcF R (f - g) = bcF R f - bcF R g := by
  apply Finsupp.lhom_ext; intro s r
  simp [bcF_single, smul_sub]

lemma bcF_zero : bcF R (0 : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)) = 0 := by
  apply Finsupp.lhom_ext; intro s r
  simp [bcF_single]

variable (R) in
/-- The base change of a `ℤ`-linear form `ℤ[S] → ℤ`. -/
def bcA (a : (S →₀ ℤ) →ₗ[ℤ] ℤ) : (S →₀ R) →ₗ[R] R :=
  Finsupp.linearCombination R (fun s => ((a (Finsupp.single s 1) : ℤ) : R))

lemma bcA_single (a : (S →₀ ℤ) →ₗ[ℤ] ℤ) (s : S) (r : R) :
    bcA R a (Finsupp.single s r) = r * (a (Finsupp.single s 1) : R) := by
  simp [bcA, Finsupp.linearCombination_single]

lemma bcA_castF (a : (S →₀ ℤ) →ₗ[ℤ] ℤ) (x : S →₀ ℤ) : bcA R a (castF x) = (a x : R) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, map_add, Int.cast_add]
  | single s n =>
    rw [castF_single, bcA_single]
    have : Finsupp.single s n = n • Finsupp.single s (1 : ℤ) := by simp [Finsupp.smul_single]
    rw [this, map_zsmul, smul_eq_mul, Int.cast_mul]

lemma bcA_zero : bcA R (0 : (S →₀ ℤ) →ₗ[ℤ] ℤ) = 0 := by
  apply Finsupp.lhom_ext; intro s r
  simp [bcA_single]

lemma bcA_comp (f : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)) (a : (T →₀ ℤ) →ₗ[ℤ] ℤ) :
    bcA R (a ∘ₗ f) = bcA R a ∘ₗ bcF R f := by
  apply Finsupp.lhom_ext; intro s r
  simp only [LinearMap.coe_comp, Function.comp_apply, bcA_single, bcF_single, map_smul, bcA_castF, smul_eq_mul]

lemma bcF_smulRight (a : (S →₀ ℤ) →ₗ[ℤ] ℤ) (z : T →₀ ℤ) (x : S →₀ R) :
    bcF R (a.smulRight z) x = bcA R a x • castF z := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [map_add, map_add, hx, hy, add_smul]
  | single s r =>
    rw [bcF_single, bcA_single, LinearMap.smulRight_apply, castF_zsmul, smul_smul]

variable {G : Type} [Group G] [MulAction G S] [MulAction G T]

lemma bcF_mapDomain {f : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)}
    (hf : ∀ (g : G) x, f (Finsupp.mapDomain (g • ·) x) = Finsupp.mapDomain (g • ·) (f x)) (g : G) (x : S →₀ R) :
    bcF R f (Finsupp.mapDomain (g • ·) x) = Finsupp.mapDomain (g • ·) (bcF R f x) := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add, Finsupp.mapDomain_add]
  | single s r =>
    rw [Finsupp.mapDomain_single, bcF_single, bcF_single, Finsupp.mapDomain_smul]
    congr 1
    rw [← castF_mapDomain, ← hf, Finsupp.mapDomain_single]

lemma bcA_mapDomain {a : (S →₀ ℤ) →ₗ[ℤ] ℤ} (ha : ∀ (g : G) x, a (Finsupp.mapDomain (g • ·) x) = a x) (g : G)
    (x : S →₀ R) : bcA R a (Finsupp.mapDomain (g • ·) x) = bcA R a x := by
  induction x using Finsupp.induction_linear with
  | zero => simp
  | add x y hx hy => rw [Finsupp.mapDomain_add, map_add, hx, hy, map_add]
  | single s r =>
    rw [Finsupp.mapDomain_single, bcA_single, bcA_single]
    have := ha g (Finsupp.single s 1)
    rw [Finsupp.mapDomain_single] at this
    rw [this]

open MonoidAlgebra

variable (R) in
/-- The base-changed map as a morphism of permutation representations `R[S] → R[T]`. -/
def repMap (f : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ))
    (hf : ∀ (g : G) x, f (Finsupp.mapDomain (g • ·) x) = Finsupp.mapDomain (g • ·) (f x)) :
    Rep.ofMulAction R G S ⟶ Rep.ofMulAction R G T :=
  Rep.ofHom ⟨(coeffLinearEquiv R).symm.toLinearMap ∘ₗ bcF R f ∘ₗ (coeffLinearEquiv R).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply]
    rw [bcF_mapDomain hf]⟩

lemma repMap_hom_apply (f : (S →₀ ℤ) →ₗ[ℤ] (T →₀ ℤ)) (hf) (x : R[S]) :
    (repMap R (G := G) f hf).hom x = (coeffLinearEquiv R).symm (bcF R f (coeffLinearEquiv R x)) := rfl

variable (R) in
/-- The base-changed form as a morphism `R[S] → R` (trivial action on `R`). -/
def repAugMap (a : (S →₀ ℤ) →ₗ[ℤ] ℤ) (ha : ∀ (g : G) x, a (Finsupp.mapDomain (g • ·) x) = a x) :
    Rep.ofMulAction R G S ⟶ Rep.trivial R G R :=
  Rep.ofHom ⟨bcA R a ∘ₗ (coeffLinearEquiv R).toLinearMap, fun g => by
    refine LinearMap.ext fun x => ?_
    simp only [LinearMap.coe_comp, Function.comp_apply, LinearEquiv.coe_coe, Representation.ofMulAction_def,
      LinearEquiv.apply_symm_apply, Finsupp.lmapDomain_apply, Representation.trivial_apply]
    rw [bcA_mapDomain ha]⟩

lemma repAugMap_hom_apply (a : (S →₀ ℤ) →ₗ[ℤ] ℤ) (ha) (x : R[S]) :
    (repAugMap R (G := G) a ha).hom x = bcA R a (coeffLinearEquiv R x) := rfl

end BaseChange

/-! ### The complex of permutation representations and its augmentation -/

section RepCx

open CategoryTheory MonoidAlgebra

variable (R : Type) [CommRing R]

/-- `R[Simp_n]`, the permutation representation on the `n`-simplices of `X_bulk`. -/
abbrev repX (n : ℕ) : Rep R SL(2, ℤ) := Rep.ofMulAction R SL(2, ℤ) (Simp D n)

/-- The boundary `R[Simp_{n+1}] → R[Simp_n]`. -/
def repD (n : ℕ) : repX D R (n + 1) ⟶ repX D R n := repMap R (dZ D n) (dZ_smul D n)

lemma repD_repD (n : ℕ) : repD D R (n + 1) ≫ repD D R n = 0 := by
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show (coeffLinearEquiv R).symm (bcF R (dZ D n) (coeffLinearEquiv R ((coeffLinearEquiv R).symm
    (bcF R (dZ D (n + 1 : ℕ)) (coeffLinearEquiv R x))))) = 0
  rw [LinearEquiv.apply_symm_apply, ← LinearMap.comp_apply, ← bcF_comp]
  have : dZ D n ∘ₗ dZ D (n + 1 : ℕ) = 0 := LinearMap.ext fun f => dZ_dZ D n f
  rw [this, bcF_zero, LinearMap.zero_apply, map_zero]

/-- **`XBulk.repCx`**: `C_•(X_bulk) ⊗ R` as a chain complex of representations `R[Simp_2] → R[Simp_1] → R[Simp_0]`
(zero above degree `2`). -/
def repCx : ChainComplex (Rep R SL(2, ℤ)) ℕ := ChainComplex.of (repX D R) (repD D R) (repD_repD D R)

/-- **`XBulk.repCx_iso_perm`**: the terms of `repCx` are the permutation representations `R[Simp_n]`. -/
def repCx_iso_perm (n : ℕ) : (repCx D R).X n ≅ Rep.ofMulAction R SL(2, ℤ) (Simp D n) := Iso.refl _

lemma repCx_d (n : ℕ) : (repCx D R).d (n + 1) n = repD D R n := ChainComplex.of_d _ _ n

/-- The augmentation `R[Simp_0] → R`. -/
def repEps : repX D R 0 ⟶ Rep.trivial R SL(2, ℤ) R := repAugMap R (augZ D) (augZ_smul D)

lemma repD_repEps : (repCx D R).d 1 0 ≫ repEps D R = 0 := by
  rw [repCx_d D R 0]
  apply Rep.hom_ext; apply Representation.IntertwiningMap.ext; refine LinearMap.ext fun x => ?_
  show bcA R (augZ D) (coeffLinearEquiv R ((coeffLinearEquiv R).symm
    (bcF R (dZ D 0) (coeffLinearEquiv R x)))) = 0
  rw [LinearEquiv.apply_symm_apply, ← LinearMap.comp_apply, ← bcA_comp]
  have : augZ D ∘ₗ dZ D 0 = 0 := LinearMap.ext fun f => augZ_dZ D f
  rw [this, bcA_zero, LinearMap.zero_apply]

/-- **`XBulk.repAug`**: the augmentation `repCx → R` (the trivial representation in degree `0`). -/
def repAug : repCx D R ⟶ (ChainComplex.single₀ (Rep R SL(2, ℤ))).obj (Rep.trivial R SL(2, ℤ) R) :=
  ((repCx D R).toSingle₀Equiv _).symm ⟨repEps D R, repD_repEps D R⟩

lemma repAug_f_zero : (repAug D R).f 0 = repEps D R :=
  ChainComplex.toSingle₀Equiv_symm_apply_f_zero _ _

end RepCx

/-! ### The quasi-isomorphism -/

section QuasiIso

open CategoryTheory MonoidAlgebra

variable {D} (R : Type) [CommRing R]

/-- The contracting homotopy over `R` (base change of `ContractZ`). -/
lemma contractR (c : ContractZ D) :
    bcA R (augZ D) (castF c.z) = 1 ∧
    (∀ x, bcF R (dZ D 0) (bcF R c.s0 x) = x - bcA R (augZ D) x • castF c.z) ∧
    (∀ x, bcF R (dZ D (0 + 1)) (bcF R c.s1 x) = x - bcF R c.s0 (bcF R (dZ D 0) x)) ∧
    (∀ x, bcF R c.s1 (bcF R (dZ D (0 + 1)) x) = x) := by
  refine ⟨by rw [bcA_castF, c.aug_z, Int.cast_one], fun x => ?_, fun x => ?_, fun x => ?_⟩
  · have := congrArg (bcF R) c.d_s0
    rw [bcF_comp, bcF_sub, bcF_id] at this
    have := LinearMap.congr_fun this x
    simpa [bcF_smulRight] using this
  · have := congrArg (bcF R) c.d_s1
    rw [bcF_comp, bcF_sub, bcF_id, bcF_comp] at this
    simpa using LinearMap.congr_fun this x
  · have := congrArg (bcF R) c.s1_d
    rw [bcF_comp, bcF_id] at this
    simpa using LinearMap.congr_fun this x

/-- Exactness at `R[Simp_0]` of the augmented complex, from the base-changed contracting homotopy. -/
lemma exact_zero (c : ContractZ D)
    (h0 : ∀ x, bcF R (dZ D 0) (bcF R c.s0 x) = x - bcA R (augZ D) x • castF c.z)
    (w : (repCx D R).d 1 0 ≫ (repAug D R).f 0 = 0) :
    (ShortComplex.mk ((repCx D R).d 1 0) ((repAug D R).f 0) w).Exact := by
  apply (forget₂ _ (ModuleCat R)).reflects_exact_of_faithful
  rw [ShortComplex.moduleCat_exact_iff]
  intro x hx
  change ((repAug D R).f 0).hom x = 0 at hx
  rw [repAug_f_zero] at hx
  refine ⟨(coeffLinearEquiv R).symm (bcF R c.s0 (coeffLinearEquiv R x)), ?_⟩
  change ((repCx D R).d 1 0).hom _ = x
  rw [repCx_d D R 0]
  show (coeffLinearEquiv R).symm (bcF R (dZ D 0) (coeffLinearEquiv R ((coeffLinearEquiv R).symm
    (bcF R c.s0 (coeffLinearEquiv R x))))) = x
  rw [LinearEquiv.apply_symm_apply, h0]
  have hx' : bcA R (augZ D) (coeffLinearEquiv R x) = 0 := hx
  rw [hx', zero_smul, sub_zero]
  exact (coeffLinearEquiv R).symm_apply_apply (x : R[Simp D 0])

/-- The core of `exact_succ`, on the underlying modules. -/
lemma exact_succ_core (c : ContractZ D)
    (h1 : ∀ x, bcF R (dZ D (0 + 1)) (bcF R c.s1 x) = x - bcF R c.s0 (bcF R (dZ D 0) x))
    (h2 : ∀ x, bcF R c.s1 (bcF R (dZ D (0 + 1)) x) = x) (m : ℕ) (x : R[Simp D ((m + 1 : ℕ) : ℤ)])
    (hx : bcF R (dZ D m) (coeffLinearEquiv R x) = 0) :
    ∃ y : R[Simp D ((m + 1 + 1 : ℕ) : ℤ)],
      (coeffLinearEquiv R).symm (bcF R (dZ D (m + 1 : ℕ)) (coeffLinearEquiv R y)) = x := by
  cases m with
  | zero =>
    refine ⟨(coeffLinearEquiv R).symm (bcF R c.s1 (coeffLinearEquiv R x)), ?_⟩
    rw [LinearEquiv.apply_symm_apply]
    have := h1 (coeffLinearEquiv R x)
    rw [show bcF R (dZ D 0) (coeffLinearEquiv R x) = 0 from hx, map_zero, sub_zero] at this
    exact (congrArg (coeffLinearEquiv R).symm this).trans ((coeffLinearEquiv R).symm_apply_apply x)
  | succ m =>
    have hxz : x = 0 := by
      rw [← (coeffLinearEquiv R).map_eq_zero_iff]
      cases m with
      | zero =>
        have := h2 (coeffLinearEquiv R x)
        rw [show bcF R (dZ D (0 + 1)) (coeffLinearEquiv R x) = 0 from hx, map_zero] at this
        exact this.symm
      | succ m =>
        have : IsEmpty (Simp D ((m + 1 + 1 + 1 : ℕ) : ℤ)) := by
          have := D.dim2 ((m + 1 + 1 + 1 : ℕ) : ℤ) (Or.inr (by omega))
          exact ⟨fun s => by induction s using Quot.ind with | _ y => exact this.false y.2⟩
        exact Subsingleton.elim _ _
    exact ⟨0, by rw [hxz, map_zero, map_zero, map_zero]⟩

/-- `ε_R` is onto. -/
lemma repEps_surjective (c : ContractZ D) (hz : bcA R (augZ D) (castF c.z) = 1) :
    Function.Surjective (repEps D R).hom := by
  intro r
  refine ⟨(coeffLinearEquiv R).symm (r • castF (R := R) c.z), ?_⟩
  show bcA R (augZ D) (coeffLinearEquiv R ((coeffLinearEquiv R).symm (r • castF c.z))) = r
  rw [LinearEquiv.apply_symm_apply, map_smul, hz, smul_eq_mul, mul_one]

/-- Exactness at `R[Simp_{m+1}]`, from the base-changed contracting homotopy (and `R[Simp_q] = 0` for `q ≥ 3`). -/
lemma exact_succ (c : ContractZ D)
    (h1 : ∀ x, bcF R (dZ D (0 + 1)) (bcF R c.s1 x) = x - bcF R c.s0 (bcF R (dZ D 0) x))
    (h2 : ∀ x, bcF R c.s1 (bcF R (dZ D (0 + 1)) x) = x) (m : ℕ) :
    ((repCx D R).sc' (m + 2) (m + 1) m).Exact := by
  apply (forget₂ _ (ModuleCat R)).reflects_exact_of_faithful
  rw [ShortComplex.moduleCat_exact_iff]
  intro x hx
  change ((repCx D R).d (m + 1) m).hom x = 0 at hx
  rw [repCx_d D R m] at hx
  change ∃ y, ((repCx D R).d (m + 1 + 1) (m + 1)).hom y = x
  rw [repCx_d D R (m + 1)]
  have hx' : (coeffLinearEquiv R).symm (bcF R (dZ D m) (coeffLinearEquiv R x)) = 0 := hx
  rw [LinearEquiv.map_eq_zero_iff] at hx'
  exact exact_succ_core R c h1 h2 m x hx'

/-- **`XBulk.repAug_quasiIso`**: for an acyclic `X_bulk` (H4(ii)) with a vertex, the augmentation `repCx → R` is a
quasi-isomorphism in `Rep R SL(2, ℤ)`, over any commutative ring `R`. -/
theorem repAug_quasiIso (hA : Acyclic D) (v : D.V 0) : QuasiIso (repAug D R) where
  quasiIsoAt m := by
    obtain ⟨c⟩ := contractZ hA v
    obtain ⟨hz, h0, h1, h2⟩ := contractR R c
    cases m with
    | zero =>
      rw [ChainComplex.quasiIsoAt₀_iff, ShortComplex.quasiIso_iff_of_zeros' _ rfl rfl rfl]
      constructor
      · refine exact_zero R c h0 ?_
      · rw [Rep.epi_iff_surjective]
        change Function.Surjective ((repAug D R).f 0).hom
        rw [repAug_f_zero]
        exact repEps_surjective R c hz
    | succ m =>
      rw [quasiIsoAt_iff_exactAt' (hL := ChainComplex.exactAt_succ_single_obj ..),
          HomologicalComplex.exactAt_iff' _ (m + 2) (m + 1) m (by simp) (by simp)]
      exact exact_succ R c h1 h2 m

end QuasiIso

end XBulk

end GL2.Model

end
