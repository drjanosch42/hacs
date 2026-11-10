import GL2.Model.Oriented
import GL2.Cone.Pair

/-!
# M2-c (L18-1): the combinatorial `X_bulk` as a descriptor, and its chain complex as a `GChainCx`

[GT] `def:xbulk`, `thm:xbulk-cx`, mirroring [C] v1.17.0 `H4`, `def:model7`, `rmk:twosettings` (`Kᵘ ≅ X_bulk`).

**The MP rule.** `XBulkDatum` is a *descriptor*, never the 7-vertex model: per degree `q : ℤ` a finite type `V q` of
model simplices with an oriented key model `M q` (transporter groupoid + orientation signs), a finite type `F q` of face
slots, and **signed faces** `face q τ j = (h, σ, ε)`: the `j`-th face of `ι(τ)` is the oriented simplex `ε · h ι(σ)`
(the alternating sign `(-1)^j` is part of `ε`).  The axioms are exactly what the oriented simplicial chains need:

* `face_transport` -- a transporter `t ∈ S(τ, τ')` permutes the faces (by some `π`) compatibly with the signs;
* `face_simplicial` -- the double faces of every `τ` cancel in pairs under some involution (the simplicial identity
  `∂_i ∂_j = ∂_{j-1} ∂_i`, `i < j`, in signed form);
* `sgn0`, `aug_face` -- vertices carry no orientation, and the signed faces of an edge sum to `0`;
* `dim2` -- no simplices outside degrees `0, 1, 2`.

**The chains.** `XBulk.chains D q` is the free `ℤ`-module on pairs `(γ, σ) ∈ SL(2, ℤ) × V q` modulo the oriented
relation `[γ, σ, ε] = [γ', σ', ε']` whenever `γ⁻¹γ' ∈ S(σ, σ')` and `ε = sgn · ε'` (`OrRel`): that is `C_q(Kᵘ)`, the
oriented simplicial chains of the `Γ`-complex.  `SL(2, ℤ)` acts by left translation, `∂` is the signed face sum,
`∂∂ = 0` (`bd_bd`) and the augmentation counts vertices.  `toGChainCx` packages it as the `GChainCx` the cone layer
(`GL2.Cone.Pair`) consumes, for any subgroup `H ≤ SL(2, ℤ)` (`toGChainCxOn`).
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open GL2.Ku

lemma quot_mk_sum {R M ι : Type*} [Ring R] [AddCommGroup M] [Module R M]
    (p : Submodule R M) (s : Finset ι) (f : ι → M) :
    (Submodule.Quotient.mk (∑ i ∈ s, f i) : M ⧸ p) = ∑ i ∈ s, Submodule.Quotient.mk (f i) :=
  map_sum (Submodule.mkQ p) f s

/-- An oriented model simplex `ε · γ ι(σ)`. -/
abbrev OrPt (V : Type) := SL(2, ℤ) × V × ℤˣ

namespace OrPt
variable {V : Type}

/-- Left translation `γ · (h, σ, ε) = (γ h, σ, ε)`. -/
def lt (γ : SL(2, ℤ)) (x : OrPt V) : OrPt V := (γ * x.1, x.2.1, x.2.2)

/-- Transport by `t` with an extra sign `s`: `(t h, σ, s ε)`. -/
def tr (t : SL(2, ℤ)) (x : OrPt V) (s : ℤˣ) : OrPt V := (t * x.1, x.2.1, s * x.2.2)

/-- Orientation reversal. -/
def neg (x : OrPt V) : OrPt V := (x.1, x.2.1, -x.2.2)

/-- The face `c` of the face `a`: `(h_a h_c, σ_c, ε_a ε_c)`. -/
def comp2 {W : Type} (a : OrPt W) (c : OrPt V) : OrPt V := (a.1 * c.1, c.2.1, a.2.2 * c.2.2)

lemma lt_tr (γ t : SL(2, ℤ)) (x : OrPt V) (s : ℤˣ) : lt γ (tr t x s) = tr (γ * t) x s := by
  simp [lt, tr, mul_assoc]

end OrPt

/-- The oriented relation: `(γ, σ, ε)` and `(γ', σ', ε')` are the same oriented simplex of `Kᵘ`. -/
def OrRel {V : Type} (M : OrKeyModel V) (x y : OrPt V) : Prop :=
  x.1⁻¹ * y.1 ∈ M.S x.2.1 y.2.1 ∧ x.2.2 = M.sgn x.2.1 y.2.1 (x.1⁻¹ * y.1) * y.2.2

/-- **`def:xbulk`**: the combinatorial `X_bulk` as a finite `Γ`-complex descriptor. -/
structure XBulkDatum where
  V : ℤ → Type
  [finV : ∀ q, Fintype (V q)]
  [deqV : ∀ q, DecidableEq (V q)]
  M : ∀ q, OrKeyModel (V q)
  /-- the face slots of a `(q+1)`-simplex -/
  F : ℤ → Type
  [finF : ∀ q, Fintype (F q)]
  [deqF : ∀ q, DecidableEq (F q)]
  /-- the signed faces: `∂_j ι(τ) = ε · h ι(σ)` -/
  face : ∀ q, V (q + 1) → F q → OrPt (V q)
  face_transport : ∀ q {τ τ' : V (q + 1)} {t : SL(2, ℤ)}, t ∈ (M (q + 1)).S τ τ' →
    ∃ π : F q ≃ F q, ∀ j, OrRel (M q) (face q τ j) (OrPt.tr t (face q τ' (π j)) ((M (q + 1)).sgn τ τ' t))
  face_simplicial : ∀ q (τ : V (q + 1 + 1)), ∃ ι : F (q + 1) × F q → F (q + 1) × F q, Function.Involutive ι ∧
    ∀ p, OrRel (M q) (OrPt.comp2 (face (q + 1) τ p.1) (face q (face (q + 1) τ p.1).2.1 p.2))
      (OrPt.neg (OrPt.comp2 (face (q + 1) τ (ι p).1) (face q (face (q + 1) τ (ι p).1).2.1 (ι p).2)))
  sgn0 : ∀ {σ σ' : V 0} {t : SL(2, ℤ)}, t ∈ (M 0).S σ σ' → (M 0).sgn σ σ' t = 1
  aug_face : ∀ τ : V (0 + 1), ∑ j, ((face 0 τ j).2.2 : ℤ) = 0
  dim2 : ∀ q, q < 0 ∨ 2 < q → IsEmpty (V q)

attribute [instance] XBulkDatum.finV XBulkDatum.deqV XBulkDatum.finF XBulkDatum.deqF

namespace XBulk

variable {V : Type} (M : OrKeyModel V)

/-! ### The oriented relation -/

lemma orRel_lt {x y : OrPt V} (γ : SL(2, ℤ)) (h : OrRel M x y) : OrRel M (OrPt.lt γ x) (OrPt.lt γ y) := by
  have e : (γ * x.1)⁻¹ * (γ * y.1) = x.1⁻¹ * y.1 := by group
  exact ⟨by simp only [OrPt.lt, e]; exact h.1, by simp only [OrPt.lt, e]; exact h.2⟩

/-- An oriented point is never related to its own reversal (the orientation is `ℤˣ`-valued, `sgn σ σ 1 = 1`). -/
lemma not_orRel_neg (x : OrPt V) : ¬ OrRel M x (OrPt.neg x) := by
  rintro ⟨-, h⟩
  simp only [OrPt.neg, inv_mul_cancel, M.sgn_one, one_mul] at h
  rcases Int.units_eq_one_or x.2.2 with e | e <;> rw [e] at h <;> exact absurd h (by decide)

/-! ### Chains -/

/-- The free generator of an oriented point: `ε · e_(γ, σ)`. -/
def gen (x : OrPt V) : SL(2, ℤ) × V →₀ ℤ := Finsupp.single (x.1, x.2.1) (x.2.2 : ℤ)

/-- The relations `gen x - gen y`, `x ~ y`. -/
def relSub : Submodule ℤ (SL(2, ℤ) × V →₀ ℤ) :=
  Submodule.span ℤ {f | ∃ x y, OrRel M x y ∧ f = gen x - gen y}

/-- **The chains** `C_q(Kᵘ)` of one degree. -/
abbrev Ch := (SL(2, ℤ) × V →₀ ℤ) ⧸ relSub M

lemma mk_gen_eq {x y : OrPt V} (h : OrRel M x y) :
    (Submodule.mkQ (relSub M)) (gen x) = (Submodule.mkQ (relSub M)) (gen y) := by
  rw [Submodule.mkQ_apply, Submodule.mkQ_apply, Submodule.Quotient.eq]
  exact Submodule.subset_span ⟨x, y, h, rfl⟩

lemma gen_tr (t : SL(2, ℤ)) (x : OrPt V) (s : ℤˣ) : gen (OrPt.tr t x s) = (s : ℤ) • gen (OrPt.lt t x) := by
  simp [gen, OrPt.tr, OrPt.lt, Finsupp.smul_single]

lemma gen_neg (x : OrPt V) : gen (OrPt.neg x) = - gen x := by
  simp [gen, OrPt.neg]

/-! ### The action -/

/-- Left translation on the free module. -/
def actF (γ : SL(2, ℤ)) : (SL(2, ℤ) × V →₀ ℤ) →ₗ[ℤ] (SL(2, ℤ) × V →₀ ℤ) :=
  Finsupp.lmapDomain ℤ ℤ (fun x => (γ * x.1, x.2))

lemma actF_gen (γ : SL(2, ℤ)) (x : OrPt V) : actF γ (gen x) = gen (OrPt.lt γ x) := by
  simp [actF, gen, OrPt.lt, Finsupp.mapDomain_single]

lemma actF_single (γ : SL(2, ℤ)) (a : SL(2, ℤ) × V) (c : ℤ) :
    actF γ (Finsupp.single a c) = Finsupp.single (γ * a.1, a.2) c := by
  simp [actF, Finsupp.mapDomain_single]

lemma relSub_le_actF (γ : SL(2, ℤ)) : relSub M ≤ (relSub M).comap (actF γ) := by
  rw [relSub, Submodule.span_le]
  rintro _ ⟨x, y, h, rfl⟩
  simp only [SetLike.mem_coe, Submodule.mem_comap, map_sub, actF_gen]
  exact Submodule.subset_span ⟨_, _, orRel_lt M γ h, rfl⟩

/-- Left translation on the chains. -/
def actQ (γ : SL(2, ℤ)) : Ch M →ₗ[ℤ] Ch M := (relSub M).mapQ (relSub M) (actF γ) (relSub_le_actF M γ)

lemma actF_one : actF (V := V) 1 = LinearMap.id := by
  apply Finsupp.lhom_ext; intro a b; simp [actF_single]

lemma actF_mul (γ δ : SL(2, ℤ)) : actF (V := V) (γ * δ) = (actF γ).comp (actF δ) := by
  apply Finsupp.lhom_ext; intro a b; simp [actF_single, mul_assoc]

instance instAct : DistribMulAction SL(2, ℤ) (Ch M) where
  smul γ x := actQ M γ x
  one_smul x := by
    induction x using Submodule.Quotient.induction_on with
    | _ f => show actQ M 1 _ = _; simp [actQ, actF_one]
  mul_smul γ δ x := by
    induction x using Submodule.Quotient.induction_on with
    | _ f => show actQ M (γ * δ) _ = actQ M γ (actQ M δ _); simp [actQ, actF_mul]
  smul_zero γ := map_zero (actQ M γ)
  smul_add γ := map_add (actQ M γ)

lemma smul_mk (γ : SL(2, ℤ)) (f : SL(2, ℤ) × V →₀ ℤ) :
    γ • (Submodule.Quotient.mk f : Ch M) = Submodule.Quotient.mk (actF γ f) := rfl

/-! ### The simplices `Simp q = (SL(2, ℤ) × V) / ~` with their `SL(2, ℤ)`-action -/

/-- The simplices of `Kᵘ` of one degree (L17-3's `Rel`). -/
def SimpM := Quot (Ku.Rel M.toKeyModel)

/-- The class of `(γ, σ)`. -/
def SimpM.mk (x : SL(2, ℤ) × V) : SimpM M := Quot.mk _ x

lemma rel_lt {x y : SL(2, ℤ) × V} (γ : SL(2, ℤ)) (h : Ku.Rel M.toKeyModel x y) :
    Ku.Rel M.toKeyModel (γ * x.1, x.2) (γ * y.1, y.2) := by
  have e : (γ * x.1)⁻¹ * (γ * y.1) = x.1⁻¹ * y.1 := by group
  unfold Ku.Rel; rw [e]; exact h

instance instSimpAct : MulAction SL(2, ℤ) (SimpM M) where
  smul γ := Quot.map (fun x => (γ * x.1, x.2)) (fun _ _ h => rel_lt M γ h)
  one_smul s := by
    induction s using Quot.ind with
    | _ x => show Quot.mk _ _ = _; simp
  mul_smul γ δ s := by
    induction s using Quot.ind with
    | _ x => show Quot.mk _ _ = Quot.mk _ _; simp [mul_assoc]

lemma smul_simpMk (γ : SL(2, ℤ)) (x : SL(2, ℤ) × V) : γ • SimpM.mk M x = SimpM.mk M (γ * x.1, x.2) := rfl

lemma simpMk_eq_iff (x y : SL(2, ℤ) × V) : SimpM.mk M x = SimpM.mk M y ↔ Ku.Rel M.toKeyModel x y := by
  constructor
  · intro h; exact (Quot.eqvGen_exact h).rec (fun _ _ h => h) (fun x => (rel_equivalence _).refl x)
      (fun _ _ _ h => (rel_equivalence _).symm h) (fun _ _ _ _ _ h₁ h₂ => (rel_equivalence _).trans h₁ h₂)
  · exact Quot.sound

variable (D : XBulkDatum)

/-- `XBulk.Simp D q`: the `q`-simplices of `X_bulk ≅ Kᵘ`. -/
abbrev Simp (q : ℤ) := SimpM (D.M q)

/-- `XBulk.chains D q = C_q(X_bulk)`. -/
abbrev chains (q : ℤ) := Ch (D.M q)

/-! ### The boundary -/

/-- `∂(γ, τ) = ∑_j γ · face τ j` on the free module. -/
def bdF (q : ℤ) : (SL(2, ℤ) × D.V (q + 1) →₀ ℤ) →ₗ[ℤ] (SL(2, ℤ) × D.V q →₀ ℤ) :=
  Finsupp.linearCombination ℤ (fun x => ∑ j, gen (OrPt.lt x.1 (D.face q x.2 j)))

lemma bdF_single (q : ℤ) (g : SL(2, ℤ)) (τ : D.V (q + 1)) (c : ℤ) :
    bdF D q (Finsupp.single (g, τ) c) = c • ∑ j, gen (OrPt.lt g (D.face q τ j)) := by
  simp [bdF, Finsupp.linearCombination_single]

lemma bdF_gen (q : ℤ) (x : OrPt (D.V (q + 1))) :
    bdF D q (gen x) = ((x.2.2 : ℤˣ) : ℤ) • ∑ j, gen (OrPt.lt x.1 (D.face q x.2.1 j)) := by
  rw [gen, bdF_single]

lemma sgn_sq (u : ℤˣ) : u * u = 1 := by rcases Int.units_eq_one_or u with e | e <;> rw [e] <;> decide

lemma relSub_le_bdF (q : ℤ) : relSub (D.M (q + 1)) ≤ (relSub (D.M q)).comap (bdF D q) := by
  rw [relSub, Submodule.span_le]
  rintro _ ⟨x, y, h, rfl⟩
  simp only [SetLike.mem_coe, Submodule.mem_comap, map_sub]
  rw [← Submodule.Quotient.mk_eq_zero, Submodule.Quotient.mk_sub, sub_eq_zero]
  obtain ⟨π, hπ⟩ := D.face_transport q h.1
  set s := (D.M (q + 1)).sgn x.2.1 y.2.1 (x.1⁻¹ * y.1)
  have hj : ∀ j, (Submodule.Quotient.mk (gen (OrPt.lt x.1 (D.face q x.2.1 j))) : chains D q) =
      (s : ℤ) • Submodule.Quotient.mk (gen (OrPt.lt y.1 (D.face q y.2.1 (π j)))) := by
    intro j
    have := mk_gen_eq (D.M q) (orRel_lt (D.M q) x.1 (hπ j))
    rw [Submodule.mkQ_apply, Submodule.mkQ_apply] at this
    rw [this, OrPt.lt_tr, mul_inv_cancel_left, gen_tr, Submodule.Quotient.mk_smul]
  rw [bdF_gen, bdF_gen, Submodule.Quotient.mk_smul, Submodule.Quotient.mk_smul, quot_mk_sum,
    quot_mk_sum]
  simp_rw [hj]
  rw [← Finset.smul_sum, smul_smul, ← Equiv.sum_comp π (fun k =>
    (Submodule.Quotient.mk (gen (OrPt.lt y.1 (D.face q y.2.1 k))) : chains D q))]
  congr 1
  rw [h.2, Units.val_mul, mul_comm, ← mul_assoc, ← Units.val_mul, sgn_sq, Units.val_one, one_mul]

/-- **The boundary** `∂ : C_{q+1} → C_q`. -/
def bd (q : ℤ) : chains D (q + 1) →ₗ[ℤ] chains D q :=
  (relSub (D.M (q + 1))).mapQ (relSub (D.M q)) (bdF D q) (relSub_le_bdF D q)

lemma bd_mk (q : ℤ) (f : SL(2, ℤ) × D.V (q + 1) →₀ ℤ) :
    bd D q (Submodule.Quotient.mk f) = Submodule.Quotient.mk (bdF D q f) := rfl

/-- `∂` is `SL(2, ℤ)`-equivariant. -/
lemma bd_equiv (q : ℤ) (γ : SL(2, ℤ)) (x : chains D (q + 1)) : bd D q (γ • x) = γ • bd D q x := by
  induction x using Submodule.Quotient.induction_on with
  | _ f =>
    rw [smul_mk, bd_mk, bd_mk, smul_mk]
    congr 1
    induction f using Finsupp.induction_linear with
    | zero => simp
    | add f g hf hg => simp only [map_add, hf, hg]
    | single a c =>
      rw [actF_single, bdF_single, bdF_single, map_smul, map_sum]
      simp only [actF_gen, OrPt.lt, mul_assoc]

/-- `∂ ∂ = 0` on generators, from the signed simplicial identity (pairwise cancellation). -/
lemma bdF_bdF_mem (q : ℤ) (g : SL(2, ℤ)) (τ : D.V (q + 1 + 1)) :
    bdF D q (bdF D (q + 1) (Finsupp.single (g, τ) 1)) ∈ relSub (D.M q) := by
  rw [← Submodule.Quotient.mk_eq_zero, bdF_single, one_smul, map_sum]
  obtain ⟨ι, hι, hrel⟩ := D.face_simplicial q τ
  have expand : ∀ j, bdF D q (gen (OrPt.lt g (D.face (q + 1) τ j))) =
      ∑ i, gen (OrPt.lt g (OrPt.comp2 (D.face (q + 1) τ j) (D.face q (D.face (q + 1) τ j).2.1 i))) := by
    intro j
    rw [bdF_gen, Finset.smul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp [gen, OrPt.lt, OrPt.comp2, Finsupp.smul_single, mul_assoc]
  simp_rw [expand]
  rw [← Fintype.sum_prod_type', quot_mk_sum]
  set f : D.F (q + 1) × D.F q → chains D q := fun p => Submodule.Quotient.mk
    (gen (OrPt.lt g (OrPt.comp2 (D.face (q + 1) τ p.1) (D.face q (D.face (q + 1) τ p.1).2.1 p.2))))
  show ∑ p, f p = 0
  have hpair : ∀ p, f p + f (ι p) = 0 := by
    intro p
    have := mk_gen_eq (D.M q) (orRel_lt (D.M q) g (hrel p))
    rw [Submodule.mkQ_apply, Submodule.mkQ_apply] at this
    simp only [f]
    rw [this]
    have hn : OrPt.lt g (OrPt.neg (OrPt.comp2 (D.face (q + 1) τ (ι p).1) (D.face q (D.face (q + 1) τ (ι p).1).2.1 (ι p).2)))
        = OrPt.neg (OrPt.lt g (OrPt.comp2 (D.face (q + 1) τ (ι p).1)
          (D.face q (D.face (q + 1) τ (ι p).1).2.1 (ι p).2))) := rfl
    rw [hn, gen_neg, Submodule.Quotient.mk_neg, neg_add_cancel]
  have hne : ∀ p, ι p ≠ p := by
    intro p e
    have := hrel p
    rw [e] at this
    exact not_orRel_neg (D.M q) _ this
  exact Finset.sum_involution (fun p _ => ι p) (fun p _ => hpair p) (fun p _ _ => hne p)
    (fun p _ => Finset.mem_univ _) (fun p _ => hι p)

/-- **`∂∂ = 0`.** -/
theorem bd_bd (q : ℤ) (x : chains D (q + 1 + 1)) : bd D q (bd D (q + 1) x) = 0 := by
  induction x using Submodule.Quotient.induction_on with
  | _ f =>
    rw [bd_mk, bd_mk, Submodule.Quotient.mk_eq_zero]
    induction f using Finsupp.induction_linear with
    | zero => simp
    | add f g hf hg => simp only [map_add]; exact Submodule.add_mem _ hf hg
    | single a c =>
      have : Finsupp.single a c = c • Finsupp.single a 1 := by simp [Finsupp.smul_single]
      rw [this, map_smul, map_smul]
      exact Submodule.smul_mem _ _ (bdF_bdF_mem D q a.1 a.2)

/-! ### The augmentation -/

/-- `ε(γ, v) = 1` on the free module. -/
def augF : (SL(2, ℤ) × D.V 0 →₀ ℤ) →ₗ[ℤ] ℤ := Finsupp.linearCombination ℤ (fun _ => (1 : ℤ))

lemma augF_gen (x : OrPt (D.V 0)) : augF D (gen x) = (x.2.2 : ℤ) := by
  simp [augF, gen, Finsupp.linearCombination_single]

lemma relSub_le_augF : relSub (D.M 0) ≤ LinearMap.ker (augF D) := by
  rw [relSub, Submodule.span_le]
  rintro _ ⟨x, y, h, rfl⟩
  simp only [SetLike.mem_coe, LinearMap.mem_ker, map_sub, augF_gen, h.2, D.sgn0 h.1, one_mul, sub_self]

/-- **The augmentation** `ε : C_0 → ℤ`. -/
def aug : chains D 0 →ₗ[ℤ] ℤ := (relSub (D.M 0)).liftQ (augF D) (relSub_le_augF D)

lemma aug_bd (x : chains D (0 + 1)) : aug D (bd D 0 x) = 0 := by
  induction x using Submodule.Quotient.induction_on with
  | _ f =>
    rw [bd_mk]
    show augF D (bdF D 0 f) = 0
    induction f using Finsupp.induction_linear with
    | zero => simp
    | add f g hf hg => simp only [map_add, hf, hg, add_zero]
    | single a c =>
      rw [bdF_single, map_smul, map_sum]
      simp only [augF_gen, OrPt.lt, D.aug_face, smul_zero]

lemma aug_equiv (γ : SL(2, ℤ)) (x : chains D 0) : aug D (γ • x) = aug D x := by
  induction x using Submodule.Quotient.induction_on with
  | _ f =>
    rw [smul_mk]
    show augF D (actF γ f) = augF D f
    induction f using Finsupp.induction_linear with
    | zero => simp
    | add f g hf hg => simp only [map_add, hf, hg]
    | single a c => simp [actF_single, augF, Finsupp.linearCombination_single]

/-! ### The `GChainCx` -/

/-- **`X_bulk`'s chain complex as the cone layer's `GChainCx`**, for the action of any subgroup `H ≤ SL(2, ℤ)`. -/
def toGChainCxOn (H : Subgroup SL(2, ℤ)) : GL2.Cone.GChainCx.{0} H where
  X q := chains D q
  act _ := DistribMulAction.compHom _ H.subtype
  bd q := (bd D q).toAddMonoidHom
  bd_equiv q γ x := bd_equiv D q (γ : SL(2, ℤ)) x
  bd_bd q x := bd_bd D q x
  aug := (aug D).toAddMonoidHom
  aug_bd x := aug_bd D x
  aug_equiv γ x := aug_equiv D (γ : SL(2, ℤ)) x

/-- **`X_bulk`'s chain complex as a `GChainCx SL(2, ℤ)`.** -/
def toGChainCx : GL2.Cone.GChainCx.{0} SL(2, ℤ) where
  X q := chains D q
  bd q := (bd D q).toAddMonoidHom
  bd_equiv q γ x := bd_equiv D q γ x
  bd_bd q x := bd_bd D q x
  aug := (aug D).toAddMonoidHom
  aug_bd x := aug_bd D x
  aug_equiv γ x := aug_equiv D γ x

end XBulk

end GL2.Model

end
