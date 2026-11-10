import GL2.Model.Psi

/-!
# M3 / C2 (L19-4): the boundary descriptor `X_∂`, the pair `(X_bulk, X_∂)` and its `PairHyp`

[GT] `def:xbdry`, `thm:xbdry-pair`, `thm:xbdry-free`, `thm:pairhyp-xbdry`, mirroring [C] v1.18.0 `def:pair`, (B1)–(B3),
`eq:contractibleuse` and `lem:etaexists`'s hypotheses.

**The descriptor.** `XBdryDatum D` is a *sub-descriptor* of an `X_bulk` descriptor `D`: a decidable set `B q` of model
`q`-simplices which is

* `Γ`-saturated (`closed_S`: a transporter `t ∈ S(σ, σ')` carries `B` to `B`), so `X_∂ = Γ · ι(B)` is a union of orbits;
* closed under faces (`closed_face`), so `X_∂` is a subcomplex;
* of dimension `≤ 1` (`dim1`, [C] (B2));
* with central stabilisers (`central`: `S(σ, σ) ⊆ {±1}` for `σ ∈ B`, [C] (B1)).

`XBdryDatum.toXBulk` is `X_∂` as an `X_bulk` descriptor in its own right (same transporters, signs and faces), and
`inc` is the inclusion `j : C_•(X_∂) → C_•(X_bulk)` -- equivariant, a chain map, compatible with the augmentations
(`inc_equiv`, `inc_bd`, `inc_aug`).  `pairVia e` is the cone layer's `PairData` for `Γ` acting through any
`e : Γ →* SL(2, ℤ)`.

**The (T1)-extension property** (`free_twisted`, [C] (B1) + (B3)).  If `-1` lies in every stabiliser of the bulk (so it
acts trivially on `C_•(X_bulk)`, `negOne_smul`) and `g⁻¹ (±1) g` acts as `±1` (`hc`; true for `−1` central), then a
`g`-twisted map out of `C_q(X_∂)` is **freely** determined by its values on one generator per `Γ''`-orbit of
`q`-simplices, `Γ'' = Γ ∩ g Γ g⁻¹`: the index type is the orbit set itself (`orbSetoid`), the representatives are
`genQ` of `Quotient.out`, and the extension is `W` (the sign-tracked translate), well defined because a `Γ''`-element
stabilising a boundary simplex is `±1` and so is its conjugate.

`pairHyp_via` is then [C]'s full pair hypothesis from `Acyclic D` (H4(ii)) alone: nothing about `X_∂` is assumed beyond
the descriptor.
-/

noncomputable section

namespace GL2.Model

open scoped MatrixGroups
open GL2.Ku GL2.Cone

/-- **`def:xbdry`**: a boundary sub-descriptor `X_∂ ⊆ X_bulk` ([C] (B1), (B2); `Γ`-saturated and face-closed). -/
structure XBdryDatum (D : XBulkDatum) where
  B : ∀ q, D.V q → Prop
  [decB : ∀ q, DecidablePred (B q)]
  closed_S : ∀ q {σ σ' : D.V q} {t : SL(2, ℤ)}, t ∈ (D.M q).S σ σ' → B q σ → B q σ'
  closed_face : ∀ q (τ : D.V (q + 1)) (j : D.F q), B (q + 1) τ → B q (D.face q τ j).2.1
  dim1 : ∀ q (σ : D.V q), B q σ → q ≤ 1
  central : ∀ q (σ : D.V q), B q σ → ∀ t ∈ (D.M q).S σ σ, t = 1 ∨ t = -1

attribute [instance] XBdryDatum.decB

namespace XBdryDatum

variable {D : XBulkDatum} (X : XBdryDatum D)

/-- The boundary model simplices of degree `q`. -/
abbrev Vb (q : ℤ) := {σ : D.V q // X.B q σ}

/-- The oriented key model of `X_∂`: `D`'s, restricted (the transporter sets and signs are `D`'s). -/
def subM (q : ℤ) : OrKeyModel (X.Vb q) where
  rep σ := ⟨(D.M q).rep σ.1, by obtain ⟨t, ht⟩ := (D.M q).rep_mem σ.1; exact X.closed_S q ht σ.2⟩
  S a b := (D.M q).S a.1 b.1
  one_mem σ := (D.M q).one_mem σ.1
  mul_mem h1 h2 := (D.M q).mul_mem h1 h2
  inv_mem h := (D.M q).inv_mem h
  rep_mem σ := (D.M q).rep_mem σ.1
  rep_eq h := Subtype.ext ((D.M q).rep_eq h)
  sgn a b t := (D.M q).sgn a.1 b.1 t
  sgn_mul h1 h2 := (D.M q).sgn_mul h1 h2
  noInv h := (D.M q).noInv h

/-- An oriented boundary point as an oriented bulk point. -/
def emb {q : ℤ} (x : OrPt (X.Vb q)) : OrPt (D.V q) := (x.1, x.2.1.1, x.2.2)

lemma orRel_emb {q : ℤ} {x y : OrPt (X.Vb q)} : OrRel (X.subM q) x y ↔ OrRel (D.M q) (X.emb x) (X.emb y) := Iff.rfl

/-- **`X_∂` as an `X_bulk` descriptor** (dimension `≤ 1`, the faces of `D`). -/
def toXBulk : XBulkDatum where
  V q := X.Vb q
  M q := X.subM q
  F q := D.F q
  face q τ j := ((D.face q τ.1 j).1, ⟨(D.face q τ.1 j).2.1, X.closed_face q τ.1 j τ.2⟩, (D.face q τ.1 j).2.2)
  face_transport q τ τ' t ht := by
    obtain ⟨π, hπ⟩ := D.face_transport q (τ := τ.1) (τ' := τ'.1) ht
    exact ⟨π, fun j => hπ j⟩
  face_simplicial q τ := by
    obtain ⟨ι, hι, h⟩ := D.face_simplicial q τ.1
    exact ⟨ι, hι, fun p => h p⟩
  sgn0 h := D.sgn0 h
  aug_face τ := D.aug_face τ.1
  dim2 q hq := ⟨fun σ => (D.dim2 q hq).false σ.1⟩

/-- `X_∂` has no simplices above dimension `1` ([C] (B2)). -/
lemma isEmpty_dim (q : ℤ) (hq : 2 ≤ q) : IsEmpty (X.toXBulk.V q) :=
  ⟨fun σ => by have := X.dim1 q σ.1 σ.2; omega⟩

/-! ### The inclusion `j : C_•(X_∂) → C_•(X_bulk)` -/

/-- `j` on the free modules. -/
def incF (q : ℤ) : (SL(2, ℤ) × X.toXBulk.V q →₀ ℤ) →ₗ[ℤ] (SL(2, ℤ) × D.V q →₀ ℤ) :=
  Finsupp.lmapDomain ℤ ℤ (fun x => (x.1, x.2.1))

lemma incF_single (q : ℤ) (a : SL(2, ℤ) × X.toXBulk.V q) (c : ℤ) :
    X.incF q (Finsupp.single a c) = Finsupp.single (a.1, a.2.1) c := by
  simp [incF, Finsupp.mapDomain_single]

lemma incF_gen (q : ℤ) (x : OrPt (X.toXBulk.V q)) : X.incF q (XBulk.gen x) = XBulk.gen (X.emb x) := by
  simp [incF, XBulk.gen, emb, Finsupp.mapDomain_single]

lemma relSub_le_incF (q : ℤ) : XBulk.relSub (X.toXBulk.M q) ≤ (XBulk.relSub (D.M q)).comap (X.incF q) := by
  rw [XBulk.relSub, Submodule.span_le]
  rintro _ ⟨x, y, h, rfl⟩
  simp only [SetLike.mem_coe, Submodule.mem_comap, map_sub, incF_gen]
  exact Submodule.subset_span ⟨_, _, (X.orRel_emb).1 h, rfl⟩

/-- **The inclusion** `j : C_q(X_∂) → C_q(X_bulk)`. -/
def inc (q : ℤ) : XBulk.chains X.toXBulk q →ₗ[ℤ] XBulk.chains D q :=
  (XBulk.relSub (X.toXBulk.M q)).mapQ (XBulk.relSub (D.M q)) (X.incF q) (X.relSub_le_incF q)

lemma inc_mk (q : ℤ) (f : SL(2, ℤ) × X.toXBulk.V q →₀ ℤ) :
    X.inc q (Submodule.Quotient.mk f : XBulk.chains X.toXBulk q) =
      (Submodule.Quotient.mk (X.incF q f) : XBulk.chains D q) := rfl

lemma inc_genQ (q : ℤ) (a : SL(2, ℤ) × X.toXBulk.V q) :
    X.inc q (XBulk.genQ X.toXBulk q a) = XBulk.genQ D q (a.1, a.2.1) := by
  unfold XBulk.genQ
  rw [inc_mk, incF_gen]; rfl

/-- `j` is `SL(2, ℤ)`-equivariant. -/
lemma inc_equiv (q : ℤ) (γ : SL(2, ℤ)) (x : XBulk.chains X.toXBulk q) : X.inc q (γ • x) = γ • X.inc q x := by
  induction x using Submodule.Quotient.induction_on with
  | _ f =>
    rw [XBulk.smul_mk, inc_mk, inc_mk, XBulk.smul_mk]
    congr 1
    induction f using Finsupp.induction_linear with
    | zero => simp
    | add f g hf hg => simp only [map_add, hf, hg]
    | single a c => rw [XBulk.actF_single, incF_single, incF_single, XBulk.actF_single]

/-- `j` is a chain map. -/
lemma inc_bd (q : ℤ) (x : XBulk.chains X.toXBulk (q + 1)) :
    XBulk.bd D q (X.inc (q + 1) x) = X.inc q (XBulk.bd X.toXBulk q x) := by
  induction x using Submodule.Quotient.induction_on with
  | _ f =>
    rw [inc_mk, XBulk.bd_mk, XBulk.bd_mk, inc_mk]
    congr 1
    induction f using Finsupp.induction_linear with
    | zero => simp
    | add f g hf hg => simp only [map_add, hf, hg]
    | single a c =>
      rw [incF_single, XBulk.bdF_single, XBulk.bdF_single, map_smul, map_sum]
      simp only [incF_gen]
      rfl

/-- `j` commutes with the augmentations ([C] `def:pair`). -/
lemma inc_aug (x : XBulk.chains X.toXBulk 0) : XBulk.aug D (X.inc 0 x) = XBulk.aug X.toXBulk x := by
  induction x using Submodule.Quotient.induction_on with
  | _ f =>
    rw [inc_mk]
    show XBulk.augF D (X.incF 0 f) = XBulk.augF X.toXBulk f
    induction f using Finsupp.induction_linear with
    | zero => simp
    | add f g hf hg => simp only [map_add, hf, hg]
    | single a c => simp [incF_single, XBulk.augF, Finsupp.linearCombination_single]

variable {G : Type} [Group G] {Γ : Subgroup G} (e : Γ →* SL(2, ℤ))

/-- **The pair `(X_bulk, X_∂)`** as the cone layer's `PairData`, `Γ` acting through `e`. -/
def pairVia : PairData.{0} Γ where
  bulk := XBulk.toGChainCxVia D e
  bdry := XBulk.toGChainCxVia X.toXBulk e
  j q := (X.inc q).toAddMonoidHom
  j_equiv q γ x := X.inc_equiv q (e γ) x
  j_bd q x := X.inc_bd q x

lemma pairVia_jaug (x : (X.pairVia e).bdry.X 0) :
    (X.pairVia e).bulk.aug ((X.pairVia e).j 0 x) = (X.pairVia e).bdry.aug x := X.inc_aug x

/-! ### The (T1)-extension property ([C] (B1) + (B3)) -/

lemma neg_one_comm (u : SL(2, ℤ)) : (-1 : SL(2, ℤ)) * u = u * -1 := by
  rw [neg_one_mul, mul_neg_one]

/-- `-1` acts trivially on the bulk chains when it lies in every stabiliser ([C] (B3)). -/
lemma negOne_smul (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ) {q : ℤ} (y : XBulk.chains D q) :
    (-1 : SL(2, ℤ)) • y = y := by
  have key : (DistribSMul.toAddMonoidHom (XBulk.chains D q) (-1 : SL(2, ℤ))) = AddMonoidHom.id _ := by
    apply XBulk.chains_hom_ext
    intro a
    simp only [DistribSMul.toAddMonoidHom_apply, AddMonoidHom.id_apply, XBulk.smul_genQ]
    simp only [XBulk.genQ]
    have hr : OrRel (D.M q) (((-1 : SL(2, ℤ)) * a.1, a.2, 1) : OrPt (D.V q)) (a.1, a.2, 1) := by
      refine ⟨?_, ?_⟩
      · show ((-1 : SL(2, ℤ)) * a.1)⁻¹ * a.1 ∈ (D.M q).S a.2 a.2
        have : ((-1 : SL(2, ℤ)) * a.1)⁻¹ * a.1 = (-1 : SL(2, ℤ))⁻¹ := by rw [neg_one_comm]; group
        rw [this]; exact (D.M q).inv_mem (hneg q a.2)
      · show (1 : ℤˣ) = (D.M q).sgn a.2 a.2 (((-1 : SL(2, ℤ)) * a.1)⁻¹ * a.1) * 1
        have : ((-1 : SL(2, ℤ)) * a.1)⁻¹ * a.1 = (-1 : SL(2, ℤ))⁻¹ := by rw [neg_one_comm]; group
        rw [this, (D.M q).noInv ((D.M q).inv_mem (hneg q a.2)), one_mul]
    have := XBulk.mk_gen_eq (D.M q) hr
    simpa only [Submodule.mkQ_apply] using this
  exact DFunLike.congr_fun key y

/-- An element acting as `±1` acts trivially on the bulk chains. -/
lemma pm_smul (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ) {q : ℤ} {u : SL(2, ℤ)}
    (hu : u = 1 ∨ u = -1) (y : XBulk.chains D q) : u • y = y := by
  rcases hu with rfl | rfl
  · exact one_smul _ y
  · exact negOne_smul hneg y

variable (g : G)

/-- The `Γ''`-orbit relation on the generators `(γ, σ)` of `C_q(X_∂)`: `(γ', σ') ∈ Γ'' · (γ, σ)` as simplices,
`Γ'' = {h ∈ Γ : g⁻¹ h g ∈ Γ}`. -/
def orbSetoid (q : ℤ) : Setoid (SL(2, ℤ) × X.toXBulk.V q) where
  r x y := ∃ h : Γ, g⁻¹ * (h : G) * g ∈ Γ ∧
    e h • XBulk.SimpM.mk (X.toXBulk.M q) x = XBulk.SimpM.mk (X.toXBulk.M q) y
  iseqv := by
    refine ⟨fun x => ⟨1, by simp, by simp⟩, ?_, ?_⟩
    · rintro x y ⟨h, hm, hh⟩
      refine ⟨h⁻¹, ?_, ?_⟩
      · have : g⁻¹ * ((h⁻¹ : Γ) : G) * g = (g⁻¹ * h * g)⁻¹ := by simp only [Subgroup.coe_inv]; group
        rw [this]; exact Γ.inv_mem hm
      · rw [← hh, map_inv, inv_smul_smul]
    · rintro x y z ⟨h, hm, hh⟩ ⟨k, km, hk⟩
      refine ⟨k * h, ?_, ?_⟩
      · have : g⁻¹ * ((k * h : Γ) : G) * g = (g⁻¹ * k * g) * (g⁻¹ * h * g) := by
          simp only [Subgroup.coe_mul]; group
        rw [this]; exact Γ.mul_mem km hm
      · rw [map_mul, mul_smul, hh, hk]

/-- The orbit set: the index type of the free generators. -/
abbrev Orb (q : ℤ) := Quotient (X.orbSetoid e g q)

/-- The orbit representative's generator. -/
def orbRep (q : ℤ) (i : X.Orb e g q) : XBulk.chains X.toXBulk q := XBulk.genQ X.toXBulk q i.out

variable {X e g}

/-- A valid pair `(h, h')` for `(i, z)`: `h' = g⁻¹ h g` and `h · ι(out i) = z` as simplices. -/
def Valid {q : ℤ} (i : X.Orb e g q) (z : SL(2, ℤ) × X.toXBulk.V q) (h h' : Γ) : Prop :=
  (h' : G) = g⁻¹ * h * g ∧
    e h • XBulk.SimpM.mk (X.toXBulk.M q) i.out = XBulk.SimpM.mk (X.toXBulk.M q) z

/-- The sign-tracked translate `sgn(σ_o, σ_z, (h γ_o)⁻¹ γ_z) · h' · v`. -/
def W {b : ℤ} {q : ℤ} (i : X.Orb e g q) (z : SL(2, ℤ) × X.toXBulk.V q) (h h' : Γ) (v : XBulk.chains D b) :
    XBulk.chains D b :=
  (((X.toXBulk.M q).sgn i.out.2 z.2 ((e h * i.out.1)⁻¹ * z.1) : ℤˣ) : ℤ) • (e h' • v)

lemma valid_mem {q : ℤ} {i : X.Orb e g q} {z : SL(2, ℤ) × X.toXBulk.V q} {h h' : Γ} (hv : Valid i z h h') :
    (e h * i.out.1)⁻¹ * z.1 ∈ (X.toXBulk.M q).S i.out.2 z.2 := by
  have := hv.2
  rw [XBulk.smul_simpMk, XBulk.simpMk_eq_iff] at this
  exact this

/-- **Independence of the choice**: two valid pairs give the same translate (stabilisers are `±1`, and so are their
conjugates). -/
lemma W_indep (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ)
    (hc : ∀ h h' : Γ, (h' : G) = g⁻¹ * h * g → (e h = 1 ∨ e h = -1) → (e h' = 1 ∨ e h' = -1))
    {b q : ℤ} {i : X.Orb e g q} {z : SL(2, ℤ) × X.toXBulk.V q} {h₁ h₁' h₂ h₂' : Γ}
    (v₁ : Valid i z h₁ h₁') (v₂ : Valid i z h₂ h₂') (v : XBulk.chains D b) :
    W i z h₁ h₁' v = W i z h₂ h₂' v := by
  set o := i.out
  set M := X.toXBulk.M q
  have m₁ := valid_mem v₁
  have m₂ := valid_mem v₂
  -- `w = o₁⁻¹ e(h₂⁻¹ h₁) o₁` stabilises `ι(σ_o)`
  set w : SL(2, ℤ) := ((e h₂ * o.1)⁻¹ * z.1) * ((e h₁ * o.1)⁻¹ * z.1)⁻¹ with hw
  have hwS : w ∈ M.S o.2 o.2 := M.mul_mem m₂ (M.inv_mem m₁)
  have hwpm : w = 1 ∨ w = -1 := X.central q o.2.1 o.2.2 w hwS
  have hk : e (h₂⁻¹ * h₁) = o.1 * w * o.1⁻¹ := by
    rw [hw, map_mul, map_inv]; group
  have hkpm : e (h₂⁻¹ * h₁) = 1 ∨ e (h₂⁻¹ * h₁) = -1 := by
    rw [hk]
    rcases hwpm with h | h <;> rw [h]
    · left; group
    · right; rw [mul_neg_one, neg_mul, mul_inv_cancel]
  have hk' : ((h₂'⁻¹ * h₁' : Γ) : G) = g⁻¹ * ((h₂⁻¹ * h₁ : Γ) : G) * g := by
    simp only [Subgroup.coe_mul, Subgroup.coe_inv, v₁.1, v₂.1]; group
  have hk'pm := hc _ _ hk' hkpm
  have hsmul : e h₁' • v = e h₂' • v := by
    have : h₁' = h₂' * (h₂'⁻¹ * h₁') := by group
    rw [this, map_mul, mul_smul, pm_smul hneg hk'pm]
  have hsgn : M.sgn o.2 z.2 ((e h₂ * o.1)⁻¹ * z.1) = M.sgn o.2 z.2 ((e h₁ * o.1)⁻¹ * z.1) := by
    have e1 : (e h₂ * o.1)⁻¹ * z.1 = w * ((e h₁ * o.1)⁻¹ * z.1) := by rw [hw]; group
    rw [e1, M.sgn_mul hwS m₁, M.noInv hwS, one_mul]
  unfold W
  rw [hsmul, hsgn]

/-- A valid pair always exists for `(⟦z⟧, z)`. -/
lemma exists_valid {q : ℤ} (z : SL(2, ℤ) × X.toXBulk.V q) :
    ∃ h h' : Γ, Valid (⟦z⟧ : X.Orb e g q) z h h' := by
  obtain ⟨h, hm, hh⟩ := Quotient.mk_out (s := X.orbSetoid e g q) z
  exact ⟨h, ⟨_, hm⟩, rfl, hh⟩

variable (X e g)

/-- The value on the generator `z`: `W` at the chosen valid pair. -/
def Wz {b q : ℤ} (val : X.Orb e g q → XBulk.chains D b) (z : SL(2, ℤ) × X.toXBulk.V q) : XBulk.chains D b :=
  W (⟦z⟧ : X.Orb e g q) z (Classical.choose (exists_valid (X := X) (e := e) (g := g) z))
    (Classical.choose (Classical.choose_spec (exists_valid (X := X) (e := e) (g := g) z))) (val ⟦z⟧)

variable {X e g}

lemma Wz_eq (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ)
    (hc : ∀ h h' : Γ, (h' : G) = g⁻¹ * h * g → (e h = 1 ∨ e h = -1) → (e h' = 1 ∨ e h' = -1))
    {b q : ℤ} (val : X.Orb e g q → XBulk.chains D b) (z : SL(2, ℤ) × X.toXBulk.V q) (i : X.Orb e g q)
    (hi : (⟦z⟧ : X.Orb e g q) = i) {h h' : Γ} (hv : Valid i z h h') : X.Wz e g val z = W i z h h' (val i) := by
  subst hi
  exact W_indep hneg hc (Classical.choose_spec (Classical.choose_spec (exists_valid (X := X) (e := e) (g := g) z))) hv _

/-- The extension on the free module. -/
def freeExt {b q : ℤ} (val : X.Orb e g q → XBulk.chains D b) :
    (SL(2, ℤ) × X.toXBulk.V q →₀ ℤ) →+ XBulk.chains D b :=
  Finsupp.liftAddHom (fun z => zmultiplesHom _ (X.Wz e g val z))

lemma freeExt_single {b q : ℤ} (val : X.Orb e g q → XBulk.chains D b) (z : SL(2, ℤ) × X.toXBulk.V q) (c : ℤ) :
    freeExt val (Finsupp.single z c) = c • X.Wz e g val z := by
  simp [freeExt]

lemma freeExt_gen {b q : ℤ} (val : X.Orb e g q → XBulk.chains D b) (x : OrPt (X.toXBulk.V q)) :
    freeExt val (XBulk.gen x) = ((x.2.2 : ℤˣ) : ℤ) • X.Wz e g val (x.1, x.2.1) := by
  rw [XBulk.gen, freeExt_single]

lemma relSub_le_freeExt (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ)
    (hc : ∀ h h' : Γ, (h' : G) = g⁻¹ * h * g → (e h = 1 ∨ e h = -1) → (e h' = 1 ∨ e h' = -1))
    {b q : ℤ} (val : X.Orb e g q → XBulk.chains D b) :
    XBulk.relSub (X.toXBulk.M q) ≤ LinearMap.ker (freeExt (X := X) (e := e) (g := g) val).toIntLinearMap := by
  rw [XBulk.relSub, Submodule.span_le]
  rintro _ ⟨x, y, hxy, rfl⟩
  simp only [SetLike.mem_coe, LinearMap.mem_ker, map_sub, AddMonoidHom.coe_toIntLinearMap, freeExt_gen, sub_eq_zero]
  set M := X.toXBulk.M q
  set i : X.Orb e g q := ⟦(x.1, x.2.1)⟧
  have hrel : XBulk.SimpM.mk M (x.1, x.2.1) = XBulk.SimpM.mk M (y.1, y.2.1) :=
    (XBulk.simpMk_eq_iff M _ _).2 hxy.1
  have hi : (⟦(y.1, y.2.1)⟧ : X.Orb e g q) = i := Quotient.sound ⟨1, by simp, by rw [map_one, one_smul]; exact hrel.symm⟩
  obtain ⟨h, h', hv⟩ := exists_valid (X := X) (e := e) (g := g) (x.1, x.2.1)
  have hv' : Valid i (y.1, y.2.1) h h' := ⟨hv.1, hv.2.trans hrel⟩
  rw [Wz_eq hneg hc val _ i rfl hv, Wz_eq hneg hc val _ i hi hv']
  unfold W
  rw [smul_smul, smul_smul]
  congr 1
  have m := valid_mem hv
  have e1 : (e h * i.out.1)⁻¹ * y.1 = ((e h * i.out.1)⁻¹ * x.1) * (x.1⁻¹ * y.1) := by group
  simp only
  rw [e1, M.sgn_mul m hxy.1, hxy.2]
  push_cast
  ring

/-- The extension on the chains. -/
def ext (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ)
    (hc : ∀ h h' : Γ, (h' : G) = g⁻¹ * h * g → (e h = 1 ∨ e h = -1) → (e h' = 1 ∨ e h' = -1))
    {b q : ℤ} (val : X.Orb e g q → XBulk.chains D b) : XBulk.chains X.toXBulk q →+ XBulk.chains D b :=
  ((XBulk.relSub (X.toXBulk.M q)).liftQ _ (relSub_le_freeExt hneg hc val)).toAddMonoidHom

lemma ext_genQ (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ)
    (hc : ∀ h h' : Γ, (h' : G) = g⁻¹ * h * g → (e h = 1 ∨ e h = -1) → (e h' = 1 ∨ e h' = -1))
    {b q : ℤ} (val : X.Orb e g q → XBulk.chains D b) (z : SL(2, ℤ) × X.toXBulk.V q) :
    ext hneg hc val (XBulk.genQ X.toXBulk q z) = X.Wz e g val z := by
  show freeExt val (XBulk.gen ((z.1, z.2, 1) : OrPt _)) = _
  rw [freeExt_gen]; simp

/-- Every generator is the sign-tracked `Γ''`-translate of its orbit representative. -/
lemma genQ_eq_translate {q : ℤ} (z : SL(2, ℤ) × X.toXBulk.V q) {i : X.Orb e g q} {h h' : Γ} (hv : Valid i z h h') :
    XBulk.genQ X.toXBulk q z =
      (((X.toXBulk.M q).sgn i.out.2 z.2 ((e h * i.out.1)⁻¹ * z.1) : ℤˣ) : ℤ) • (e h • X.orbRep e g q i) := by
  have m := valid_mem hv
  have hr : OrRel (X.toXBulk.M q) ((z.1, z.2, 1) : OrPt _)
      ((e h * i.out.1, i.out.2, (X.toXBulk.M q).sgn i.out.2 z.2 ((e h * i.out.1)⁻¹ * z.1)) : OrPt _) := by
    refine ⟨(X.toXBulk.M q).inv_mem m |> fun h1 => ?_, ?_⟩
    · have : (z.1)⁻¹ * (e h * i.out.1) = ((e h * i.out.1)⁻¹ * z.1)⁻¹ := by group
      show _ ∈ _; rw [this]; exact h1
    · show (1 : ℤˣ) = (X.toXBulk.M q).sgn z.2 i.out.2 ((z.1)⁻¹ * (e h * i.out.1)) * _
      have : (z.1)⁻¹ * (e h * i.out.1) = ((e h * i.out.1)⁻¹ * z.1)⁻¹ := by group
      rw [this, (X.toXBulk.M q).sgn_inv m, XBulk.sgn_sq]
  have := XBulk.mk_gen_eq (X.toXBulk.M q) hr
  simp only [Submodule.mkQ_apply] at this
  unfold XBulk.genQ
  rw [this, XBulk.mk_gen_eq_smul]
  unfold orbRep
  rw [XBulk.genQ_eq_smul X.toXBulk i.out, smul_smul]

variable (X e g)

/-- **`thm:xbdry-free`, the (T1)-extension property** ([C] (B1) + (B3)): a `g`-twisted map out of `C_q(X_∂)` is freely
determined by its values on the `Γ''`-orbit representatives. -/
theorem free_twisted (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ)
    (hc : ∀ h h' : Γ, (h' : G) = g⁻¹ * h * g → (e h = 1 ∨ e h = -1) → (e h' = 1 ∨ e h' = -1)) (q b : ℤ)
    (val : X.Orb e g q → XBulk.chains D b) :
    ∃! F : XBulk.chains X.toXBulk q →+ XBulk.chains D b,
      (∀ h h' : Γ, (h' : G) = g⁻¹ * h * g → ∀ y, F (e h • y) = e h' • F y) ∧ ∀ i, F (X.orbRep e g q i) = val i := by
  refine ⟨ext hneg hc val, ⟨?_, ?_⟩, ?_⟩
  · -- (T1)
    intro h h' hh y
    have key : (ext hneg hc val).comp (DistribSMul.toAddMonoidHom _ (e h)) =
        (DistribSMul.toAddMonoidHom _ (e h')).comp (ext hneg hc val) := by
      apply XBulk.chains_hom_ext
      intro z
      simp only [AddMonoidHom.comp_apply, DistribSMul.toAddMonoidHom_apply, XBulk.smul_genQ, ext_genQ]
      obtain ⟨k, k', hv⟩ := exists_valid (X := X) (e := e) (g := g) z
      have hv2 : Valid (⟦z⟧ : X.Orb e g q) (e h * z.1, z.2) (h * k) (h' * k') := by
        refine ⟨by simp only [Subgroup.coe_mul, hh, hv.1]; group, ?_⟩
        rw [map_mul, mul_smul, hv.2, XBulk.smul_simpMk]
      have hi : (⟦(e h * z.1, z.2)⟧ : X.Orb e g q) = ⟦z⟧ :=
        Quotient.sound ⟨h⁻¹, by
          have : g⁻¹ * ((h⁻¹ : Γ) : G) * g = (g⁻¹ * h * g)⁻¹ := by simp only [Subgroup.coe_inv]; group
          rw [this, ← hh]; exact Γ.inv_mem h'.2, by
          rw [XBulk.smul_simpMk, map_inv]; simp⟩
      rw [Wz_eq hneg hc val _ _ hi hv2, Wz_eq hneg hc val _ _ rfl hv]
      unfold W
      simp only [map_mul, mul_smul]
      have hs : (e h * e k * (⟦z⟧ : X.Orb e g q).out.1)⁻¹ * (e h * z.1) =
          (e k * (⟦z⟧ : X.Orb e g q).out.1)⁻¹ * z.1 := by group
      rw [hs]
      exact (smul_comm _ _ _).symm
    exact DFunLike.congr_fun key y
  · -- the values on the representatives
    intro i
    unfold orbRep
    refine (ext_genQ hneg hc val i.out).trans ?_
    rw [Wz_eq hneg hc val _ i (Quotient.out_eq i) (h := 1) (h' := 1) ⟨by simp, by simp⟩]
    unfold W
    simp only [map_one, one_mul, inv_mul_cancel, one_smul, OrKeyModel.sgn_one, Units.val_one]
  · -- uniqueness
    rintro F ⟨hF, hFv⟩
    apply XBulk.chains_hom_ext
    intro z
    rw [ext_genQ]
    obtain ⟨k, k', hv⟩ := exists_valid (X := X) (e := e) (g := g) z
    rw [Wz_eq hneg hc val _ _ rfl hv, genQ_eq_translate z hv, map_zsmul]
    unfold W
    congr 1
    rw [hF k k' hv.1 (X.orbRep e g q ⟦z⟧), hFv]

/-- **`thm:pairhyp-xbdry`**: for an acyclic `X_bulk` (H4(ii)) and a boundary descriptor, the pair `(X_bulk, X_∂)`
satisfies [C]'s full pair hypothesis -- nothing about `X_∂` is assumed beyond the descriptor ((B1), (B2)) and (B3). -/
theorem pairHyp_via (hA : XBulk.Acyclic D) (hneg : ∀ q (σ : D.V q), (-1 : SL(2, ℤ)) ∈ (D.M q).S σ σ)
    (hc : ∀ h h' : Γ, (h' : G) = g⁻¹ * h * g → (e h = 1 ∨ e h = -1) → (e h' = 1 ∨ e h' = -1)) :
    PairHyp Γ (X.pairVia e) g where
  neg q hq := ⟨XBulk.subsingleton_neg D q hq, XBulk.subsingleton_neg X.toXBulk q hq⟩
  contract0 := hA.contract0
  contract1 := hA.contract1
  contract2 := hA.contract2
  bulk_dim q hq := XBulk.subsingleton_dim D q hq
  bdry_dim q hq := XBulk.subsingleton_of_isEmpty X.toXBulk q (X.isEmpty_dim q hq)
  free_twisted q := ⟨X.Orb e g q, X.orbRep e g q, fun b val => X.free_twisted e g hneg hc q b val⟩

end XBdryDatum

end GL2.Model

end
