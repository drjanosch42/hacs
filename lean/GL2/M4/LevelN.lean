import GL2.M4.Skeleton
import Mathlib.RepresentationTheory.Homological.GroupCohomology.Functoriality
import Mathlib.NumberTheory.DirichletCharacter.Basic

/-!
# M4-g, M4-h (slot 20, L20-3): level `N` by Shapiro, the level-`N` correspondences, the nebentypus

* **M4-g.** `levelN_shapiro : levelN_shapiro_stmt` (`model7_rmkOrbifold` then mathlib's `groupCohomology.coindIso`);
  the explicit induced fibre `IndN` of [C] `def:inducedmodule` / `prop:blockmatrices` (`eq:MH`: the monomial blocks
  `ρ(h_j(γ))`, `w_j γ = h_j(γ) w_{j·γ}`) for **any** table of representatives `RTrans H`, and `indN_iso_coind :
  IndN T A ≅ Rep.coind H.subtype A`; with it `levelN_shapiro_ind`, [C] `prop:lhcompliant`(4) in [C]'s own fibre.
* **M4-h.** `levelCorr` (`eq:levelcorr` in the form of [C] `prop:levelcorrespondence`(3): under the cochain-level
  Frobenius reciprocity, `ψ ↦ (y ↦ ∑_t ρ(r_t⁻¹ g) ψ(τ(r_t y)))`), `levelCorr_isChainMap` (no hypothesis on `ℓ`),
  and `levelCorr_eq_res_cor`: `levelCorr = cor ∘ (conj_g)_* ∘ res` at the cochain level, every `R` (the instances (4)(b),
  (c) of `prop:levelcorrespondence`, composed), and so on cohomology.  The nebentypus: `Δ0Q N`, `aZ`, `ρχ` (`def:nebdata`,
  `lem:nebdata`), `nebCorr` (`eq:nebcorr`, `prop:nebcorr`(1),(2)), and `nebRelations` (`thm:nebrelations`(2),(3) as
  `\gtmodlit` over the nebentypus forms of the primitive Hecke-ring identities, `T(p,p) = χ(p) c` proved).
-/

noncomputable section

namespace GL2.M4

open scoped MatrixGroups
open GL2.Model GL2.Model.XBulk GL2.Cone GL2.Cosets CategoryTheory

/-! ## M4-g: level `N` by Shapiro -/

/-- **[C] `prop:lhcompliant`(4) on the model, mathlib's `Coind`** (`prop:levelN-shapiro-model`). -/
theorem levelN_shapiro : levelN_shapiro_stmt := fun _ _ h6 _ A n =>
  ⟨(model7_rmkOrbifold h6 _ n).trans (groupCohomology.coindIso A n).toLinearEquiv⟩

section Ind

variable {G : Type} [Group G]

/-- **A table of representatives of `H \ G`** ([C] `def:inducedmodule`, `def:levelellmodel`): `w_1, …, w_m`, and for
every `x` the unique decomposition `x = h(x) · w_{j(x)}`, `h(x) ∈ H`.  Any table: [C] fixes one
(`def:cosetrepsN`) only because the cochain-level matrices depend on it. -/
structure RTrans (H : Subgroup G) where
  m : ℕ
  w : Fin m → G
  idx : G → Fin m
  hp : G → H
  decomp : ∀ x, x = (hp x : G) * w (idx x)
  uniq : ∀ (h : H) (j : Fin m), idx ((h : G) * w j) = j ∧ hp ((h : G) * w j) = h

namespace RTrans

variable {H : Subgroup G} (T : RTrans H)

lemma idx_mul (h : H) (x : G) : T.idx ((h : G) * x) = T.idx x ∧ T.hp ((h : G) * x) = h * T.hp x := by
  have e : (h : G) * x = ((h * T.hp x : H) : G) * T.w (T.idx x) := by
    rw [Subgroup.coe_mul, mul_assoc, ← T.decomp]
  rw [e]
  exact T.uniq _ _

/-- The cocycle rule of [C] `prop:blockmatrices`: `j·(xy) = (j·x)·y`, `h(xy) = h(x) h_{j(x)}(y)`. -/
lemma split (x y : G) :
    T.idx (x * y) = T.idx (T.w (T.idx x) * y) ∧ T.hp (x * y) = T.hp x * T.hp (T.w (T.idx x) * y) := by
  have e : x * y = (T.hp x : G) * (T.w (T.idx x) * y) := by
    conv_lhs => rw [T.decomp x]
    rw [mul_assoc]
  rw [e]
  exact T.idx_mul _ _

lemma idx_w (j : Fin T.m) : T.idx (T.w j) = j := by
  simpa using (T.uniq 1 j).1

lemma hp_w (j : Fin T.m) : T.hp (T.w j) = 1 := by
  simpa using (T.uniq 1 j).2

/-- A table exists for every subgroup of finite index (right cosets, `Quotient.out`). -/
def ofFiniteIndex (H : Subgroup G) [H.FiniteIndex] : RTrans H :=
  haveI : Finite (Quotient (QuotientGroup.rightRel H)) :=
    Finite.of_equiv _ (QuotientGroup.quotientRightRelEquivQuotientLeftRel H).symm
  let e := Finite.equivFin (Quotient (QuotientGroup.rightRel H))
  have hrel : ∀ x : G, x * ((e.symm (e (Quotient.mk _ x))).out)⁻¹ ∈ H := fun x => by
    rw [Equiv.symm_apply_apply]
    exact QuotientGroup.rightRel_apply.mp (Quotient.exact (Quotient.out_eq (Quotient.mk (QuotientGroup.rightRel H) x)))
  { m := Nat.card (Quotient (QuotientGroup.rightRel H))
    w := fun j => (e.symm j).out
    idx := fun x => e (Quotient.mk _ x)
    hp := fun x => ⟨_, hrel x⟩
    decomp := fun x => by simp
    uniq := fun h j => by
      have hq : (Quotient.mk (QuotientGroup.rightRel H) ((h : G) * (e.symm j).out)) = e.symm j := by
        conv_rhs => rw [← Quotient.out_eq (e.symm j)]
        refine Quotient.sound (QuotientGroup.rightRel_apply.mpr ?_)
        simp
      refine ⟨by rw [hq, Equiv.apply_symm_apply], Subtype.ext ?_⟩
      simp only [hq, Equiv.symm_apply_apply]
      group }

end RTrans

variable {k : Type} [CommRing k] {H : Subgroup G} (T : RTrans H)

/-- **[C] `eq:MH`**: `(γ · f)_j = ρ(h_j(γ)) f_{j·γ}` on `Fin m → A`, `w_j γ = h_j(γ) w_{j·γ}`. -/
def indRep (A : Rep k H) : Representation k G (Fin T.m → A) where
  toFun γ := LinearMap.pi fun j => (A.ρ (T.hp (T.w j * γ))).comp (LinearMap.proj (T.idx (T.w j * γ)))
  map_one' := by
    apply LinearMap.ext; intro f; funext j
    simp [T.idx_w, T.hp_w]
  map_mul' γ₁ γ₂ := by
    apply LinearMap.ext; intro f; funext j
    obtain ⟨h1, h2⟩ := T.split (T.w j * γ₁) γ₂
    simp only [LinearMap.pi_apply, LinearMap.coe_comp, Function.comp_apply, LinearMap.coe_proj,
      Function.eval, Module.End.mul_apply]
    rw [← mul_assoc, h1, h2, map_mul, Module.End.mul_apply]

lemma indRep_apply (A : Rep k H) (γ : G) (f : Fin T.m → A) (j : Fin T.m) :
    indRep T A γ f j = A.ρ (T.hp (T.w j * γ)) (f (T.idx (T.w j * γ))) := rfl

/-- **[C] `def:levelellmodel`**: the explicit induced fibre `Ind_H^G A`, coordinates `(f(w_j))_j`, action `eq:MH`. -/
abbrev IndN (A : Rep k H) : Rep k G := Rep.of (indRep T A)

/-- `f ↦ (x ↦ ρ(h(x)) f_{j(x)})`, the inverse of the restriction to the representatives. -/
def indToCoind (A : Rep k H) : (Fin T.m → A) →ₗ[k] Representation.coindV H.subtype A.ρ where
  toFun f := ⟨fun x => A.ρ (T.hp x) (f (T.idx x)), fun h x => by
    obtain ⟨h1, h2⟩ := T.idx_mul h x
    simp only [Subgroup.coe_subtype, h1, h2, map_mul, Module.End.mul_apply]⟩
  map_add' f f' := by ext x; simp
  map_smul' c f := by ext x; simp

/-- **`indN_iso_coind`** ([C] `def:inducedmodule`: restriction to the `w_j` is bijective, `prop:blockmatrices`): the
explicit fibre is mathlib's coinduced representation. -/
def indN_iso_coind (A : Rep k H) : IndN T A ≅ Rep.coind H.subtype A :=
  Rep.mkIso (Representation.Equiv.mk
    { indToCoind T A with
      invFun := fun F j => F.1 (T.w j)
      left_inv := fun f => by
        funext j
        simp [indToCoind, T.idx_w, T.hp_w]
      right_inv := fun F => by
        apply Subtype.ext; funext x
        simp only [indToCoind]
        rw [← F.2 (T.hp x) (T.w (T.idx x))]
        simp only [Subgroup.coe_subtype]
        rw [← T.decomp] }
    (fun γ => by
      apply LinearMap.ext; intro f
      apply Subtype.ext; funext x
      obtain ⟨h1, h2⟩ := T.split x γ
      show A.ρ (T.hp x) (indRep T A γ f (T.idx x)) = A.ρ (T.hp (x * γ)) (f (T.idx (x * γ)))
      rw [indRep_apply, h1, h2, map_mul, Module.End.mul_apply]))

/-- The explicit fibre in coordinates: `indN_iso_coind` sends `f` to `x ↦ ρ(h(x)) f_{j(x)}`. -/
lemma indN_iso_coind_apply (A : Rep k H) (f : Fin T.m → A) (x : G) :
    ((indN_iso_coind T A).hom.hom f).1 x = A.ρ (T.hp x) (f (T.idx x)) := rfl

end Ind

/-- **[C] `prop:lhcompliant`(4) / `def:levelellmodel` in [C]'s fibre**: for `6 ∈ k^×`, any subgroup `H ≤ SL(2, ℤ)`
with a table `T` (e.g. `Γ₀(N)`, `RTrans.ofFiniteIndex`), the model's cohomology with coefficients in the explicit
induced fibre `Ind_H^Γ A` (`eq:MH`) is `H^n(H; A)`. -/
def levelN_shapiro_ind {k : Type} [CommRing k] (h6 : IsUnit (6 : k)) {H : Subgroup SL(2, ℤ)} (T : RTrans H)
    (A : Rep k H) (n : ℕ) :
    (relCxA model7 (IndN T A)).H (n : ℤ) ≃ₗ[k] groupCohomology A n :=
  (model7_rmkOrbifold h6 _ n).trans
    (((groupCohomology.functor k SL(2, ℤ) n).mapIso (indN_iso_coind T A)).toLinearEquiv.trans
      (groupCohomology.coindIso A n).toLinearEquiv)

/-- `Γ₀(N)` has a table of representatives (`N ≥ 1`). -/
def rtGamma0 (N : ℕ) [NeZero N] : RTrans (CongruenceSubgroup.Gamma0 N) := RTrans.ofFiniteIndex _

/-! ## M4-h: the level-`N` correspondences (`prop:levelcorrespondence`) -/

section LevelCorr

variable {G : Type} [Group G] {Γ : Subgroup G}

/-- The `Γ`-chain complex `Y` restricted to a subgroup `A ≤ Γ`. -/
abbrev _root_.GL2.Cone.GChainCx.resTo (Y : GChainCx.{0} Γ) (A : Subgroup G) (hA : A ≤ Γ) : GChainCx.{0} A where
  X := Y.X
  grp := Y.grp
  act q := DistribMulAction.compHom (Y.X q) (Subgroup.inclusion hA)
  bd := Y.bd
  bd_equiv q a x := Y.bd_equiv q (Subgroup.inclusion hA a) x
  bd_bd := Y.bd_bd
  aug := Y.aug
  aug_bd := Y.aug_bd
  aug_equiv a x := Y.aug_equiv (Subgroup.inclusion hA a) x

/-- `B'' = B ∩ g A g⁻¹ = {x ∈ B : g⁻¹ x g ∈ A}` ([C] `prop:levelcorrespondence`). -/
def Bpp (A B : Subgroup G) (g : G) : Subgroup G where
  carrier := {x | x ∈ B ∧ g⁻¹ * x * g ∈ A}
  mul_mem' := by
    rintro x y ⟨hx, hx'⟩ ⟨hy, hy'⟩
    refine ⟨B.mul_mem hx hy, ?_⟩
    have := A.mul_mem hx' hy'
    convert this using 1; group
  one_mem' := ⟨B.one_mem, by simp [A.one_mem]⟩
  inv_mem' := by
    rintro x ⟨hx, hx'⟩
    refine ⟨B.inv_mem hx, ?_⟩
    have := A.inv_mem hx'
    convert this using 1; group

lemma mem_Bpp {A B : Subgroup G} {g x : G} : x ∈ Bpp A B g ↔ x ∈ B ∧ g⁻¹ * x * g ∈ A := Iff.rfl

lemma Bpp_le (A B : Subgroup G) (g : G) : Bpp A B g ≤ B := fun _ h => h.1

/-- `A'' = g⁻¹ B'' g = A ∩ g⁻¹ B g`. -/
abbrev App (A B : Subgroup G) (g : G) : Subgroup G := Bpp B A g⁻¹

lemma mem_App {A B : Subgroup G} {g x : G} : x ∈ App A B g ↔ x ∈ A ∧ g * x * g⁻¹ ∈ B := by
  simp [App, mem_Bpp]

lemma conj_mem_App {A B : Subgroup G} {g h : G} (hh : h ∈ Bpp A B g) : g⁻¹ * h * g ∈ App A B g :=
  mem_App.mpr ⟨hh.2, by simpa [mul_assoc] using hh.1⟩

/-- A table of representatives `r_t` of `H \ B` (`H ≤ B`): every `b ∈ B` is `h r_t`, `h ∈ H`, for exactly one `t`. -/
structure RelTrans (H B : Subgroup G) where
  m : ℕ
  r : Fin m → B
  cosets : ∀ b : B, ∃! t : Fin m, ∃ h : B, (h : G) ∈ H ∧ (b : G) = h * r t

variable {Δ : Submonoid G} (hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ) {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V]
  (ρ : Δ →* Module.End R V) (Y : GChainCx.{0} Γ)

lemma memΔ (hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ) {A : Subgroup G} (hA : A ≤ Γ) {x : G} (hx : x ∈ A) : x ∈ Δ := hΓΔ ⟨x, hA hx⟩

/-- `Hom_A(C_•(Y), V)`: the `A`-equivariant cochains, `A ≤ Γ` (under the cochain-level Frobenius reciprocity
`Hom_Γ(C_•, Ind_A^Γ V) ≅ Hom_A(C_•, V)` of [C] `prop:levelcorrespondence`(3), the cochains of the model `L_A`). -/
abbrev CochA (A : Subgroup G) (hA : A ≤ Γ) : CochainCx.{0} R :=
  eqCochains A Δ (fun a => memΔ hΓΔ hA a.2) ρ (Y.resTo A hA)

variable {hΓΔ ρ Y}

variable (hΓΔ ρ) in
lemma mem_cochA {A : Subgroup G} (hA : A ≤ Γ) {q : ℤ} (φ : Y.X q →+ V) :
    φ ∈ eqHom A Δ (fun a => memΔ hΓΔ hA a.2) ρ (Y.resTo A hA) q ↔
      ∀ (a : G) (ha : a ∈ A) (y : Y.X q), φ ((⟨a, hA ha⟩ : Γ) • y) = ρ ⟨a, memΔ hΓΔ hA ha⟩ (φ y) :=
  ⟨fun h a ha y => h ⟨a, ha⟩ y, fun h a y => h a a.2 y⟩

variable (hΓΔ ρ Y)

/-- **`res`** ([C] `prop:levelcorrespondence`(4)(b), `B ⊆ A`, `g = 1`): an `A`-equivariant cochain is `A'`-equivariant. -/
def resCx {A A' : Subgroup G} (hA : A ≤ Γ) (hA' : A' ≤ A) :
    GMap (CochA hΓΔ ρ Y A hA) (CochA hΓΔ ρ Y A' (hA'.trans hA)) := fun q =>
  { toFun := fun φ => ⟨φ.1, (mem_cochA hΓΔ ρ (hA'.trans hA) φ.1).2 fun a ha y => (mem_cochA hΓΔ ρ hA φ.1).1 φ.2 a (hA' ha) y⟩
    map_add' := fun _ _ => by apply Subtype.ext; rfl
    map_smul' := fun _ _ => by apply Subtype.ext; rfl }

variable {g : G}

/-- **`(conj_g)_*`** with the chain datum `τ`: `ψ ↦ ρ(g) ψ ∘ τ`, from `A'' = g⁻¹ B'' g` to `B''`. -/
def conjCx {A B : Subgroup G} (hA : A ≤ Γ) (hB : B ≤ Γ) (hg : g ∈ Δ) (τ : TwistedChainDatum Γ g Y) :
    GMap (CochA hΓΔ ρ Y (App A B g) (fun _ h => hA ((mem_App.mp h).1)))
      (CochA hΓΔ ρ Y (Bpp A B g) ((Bpp_le A B g).trans hB)) := fun q =>
  { toFun := fun φ => ⟨(ρ ⟨g, hg⟩).toAddMonoidHom.comp (φ.1.comp (τ.τ q)),
      (mem_cochA hΓΔ ρ ((Bpp_le A B g).trans hB) _).2 fun h hh y => by
        have hc := conj_mem_App hh
        have hcΓ : g⁻¹ * h * g ∈ Γ := hA (mem_App.mp hc).1
        show ρ ⟨g, hg⟩ (φ.1 (τ.τ q ((⟨h, _⟩ : Γ) • y))) = ρ ⟨h, _⟩ (ρ ⟨g, hg⟩ (φ.1 (τ.τ q y)))
        rw [τ.t1 q ⟨h, hB hh.1⟩ ⟨g⁻¹ * h * g, hcΓ⟩ rfl y]
        rw [(mem_cochA hΓΔ ρ (fun _ h => hA ((mem_App.mp h).1)) φ.1).1 φ.2 _ hc (τ.τ q y)]
        rw [← Module.End.mul_apply, ← map_mul, ← Module.End.mul_apply, ← map_mul]
        congr 2
        apply Subtype.ext
        simp only [Submonoid.coe_mul]
        group⟩
    map_add' := fun φ ψ => by
      apply Subtype.ext; ext z
      show ρ ⟨g, hg⟩ ((φ.1 + ψ.1) (τ.τ q z)) = ρ ⟨g, hg⟩ (φ.1 (τ.τ q z)) + ρ ⟨g, hg⟩ (ψ.1 (τ.τ q z))
      rw [AddMonoidHom.add_apply, map_add]
    map_smul' := fun c φ => by
      apply Subtype.ext; ext z
      show ρ ⟨g, hg⟩ ((c • φ.1) (τ.τ q z)) = c • ρ ⟨g, hg⟩ (φ.1 (τ.τ q z))
      rw [AddMonoidHom.smul_apply, map_smul] }

/-- The corestriction formula `ψ ↦ (y ↦ ∑_t ρ(r_t⁻¹) ψ(r_t y))` on additive maps. -/
def corFun {H B : Subgroup G} (hB : B ≤ Γ) (tr : RelTrans H B) {q : ℤ} (φ : Y.X q →+ V) : Y.X q →+ V :=
  ∑ t, (ρ ⟨((tr.r t : B) : G)⁻¹, memΔ hΓΔ hB (B.inv_mem (tr.r t).2)⟩).toAddMonoidHom.comp
    (φ.comp (DistribSMul.toAddMonoidHom (Y.X q) (⟨tr.r t, hB (tr.r t).2⟩ : Γ)))

lemma corFun_apply {H B : Subgroup G} (hB : B ≤ Γ) (tr : RelTrans H B) {q : ℤ} (φ : Y.X q →+ V) (y : Y.X q) :
    corFun hΓΔ ρ Y hB tr φ y = ∑ t, ρ ⟨((tr.r t : B) : G)⁻¹, memΔ hΓΔ hB (B.inv_mem (tr.r t).2)⟩
      (φ ((⟨tr.r t, hB (tr.r t).2⟩ : Γ) • y)) := by
  simp [corFun, AddMonoidHom.finsetSum_apply]

/-- `cor` is `B`-equivariant ([C] `prop:levelcorrespondence`(3), the computation of `thm:welldef` at `g = 1`). -/
theorem corFun_mem {H B : Subgroup G} (hHB : H ≤ B) (hB : B ≤ Γ) (tr : RelTrans H B) {q : ℤ} (φ : Y.X q →+ V)
    (hφ : φ ∈ eqHom H Δ (fun a => memΔ hΓΔ (hHB.trans hB) a.2) ρ (Y.resTo H (hHB.trans hB)) q) :
    corFun hΓΔ ρ Y hB tr φ ∈ eqHom B Δ (fun a => memΔ hΓΔ hB a.2) ρ (Y.resTo B hB) q := by
  classical
  refine (mem_cochA hΓΔ ρ hB _).2 ?_
  intro b hb y
  have hc : ∀ t, ∃ t', ∃ h : B, (h : G) ∈ H ∧ ((tr.r t : G) * b) = h * tr.r t' := fun t =>
    (tr.cosets (tr.r t * ⟨b, hb⟩)).exists
  choose f hf using hc
  choose hh hhmem hheq using hf
  have finj : Function.Injective f := by
    intro i i' e
    have h1 := hheq i
    have h2 := hheq i'
    rw [← e] at h2
    have hsi : ((tr.r i : B) : G) = ((hh i * (hh i')⁻¹ : B) : G) * tr.r i' := by
      simp only [Subgroup.coe_mul, Subgroup.coe_inv]
      have : ((tr.r i : B) : G) = hh i * tr.r (f i) * b⁻¹ := by rw [← h1]; group
      rw [this]
      have : ((tr.r i' : B) : G) = hh i' * tr.r (f i) * b⁻¹ := by rw [← h2]; group
      rw [this]; group
    have hmem : ((hh i * (hh i')⁻¹ : B) : G) ∈ H := by
      simp only [Subgroup.coe_mul, Subgroup.coe_inv]
      exact H.mul_mem (hhmem i) (H.inv_mem (hhmem i'))
    exact (tr.cosets (tr.r i)).unique ⟨1, by simp [H.one_mem], by simp⟩ ⟨_, hmem, hsi⟩
  let ef := Equiv.ofBijective f (Finite.injective_iff_bijective.mp finj)
  rw [corFun_apply, corFun_apply, map_sum]
  conv_rhs => rw [← Equiv.sum_comp ef]
  refine Finset.sum_congr rfl fun i _ => ?_
  have e2 : (ef i : Fin tr.m) = f i := rfl
  have e1 : (⟨tr.r i, hB (tr.r i).2⟩ : Γ) • (⟨b, hB hb⟩ : Γ) • y =
      (⟨hh i, hB (hh i).2⟩ : Γ) • (⟨tr.r (f i), hB (tr.r (f i)).2⟩ : Γ) • y := by
    rw [← mul_smul, ← mul_smul]
    congr 1
    exact Subtype.ext (hheq i)
  rw [e2, e1, (mem_cochA hΓΔ ρ (hHB.trans hB) φ).1 hφ _ (hhmem i)]
  rw [← Module.End.mul_apply, ← map_mul, ← Module.End.mul_apply, ← map_mul]
  congr 2
  apply Subtype.ext
  simp only [Submonoid.coe_mul]
  have h1 := hheq i
  have : (hh i : G) = tr.r i * b * ((tr.r (f i) : B) : G)⁻¹ := by rw [h1]; group
  rw [this]; group

/-- **`cor`** ([C] `prop:levelcorrespondence`(4)(b), `A ⊆ B`, `g = 1`: the trace `ω_{A/B}` under reciprocity). -/
def corCx {H B : Subgroup G} (hHB : H ≤ B) (hB : B ≤ Γ) (tr : RelTrans H B) :
    GMap (CochA hΓΔ ρ Y H (hHB.trans hB)) (CochA hΓΔ ρ Y B hB) := fun q =>
  { toFun := fun φ => ⟨corFun hΓΔ ρ Y hB tr φ.1, corFun_mem hΓΔ ρ Y hHB hB tr φ.1 φ.2⟩
    map_add' := fun φ ψ => by
      apply Subtype.ext; ext y
      show corFun hΓΔ ρ Y hB tr (φ.1 + ψ.1) y = corFun hΓΔ ρ Y hB tr φ.1 y + corFun hΓΔ ρ Y hB tr ψ.1 y
      simp only [corFun_apply, AddMonoidHom.add_apply, map_add, Finset.sum_add_distrib]
    map_smul' := fun c φ => by
      apply Subtype.ext; ext y
      show corFun hΓΔ ρ Y hB tr (c • φ.1) y = c • corFun hΓΔ ρ Y hB tr φ.1 y
      simp only [corFun_apply, AddMonoidHom.smul_apply, map_smul, Finset.smul_sum] }

/-- **[C] `eq:levelcorr`, the function form of `prop:levelcorrespondence`(3)**:
`(T ψ)(y) = ∑_t ρ(r_t⁻¹ g) ψ(τ(r_t y))`, `r_t` representatives of `B'' \ B`, `τ` the level-`1` `g`-twisted chain datum. -/
def levelFun {A B : Subgroup G} (hB : B ≤ Γ) (hg : g ∈ Δ) (cd : RelTrans (Bpp A B g) B)
    (τ : TwistedChainDatum Γ g Y) {q : ℤ} (φ : Y.X q →+ V) : Y.X q →+ V :=
  ∑ t, (ρ ⟨((cd.r t : B) : G)⁻¹ * g, Δ.mul_mem (memΔ hΓΔ hB (B.inv_mem (cd.r t).2)) hg⟩).toAddMonoidHom.comp
    (φ.comp ((τ.τ q).comp (DistribSMul.toAddMonoidHom (Y.X q) (⟨cd.r t, hB (cd.r t).2⟩ : Γ))))

lemma levelFun_apply {A B : Subgroup G} (hB : B ≤ Γ) (hg : g ∈ Δ) (cd : RelTrans (Bpp A B g) B)
    (τ : TwistedChainDatum Γ g Y) {q : ℤ} (φ : Y.X q →+ V) (y : Y.X q) :
    levelFun hΓΔ ρ Y hB hg cd τ φ y = ∑ t, ρ ⟨((cd.r t : B) : G)⁻¹ * g, Δ.mul_mem (memΔ hΓΔ hB (B.inv_mem (cd.r t).2)) hg⟩
      (φ (τ.τ q ((⟨cd.r t, hB (cd.r t).2⟩ : Γ) • y))) := by
  simp [levelFun, AddMonoidHom.finsetSum_apply]

/-- The formula is `cor ∘ (conj_g)_* ∘ res` on the underlying maps. -/
lemma levelFun_eq {A B : Subgroup G} (hA : A ≤ Γ) (hB : B ≤ Γ) (hg : g ∈ Δ) (cd : RelTrans (Bpp A B g) B)
    (τ : TwistedChainDatum Γ g Y) {q : ℤ} (φ : (CochA hΓΔ ρ Y A hA).X q) :
    levelFun hΓΔ ρ Y hB hg cd τ φ.1 =
      (corCx hΓΔ ρ Y (Bpp_le A B g) hB cd q (conjCx hΓΔ ρ Y hA hB hg τ q
        (resCx hΓΔ ρ Y hA (fun _ h => (mem_App.mp h).1) q φ))).1 := by
  ext y
  rw [levelFun_apply]
  show _ = corFun hΓΔ ρ Y hB cd (conjCx hΓΔ ρ Y hA hB hg τ q
    (resCx hΓΔ ρ Y hA (fun _ h => (mem_App.mp h).1) q φ)).1 y
  rw [corFun_apply]
  refine Finset.sum_congr rfl fun t _ => ?_
  show _ = ρ _ (ρ ⟨g, hg⟩ (φ.1 (τ.τ q _)))
  rw [← Module.End.mul_apply, ← map_mul]
  rfl

/-- **The level-`N` correspondence `[A g B]`** on `Hom_A(C_•, V) → Hom_B(C_•, V)` ([C] `eq:levelcorr` under the
reciprocity, `prop:levelcorrespondence`(3)). -/
def levelCorr {A B : Subgroup G} (hA : A ≤ Γ) (hB : B ≤ Γ) (hg : g ∈ Δ) (cd : RelTrans (Bpp A B g) B)
    (τ : TwistedChainDatum Γ g Y) : GMap (CochA hΓΔ ρ Y A hA) (CochA hΓΔ ρ Y B hB) := fun q =>
  { toFun := fun φ => ⟨levelFun hΓΔ ρ Y hB hg cd τ φ.1, by rw [levelFun_eq hΓΔ ρ Y hA]; exact Subtype.prop _⟩
    map_add' := fun φ ψ => by
      apply Subtype.ext; ext y
      show levelFun hΓΔ ρ Y hB hg cd τ (φ.1 + ψ.1) y =
        levelFun hΓΔ ρ Y hB hg cd τ φ.1 y + levelFun hΓΔ ρ Y hB hg cd τ ψ.1 y
      simp only [levelFun_apply, AddMonoidHom.add_apply, map_add, Finset.sum_add_distrib]
    map_smul' := fun c φ => by
      apply Subtype.ext; ext y
      show levelFun hΓΔ ρ Y hB hg cd τ (c • φ.1) y = c • levelFun hΓΔ ρ Y hB hg cd τ φ.1 y
      simp only [levelFun_apply, AddMonoidHom.smul_apply, map_smul, Finset.smul_sum] }

lemma levelCorr_apply {A B : Subgroup G} (hA : A ≤ Γ) (hB : B ≤ Γ) (hg : g ∈ Δ) (cd : RelTrans (Bpp A B g) B)
    (τ : TwistedChainDatum Γ g Y) (q : ℤ) (φ : (CochA hΓΔ ρ Y A hA).X q) (y : Y.X q) :
    (levelCorr hΓΔ ρ Y hA hB hg cd τ q φ).1 y = ∑ t, ρ ⟨((cd.r t : B) : G)⁻¹ * g,
      Δ.mul_mem (memΔ hΓΔ hB (B.inv_mem (cd.r t).2)) hg⟩ (φ.1 (τ.τ q ((⟨cd.r t, hB (cd.r t).2⟩ : Γ) • y))) :=
  levelFun_apply hΓΔ ρ Y hB hg cd τ φ.1 y

/-- **`levelCorr_eq_res_cor`** ([C] `prop:levelcorrespondence`(3),(4)(b),(c), cochain level, every `R`, no hypothesis on
`ℓ`): `[A g B] = cor_{B''}^{B} ∘ (conj_g)_* ∘ res^{A}_{A''}`. -/
theorem levelCorr_eq_res_cor {A B : Subgroup G} (hA : A ≤ Γ) (hB : B ≤ Γ) (hg : g ∈ Δ) (cd : RelTrans (Bpp A B g) B)
    (τ : TwistedChainDatum Γ g Y) (q : ℤ) :
    levelCorr hΓΔ ρ Y hA hB hg cd τ q = (corCx hΓΔ ρ Y (Bpp_le A B g) hB cd q).comp
      ((conjCx hΓΔ ρ Y hA hB hg τ q).comp (resCx hΓΔ ρ Y hA (fun _ h => (mem_App.mp h).1) q)) := by
  apply LinearMap.ext; intro φ
  exact Subtype.ext (levelFun_eq hΓΔ ρ Y hA hB hg cd τ φ)

lemma resCx_isChainMap {A A' : Subgroup G} (hA : A ≤ Γ) (hA' : A' ≤ A) : IsChainMap (resCx hΓΔ ρ Y hA hA') :=
  fun _ => rfl

lemma conjCx_isChainMap {A B : Subgroup G} (hA : A ≤ Γ) (hB : B ≤ Γ) (hg : g ∈ Δ) (τ : TwistedChainDatum Γ g Y) :
    IsChainMap (conjCx hΓΔ ρ Y hA hB hg τ) := by
  intro q
  apply LinearMap.ext; intro φ; apply Subtype.ext; ext y
  show ρ ⟨g, hg⟩ (φ.1 (τ.τ q (Y.bd q y))) = ρ ⟨g, hg⟩ (φ.1 (Y.bd q (τ.τ (q + 1) y)))
  rw [τ.t2]

lemma corCx_isChainMap {H B : Subgroup G} (hHB : H ≤ B) (hB : B ≤ Γ) (tr : RelTrans H B) :
    IsChainMap (corCx hΓΔ ρ Y hHB hB tr) := by
  intro q
  apply LinearMap.ext; intro φ; apply Subtype.ext; ext y
  rw [LinearMap.comp_apply, LinearMap.comp_apply]
  show corFun hΓΔ ρ Y hB tr φ.1 (Y.bd q y) = corFun hΓΔ ρ Y hB tr (φ.1.comp (Y.bd q)) y
  rw [corFun_apply, corFun_apply]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [AddMonoidHom.comp_apply, Y.bd_equiv]

/-- **`levelCorr_isChainMap`** ([C] `prop:levelcorrespondence`(2)): a cochain map over every `R`, no hypothesis on `ℓ`. -/
theorem levelCorr_isChainMap {A B : Subgroup G} (hA : A ≤ Γ) (hB : B ≤ Γ) (hg : g ∈ Δ) (cd : RelTrans (Bpp A B g) B)
    (τ : TwistedChainDatum Γ g Y) : IsChainMap (levelCorr hΓΔ ρ Y hA hB hg cd τ) := by
  have h := isChainMap_comp (corCx_isChainMap (hΓΔ := hΓΔ) (ρ := ρ) (Y := Y) (Bpp_le A B g) hB cd)
    (isChainMap_comp (conjCx_isChainMap (hΓΔ := hΓΔ) (ρ := ρ) (Y := Y) hA hB hg τ)
      (resCx_isChainMap (hΓΔ := hΓΔ) (ρ := ρ) (Y := Y) hA (fun _ h => (mem_App.mp h).1)))
  intro q
  rw [levelCorr_eq_res_cor, levelCorr_eq_res_cor]
  exact h q

/-- `levelCorr_eq_res_cor` on cohomology: `[A g B] = cor ∘ (conj_g)_* ∘ res` on `H^q`. -/
theorem levelCorr_H_eq_res_cor {A B : Subgroup G} (hA : A ≤ Γ) (hB : B ≤ Γ) (hg : g ∈ Δ)
    (cd : RelTrans (Bpp A B g) B) (τ : TwistedChainDatum Γ g Y) (q : ℤ) :
    HmapOf _ (levelCorr_isChainMap hΓΔ ρ Y hA hB hg cd τ) q =
      (HmapOf _ (corCx_isChainMap (hΓΔ := hΓΔ) (ρ := ρ) (Y := Y) (Bpp_le A B g) hB cd) q).comp
        ((HmapOf _ (conjCx_isChainMap (hΓΔ := hΓΔ) (ρ := ρ) (Y := Y) hA hB hg τ) q).comp
          (HmapOf _ (resCx_isChainMap (hΓΔ := hΓΔ) (ρ := ρ) (Y := Y) hA (fun _ h => (mem_App.mp h).1)) q)) := by
  rw [← HmapOf_comp _ _ _ _ (isChainMap_comp (conjCx_isChainMap hΓΔ ρ Y hA hB hg τ) (resCx_isChainMap hΓΔ ρ Y hA _)),
    ← HmapOf_comp _ _ _ _ (isChainMap_comp (corCx_isChainMap hΓΔ ρ Y _ hB cd)
      (isChainMap_comp (conjCx_isChainMap hΓΔ ρ Y hA hB hg τ) (resCx_isChainMap hΓΔ ρ Y hA _)))]
  exact HmapOf_congr (funext fun q => levelCorr_eq_res_cor hΓΔ ρ Y hA hB hg cd τ q) _ _ q

end LevelCorr

section LevelCorrMore

variable {G : Type} [Group G] {Γ : Subgroup G} {g : G}

/-- A `g`-twisted chain datum for `Γ` is one for any `Γ' ≤ Γ` (on the restricted complex). -/
def _root_.GL2.Cone.TwistedChainDatum.resTo {Y : GChainCx.{0} Γ} (τ : TwistedChainDatum Γ g Y) (Γ' : Subgroup G)
    (h : Γ' ≤ Γ) : TwistedChainDatum Γ' g (Y.resTo Γ' h) where
  τ := τ.τ
  t1 q h₁ h₁' hh x := τ.t1 q (Subgroup.inclusion h h₁) (Subgroup.inclusion h h₁') hh x
  t2 := τ.t2
  t3 := τ.t3

/-- For a central `g` (e.g. `p I`), `τ = id` is a `g`-twisted chain datum (one coset, [C] `prop:diamond`). -/
def _root_.GL2.Cone.TwistedChainDatum.ofCentral (Y : GChainCx.{0} Γ) (hc : ∀ x : G, g * x = x * g) :
    TwistedChainDatum Γ g Y where
  τ _ := AddMonoidHom.id _
  t1 q h h' hh x := by
    have e : h = h' := Subtype.ext (by rw [hh, mul_assoc, ← hc, ← mul_assoc, inv_mul_cancel, one_mul])
    subst e; rfl
  t2 _ _ := rfl
  t3 _ := rfl

variable {Δ : Submonoid G} (hΓΔ : ∀ γ : Γ, (γ : G) ∈ Δ) {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V]
  (ρ : Δ →* Module.End R V) (Y : GChainCx.{0} Γ)

/-- **A central `g` acts by its scalar** ([C] `prop:diamond`, `prop:nebcorr`(3) for `T(p,p)`; every `R`): if `g` is
central and `ρ(g) = c`, then `[A g A]` with `τ = id` is `c` on the cochains, whatever the table. -/
theorem levelCorr_central {A : Subgroup G} (hA : A ≤ Γ) (hg : g ∈ Δ) (hc : ∀ x : G, g * x = x * g) (c : R)
    (hρ : ρ ⟨g, hg⟩ = c • 1) (cd : RelTrans (Bpp A A g) A) (q : ℤ) (φ : (CochA hΓΔ ρ Y A hA).X q) :
    levelCorr hΓΔ ρ Y hA hA hg cd (TwistedChainDatum.ofCentral Y hc) q φ = c • φ := by
  have hgx : ∀ x : G, g⁻¹ * x * g = x := fun x => by rw [mul_assoc, ← hc, ← mul_assoc, inv_mul_cancel, one_mul]
  obtain ⟨t₀, -, ht₀⟩ := cd.cosets 1
  have huniq : ∀ t, t = t₀ := fun t =>
    ht₀ t ⟨(cd.r t)⁻¹, by
      refine ⟨A.inv_mem (cd.r t).2, ?_⟩
      simp only [Subgroup.coe_inv]
      rw [hgx]; exact A.inv_mem (cd.r t).2, by simp⟩
  apply Subtype.ext; ext y
  rw [levelCorr_apply, Finset.sum_eq_single_of_mem t₀ (Finset.mem_univ _)
    (fun t _ ht => absurd (huniq t) ht)]
  show ρ _ (φ.1 ((⟨cd.r t₀, _⟩ : Γ) • y)) = c • φ.1 y
  rw [(mem_cochA hΓΔ ρ hA φ.1).1 φ.2 _ (cd.r t₀).2, ← Module.End.mul_apply, ← map_mul]
  have e : (⟨((cd.r t₀ : A) : G)⁻¹ * g, Δ.mul_mem (memΔ hΓΔ hA (A.inv_mem (cd.r t₀).2)) hg⟩ : Δ) *
      ⟨((cd.r t₀ : A) : G), memΔ hΓΔ hA (cd.r t₀).2⟩ = ⟨g, hg⟩ := by
    apply Subtype.ext
    simp only [Submonoid.coe_mul]
    rw [mul_assoc, hc, ← mul_assoc, inv_mul_cancel, one_mul]
  rw [e, hρ]
  rfl

end LevelCorrMore

/-! ## M4-h: the nebentypus (`sec:nebentypus`) -/

section Neb

variable (N : ℕ)

/-- **[C] `def:nebdata`(2)**: `Δ₀(N)`, the integral matrices with `N ∣ c` and `gcd(a, N) = 1`, in `GL(2, ℚ)`.
(Restatement: no `det > 0`, as for `Δ = ΔQ`, which has none either; nothing below uses the sign of the determinant.) -/
def Δ0Q : Submonoid GLQ where
  carrier := {A | ∃ M : Mat2 ℤ, (A : Matrix (Fin 2) (Fin 2) ℚ) = κ M ∧ (N : ℤ) ∣ M 1 0 ∧ IsCoprime (M 0 0) (N : ℤ)}
  mul_mem' := by
    rintro A B ⟨M, hM, hc, ha⟩ ⟨M', hM', hc', ha'⟩
    refine ⟨M * M', by rw [Units.val_mul, hM, hM', map_mul], ?_, ?_⟩
    · rw [Matrix.mul_apply, Fin.sum_univ_two]
      exact dvd_add (dvd_mul_of_dvd_left hc _) (dvd_mul_of_dvd_right hc' _)
    · rw [Matrix.mul_apply, Fin.sum_univ_two]
      obtain ⟨k, hk⟩ := hc'
      rw [hk]
      have := (ha.mul_left ha').add_mul_left_left (M 0 1 * k)
      rwa [show M 0 0 * M' 0 0 + (N : ℤ) * (M 0 1 * k) = M 0 0 * M' 0 0 + M 0 1 * ((N : ℤ) * k) by ring] at this
  one_mem' := ⟨1, by simp, by simp, by simpa using isCoprime_one_left⟩

variable {N}

lemma mem_Δ0Q {A : GLQ} :
    A ∈ Δ0Q N ↔ ∃ M : Mat2 ℤ, (A : Matrix (Fin 2) (Fin 2) ℚ) = κ M ∧ (N : ℤ) ∣ M 1 0 ∧ IsCoprime (M 0 0) (N : ℤ) :=
  Iff.rfl

variable (N)

lemma Δ0Q_le_ΔQ : Δ0Q N ≤ ΔQ := fun _ ⟨M, hM, _⟩ => ⟨M, hM⟩

/-- `a_δ ∈ ℤ`, the upper-left entry. -/
def aZ (δ : Δ0Q N) : ℤ := (Classical.choose (mem_Δ0Q.mp δ.2)) 0 0

variable {N}

lemma aZ_eq (δ : Δ0Q N) (M : Mat2 ℤ) (hM : ((δ : GLQ) : Matrix (Fin 2) (Fin 2) ℚ) = κ M) : aZ N δ = M 0 0 := by
  have e : Classical.choose (mem_Δ0Q.mp δ.2) = M :=
    κ_injective ((Classical.choose_spec (mem_Δ0Q.mp δ.2)).1.symm.trans hM)
  simp [aZ, e]

variable (N)

/-- **[C] `lem:nebdata`(2)**: `δ ↦ a_δ mod N` is a homomorphism of monoids `Δ₀(N) → ℤ/N`. -/
def aMod : Δ0Q N →* ZMod N where
  toFun δ := (aZ N δ : ZMod N)
  map_one' := by
    rw [aZ_eq 1 1 (by simp)]; simp
  map_mul' δ δ' := by
    obtain ⟨M, hM, -, -⟩ := mem_Δ0Q.mp δ.2
    obtain ⟨M', hM', hc', -⟩ := mem_Δ0Q.mp δ'.2
    rw [aZ_eq δ M hM, aZ_eq δ' M' hM', aZ_eq (δ * δ') (M * M') (by
      rw [Submonoid.coe_mul, Units.val_mul, hM, hM', map_mul])]
    rw [Matrix.mul_apply, Fin.sum_univ_two, Int.cast_add, Int.cast_mul, Int.cast_mul,
      (ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mpr hc', mul_zero, add_zero]

variable {N} {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)
  (χ : DirichletCharacter R N)

/-- **[C] `def:nebdata`(5)**: `ρ_{n,χ}(δ) = χ(a_δ) ρ(δ)` on `Δ₀(N)`, multiplicative (`lem:nebdata`(2)). -/
def ρχ : Δ0Q N →* Module.End R V where
  toFun δ := χ (aMod N δ) • ρ ⟨δ, Δ0Q_le_ΔQ N δ.2⟩
  map_one' := by
    rw [map_one, MulChar.map_one, one_smul]
    exact map_one ρ
  map_mul' δ δ' := by
    rw [map_mul, map_mul χ, smul_mul_smul_comm]
    congr 1
    rw [← map_mul]
    rfl

lemma ρχ_apply (δ : Δ0Q N) : ρχ ρ χ δ = χ (aMod N δ) • ρ ⟨δ, Δ0Q_le_ΔQ N δ.2⟩ := rfl

/-- `Γ₀(N)` inside `GL(2, ℚ)`. -/
def Γ0Q (N : ℕ) : Subgroup GLQ := (CongruenceSubgroup.Gamma0 N).map (Matrix.SpecialLinearGroup.mapGL ℚ)

lemma Γ0Q_le_ΓQ (N : ℕ) : Γ0Q N ≤ ΓQ := Subgroup.map_le_range _ _

lemma Γ0Q_le_Δ0Q (N : ℕ) (γ : Γ0Q N) : (γ : GLQ) ∈ Δ0Q N := by
  obtain ⟨_, ⟨t, ht, rfl⟩⟩ := γ
  refine ⟨t, coe_mapGL t, ?_, ?_⟩
  · exact (ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mp (CongruenceSubgroup.Gamma0_mem.mp ht)
  · obtain ⟨k, hk⟩ := (ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mp (CongruenceSubgroup.Gamma0_mem.mp ht)
    have hdet : (t : Mat2 ℤ).det = 1 := t.2
    rw [Matrix.det_fin_two, hk] at hdet
    exact ⟨t 1 1, -(t 0 1 * k), by linear_combination hdet⟩

/-- **[C] `lem:nebdata`** (the parts the model uses): (2) `a` is multiplicative mod `N` and `ρ_{n,χ}` is a
representation of `Δ₀(N)` (`ρχ`, by construction); `Γ₀(N) ⊆ Δ₀(N)`; on `Γ₀(N)`, `a_γ d_γ ≡ 1 (N)`, so
`χ(a_γ) = χ(d_γ)⁻¹`; (4) where `a_δ ≡ 1 (N)` (e.g. on `Γ₁(N)`), `ρ_{n,χ}(δ) = ρ(δ)`. -/
theorem nebData :
    (∀ δ δ' : Δ0Q N, aMod N (δ * δ') = aMod N δ * aMod N δ') ∧
    (∀ γ : Γ0Q N, (γ : GLQ) ∈ Δ0Q N) ∧
    (∀ t : SL(2, ℤ), t ∈ CongruenceSubgroup.Gamma0 N →
      ((t 0 0 : ℤ) : ZMod N) * ((t 1 1 : ℤ) : ZMod N) = 1) ∧
    (∀ δ : Δ0Q N, aMod N δ = 1 → ρχ ρ χ δ = ρ ⟨δ, Δ0Q_le_ΔQ N δ.2⟩) := by
  refine ⟨fun δ δ' => map_mul _ δ δ', Γ0Q_le_Δ0Q N, fun t ht => ?_, fun δ h => ?_⟩
  · have hdet : (t : Mat2 ℤ).det = 1 := t.2
    rw [Matrix.det_fin_two] at hdet
    have h0 := CongruenceSubgroup.Gamma0_mem.mp ht
    have := congrArg (fun z : ℤ => (z : ZMod N)) hdet
    simp only [Int.cast_sub, Int.cast_mul, Int.cast_one, h0, mul_zero, sub_zero] at this
    exact this
  · rw [ρχ_apply, h, MulChar.map_one, one_smul]

/-- The nebentypus correspondences live on `Γ = Γ₀(N)`; the level-`1` complex `Y` restricted to it. -/
abbrev YN (Y : GChainCx.{0} ΓQ) : GChainCx.{0} (Γ0Q N) := Y.resTo (Γ0Q N) (Γ0Q_le_ΓQ N)

/-- **[C] `eq:nebcorr`** (`prop:nebcorr`): `[A g B]_χ` for `A, B ≤ Γ₀(N)`, `g ∈ Δ₀(N)`, on cochains with values in
`V(χ)` (`ρχ`), with the **level-`1`** `g`-twisted chain datum `τ` (no new `τ`-table): `levelCorr` at `ρ_{n,χ}`. -/
def nebCorr (Y : GChainCx.{0} ΓQ) {A B : Subgroup GLQ} (hA : A ≤ Γ0Q N) (hB : B ≤ Γ0Q N) {g : GLQ} (hg : g ∈ Δ0Q N)
    (cd : RelTrans (Bpp A B g) B) (τ : TwistedChainDatum ΓQ g Y) :=
  levelCorr (Γ0Q_le_Δ0Q N) (ρχ ρ χ) (YN Y) hA hB hg cd (τ.resTo (Γ0Q N) (Γ0Q_le_ΓQ N))

/-- **[C] `prop:nebcorr`(2)**: a cochain map over every `R`. -/
theorem nebCorr_isChainMap (Y : GChainCx.{0} ΓQ) {A B : Subgroup GLQ} (hA : A ≤ Γ0Q N) (hB : B ≤ Γ0Q N) {g : GLQ}
    (hg : g ∈ Δ0Q N) (cd : RelTrans (Bpp A B g) B) (τ : TwistedChainDatum ΓQ g Y) :
    IsChainMap (nebCorr ρ χ Y hA hB hg cd τ) :=
  levelCorr_isChainMap _ _ _ hA hB hg cd _

/-- **[C] `eq:nebcorr`, `prop:nebcorr`(1)**: one scalar `χ(a_{δ_t})` per term, `δ_t = r_t⁻¹ g ∈ Δ₀(N)`. -/
theorem nebCorr_apply (Y : GChainCx.{0} ΓQ) {A B : Subgroup GLQ} (hA : A ≤ Γ0Q N) (hB : B ≤ Γ0Q N) {g : GLQ}
    (hg : g ∈ Δ0Q N) (cd : RelTrans (Bpp A B g) B) (τ : TwistedChainDatum ΓQ g Y) (q : ℤ)
    (φ : (CochA (Γ0Q_le_Δ0Q N) (ρχ ρ χ) (YN Y) A hA).X q) (y : Y.X q) :
    (nebCorr ρ χ Y hA hB hg cd τ q φ).1 y = ∑ t,
      χ (aMod N ⟨((cd.r t : B) : GLQ)⁻¹ * g, (Δ0Q N).mul_mem (Γ0Q_le_Δ0Q N ⟨_, hB (B.inv_mem (cd.r t).2)⟩) hg⟩) •
        ρ ⟨((cd.r t : B) : GLQ)⁻¹ * g, Δ0Q_le_ΔQ N
          ((Δ0Q N).mul_mem (Γ0Q_le_Δ0Q N ⟨_, hB (B.inv_mem (cd.r t).2)⟩) hg)⟩
          (φ.1 (τ.τ q ((⟨cd.r t, Γ0Q_le_ΓQ N (hB (cd.r t).2)⟩ : ΓQ) • y))) :=
  levelCorr_apply _ _ _ hA hB hg cd _ q φ y

/-- `prop:nebcorr`(2), the factorisation: `[A g B]_χ = cor ∘ (conj_g)_* ∘ res` (cochain level, every `R`). -/
theorem nebCorr_eq_res_cor (Y : GChainCx.{0} ΓQ) {A B : Subgroup GLQ} (hA : A ≤ Γ0Q N) (hB : B ≤ Γ0Q N) {g : GLQ}
    (hg : g ∈ Δ0Q N) (cd : RelTrans (Bpp A B g) B) (τ : TwistedChainDatum ΓQ g Y) (q : ℤ) :
    nebCorr ρ χ Y hA hB hg cd τ q =
      (corCx (Γ0Q_le_Δ0Q N) (ρχ ρ χ) (YN Y) (Bpp_le A B g) hB cd q).comp
        ((conjCx (Γ0Q_le_Δ0Q N) (ρχ ρ χ) (YN Y) hA hB hg (τ.resTo (Γ0Q N) (Γ0Q_le_ΓQ N)) q).comp
          (resCx (Γ0Q_le_Δ0Q N) (ρχ ρ χ) (YN Y) hA (fun _ h => (mem_App.mp h).1) q)) :=
  levelCorr_eq_res_cor _ _ _ hA hB hg cd _ q

/-- **[C] `prop:nebcorr`(4)**, the term-wise compatibility: where every `a_{δ_t} ≡ 1 (N)` (the `r_t` and `g` with
`a ≡ 1`, e.g. `r_t ∈ Γ₁(N)`, `g = diag(1, p)`), the character drops out and `[A g B]_χ` is the plain formula. -/
theorem nebCorr_apply_of_a_one (Y : GChainCx.{0} ΓQ) {A B : Subgroup GLQ} (hA : A ≤ Γ0Q N) (hB : B ≤ Γ0Q N)
    {g : GLQ} (hg : g ∈ Δ0Q N) (cd : RelTrans (Bpp A B g) B) (τ : TwistedChainDatum ΓQ g Y)
    (ha : ∀ t, aMod N ⟨((cd.r t : B) : GLQ)⁻¹ * g,
      (Δ0Q N).mul_mem (Γ0Q_le_Δ0Q N ⟨_, hB (B.inv_mem (cd.r t).2)⟩) hg⟩ = 1) (q : ℤ)
    (φ : (CochA (Γ0Q_le_Δ0Q N) (ρχ ρ χ) (YN Y) A hA).X q) (y : Y.X q) :
    (nebCorr ρ χ Y hA hB hg cd τ q φ).1 y = ∑ t,
        ρ ⟨((cd.r t : B) : GLQ)⁻¹ * g, Δ0Q_le_ΔQ N
          ((Δ0Q N).mul_mem (Γ0Q_le_Δ0Q N ⟨_, hB (B.inv_mem (cd.r t).2)⟩) hg)⟩
          (φ.1 (τ.τ q ((⟨cd.r t, Γ0Q_le_ΓQ N (hB (cd.r t).2)⟩ : ΓQ) • y))) := by
  rw [nebCorr_apply]
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [ha t, MulChar.map_one, one_smul]

/-- **[C] `prop:nebcorr`(3), `T(p,p)`**: a central `g ∈ Δ₀(N)` with `ρ(g) = c` (`g = p I`, `p ∤ N`, `c = p^n` for `Sym^n`)
acts by `χ(a_g) c` (`= χ(p) p^n`), one coset, `τ = id`, every `R`. -/
theorem nebTpp (Y : GChainCx.{0} ΓQ) {A : Subgroup GLQ} (hA : A ≤ Γ0Q N) {g : GLQ} (hg : g ∈ Δ0Q N)
    (hc : ∀ x : GLQ, g * x = x * g) (c : R) (hρ : ρ ⟨g, Δ0Q_le_ΔQ N hg⟩ = c • 1) (cd : RelTrans (Bpp A A g) A)
    (q : ℤ) (φ : (CochA (Γ0Q_le_Δ0Q N) (ρχ ρ χ) (YN Y) A hA).X q) :
    nebCorr ρ χ Y hA hA hg cd (TwistedChainDatum.ofCentral Y hc) q φ = (χ (aMod N ⟨g, hg⟩) * c) • φ :=
  levelCorr_central (Γ0Q_le_Δ0Q N) (ρχ ρ χ) (YN Y) hA hg hc _
    (by rw [ρχ_apply, hρ, smul_smul]) cd q φ

end Neb

/-! ### `thm:nebrelations`: the Hecke relations with a character -/

section NebRelations

/-- The full operator assembled from primitive ones, `F_ν = ∑_{0 ≤ 2j ≤ ν} c^j P_{ν - 2j}` (the shape of `Tfull`). -/
def fullOf {R A : Type*} [CommRing R] [Ring A] [Algebra R A] (P : ℕ → A) (c : R) (ν : ℕ) : A :=
  ∑ j ∈ Finset.range (ν / 2 + 1), c ^ j • P (ν - 2 * j)

lemma fullOf_add_two {R A : Type*} [CommRing R] [Ring A] [Algebra R A] (P : ℕ → A) (c : R) (ν : ℕ) :
    fullOf P c (ν + 2) = P (ν + 2) + c • fullOf P c ν := by
  unfold fullOf
  rw [show (ν + 2) / 2 + 1 = (ν / 2 + 1) + 1 by omega, Finset.sum_range_succ', Finset.smul_sum, add_comm]
  congr 1
  · simp
  · refine Finset.sum_congr rfl fun j _ => ?_
    rw [show ν + 2 - 2 * (j + 1) = ν - 2 * j by omega, pow_succ', mul_smul]

/-- **The primitive-to-full recursion** (algebra only): if the primitive operators satisfy the double-coset products
`P₁² = P₂ + (p + 1) c` and `P₁ P_{μ+1} = P_{μ+2} + p c P_μ` (`μ ≥ 1`), with `P₀ = 1`, then the full operators satisfy
`F_{ν+2} = P₁ F_{ν+1} − p c F_ν`.  (The `p^n I` lesson: the scalar term at `μ = 0` is `(p + 1) c`, not `p c`.) -/
theorem fullOf_rec {R A : Type*} [CommRing R] [Ring A] [Algebra R A] (P : ℕ → A) (hP0 : P 0 = 1) (p c : R)
    (h0 : P 1 * P 1 = P 2 + ((p + 1) * c) • (1 : A))
    (h1 : ∀ μ, 1 ≤ μ → P 1 * P (μ + 1) = P (μ + 2) + (p * c) • P μ) (ν : ℕ) :
    fullOf P c (ν + 2) = P 1 * fullOf P c (ν + 1) - (p * c) • fullOf P c ν := by
  have F0 : fullOf P c 0 = 1 := by simp [fullOf, hP0]
  have F1 : fullOf P c 1 = P 1 := by simp [fullOf]
  suffices H : ∀ ν, (fullOf P c (ν + 2) = P 1 * fullOf P c (ν + 1) - (p * c) • fullOf P c ν) ∧
      (fullOf P c (ν + 3) = P 1 * fullOf P c (ν + 2) - (p * c) • fullOf P c (ν + 1)) from (H ν).1
  intro ν
  induction ν with
  | zero =>
    refine ⟨?_, ?_⟩
    · rw [fullOf_add_two, F0, F1, h0]
      module
    · rw [show (0 + 3 : ℕ) = 1 + 2 by rfl, fullOf_add_two, show (0 + 2 : ℕ) = 0 + 2 by rfl, fullOf_add_two, F0, F1,
        mul_add, mul_smul_comm, mul_one, h1 1 le_rfl]
      module
  | succ k ih =>
    refine ⟨ih.2, ?_⟩
    have e1 := fullOf_add_two P c (k + 2)
    have e2 := fullOf_add_two P c (k + 1)
    have e3 := fullOf_add_two P c k
    have hX : P 1 * fullOf P c (k + 1) = P (k + 2) + c • fullOf P c k + (p * c) • fullOf P c k := by
      rw [← e3, ih.1]; abel
    rw [show k + 1 + 3 = k + 2 + 2 by omega, show k + 1 + 2 = k + 1 + 2 from rfl,
      show k + 1 + 1 = k + 2 by omega, e1, e2, e3, mul_add, mul_smul_comm,
      show P 1 * P (k + 1 + 2) = P (k + 2 + 2) + (p * c) • P (k + 2) from h1 (k + 2) (by omega), hX]
    module

variable {N : ℕ} {R : Type} [CommRing R] {V : Type} [AddCommGroup V] [Module R V] (ρ : ΔQ →* Module.End R V)
  (χ : DirichletCharacter R N) (Y : GChainCx.{0} ΓQ)

/-- `diag(1, p^μ) ∈ Δ₀(N)`. -/
lemma gP_mem_Δ0Q (p μ : ℕ) (hp : p.Prime) : gP p μ hp ∈ Δ0Q N :=
  ⟨gMat ((p : ℤ) ^ μ), coe_gQ _ _, by simp [gMat], by simpa [gMat] using isCoprime_one_left⟩

/-- `H^n` of the character model `Hom_{Γ₀(N)}(C_•, V(χ))`. -/
abbrev HN (n : ℕ) := (CochA (Γ0Q_le_Δ0Q N) (ρχ ρ χ) (YN Y) (Γ0Q N) le_rfl).H (n : ℤ)

/-- The operator `[Γ₀(N) g Γ₀(N)]_χ` on `H^n` (`T^{(N,χ)}_p` for `g = diag(1,p)`, `p ∤ N`; `U^{(N,χ)}_p` for `p ∣ N`). -/
def nebOp {g : GLQ} (hg : g ∈ Δ0Q N) (cd : RelTrans (Bpp (Γ0Q N) (Γ0Q N) g) (Γ0Q N)) (τ : TwistedChainDatum ΓQ g Y)
    (n : ℕ) : Module.End R (HN ρ χ Y n) :=
  HmapOf _ (nebCorr_isChainMap ρ χ Y le_rfl le_rfl hg cd τ) n

/-- The primitive operators `T_{Γ₀(N) diag(1,p^μ) Γ₀(N)}` with character (`1` at `μ = 0`). -/
def nebTprim (p : ℕ) (hp : p.Prime) (cd : ∀ μ, RelTrans (Bpp (Γ0Q N) (Γ0Q N) (gP p μ hp)) (Γ0Q N))
    (τ : ∀ μ, TwistedChainDatum ΓQ (gP p μ hp) Y) (n μ : ℕ) : Module.End R (HN ρ χ Y n) :=
  if 1 ≤ μ then nebOp ρ χ Y (gP_mem_Δ0Q p μ hp) (cd μ) (τ μ) n else 1

/-- **[C] `thm:nebrelations`(2)** (`p ∤ N`, `6 ∈ R^×`): `T_{p^{ν+2}} = T_p T_{p^{ν+1}} − χ(p) p c T_{p^ν}` for the full
operators `T_{p^ν} = ∑ (χ(p) c)^j T_{Δ^prim_{p^{ν−2j}}}` (`T(p,p) = χ(p) c`, `nebTpp`; `c = p^n` for `Sym^n`), given the
double-coset products of the primitive operators with character, the literature hypotheses `hTpchi0` (`μ = 0`) and
`hTpchi` (`μ ≥ 1`): Shimura Thm. 3.34, 3.35 and §3.3 in `𝓗(Γ₀(N), Δ₀(N))`, transported to `H^n` by the ring
homomorphism (Brown III.9–10 with [C] `obl:algebra`); locators **not** verified. -/
theorem nebRelations (p : ℕ) (hp : p.Prime) (_hpN : ¬ (p : ℤ) ∣ N) (_h6 : IsUnit (6 : R))
    (cd : ∀ μ, RelTrans (Bpp (Γ0Q N) (Γ0Q N) (gP p μ hp)) (Γ0Q N)) (τ : ∀ μ, TwistedChainDatum ΓQ (gP p μ hp) Y)
    (c : R) (n : ℕ)
    (hTpchi0 : nebTprim ρ χ Y p hp cd τ n 1 * nebTprim ρ χ Y p hp cd τ n 1 =
      nebTprim ρ χ Y p hp cd τ n 2 + (((p : R) + 1) * (χ (p : ZMod N) * c)) • (1 : Module.End R (HN ρ χ Y n)))
    (hTpchi : ∀ μ, 1 ≤ μ → nebTprim ρ χ Y p hp cd τ n 1 * nebTprim ρ χ Y p hp cd τ n (μ + 1) =
      nebTprim ρ χ Y p hp cd τ n (μ + 2) + ((p : R) * (χ (p : ZMod N) * c)) • nebTprim ρ χ Y p hp cd τ n μ)
    (ν : ℕ) :
    fullOf (nebTprim ρ χ Y p hp cd τ n) (χ (p : ZMod N) * c) (ν + 2) =
      nebTprim ρ χ Y p hp cd τ n 1 * fullOf (nebTprim ρ χ Y p hp cd τ n) (χ (p : ZMod N) * c) (ν + 1) -
        ((p : R) * (χ (p : ZMod N) * c)) • fullOf (nebTprim ρ χ Y p hp cd τ n) (χ (p : ZMod N) * c) ν :=
  fullOf_rec _ (by simp [nebTprim]) _ _ hTpchi0 hTpchi ν

/-- **[C] `thm:nebrelations`(3)** (`p ∣ N`): `U_{p^ν} = (U_p)^ν`, given the product `U_p U_{p^μ} = U_{p^{μ+1}}` of the
primitive double cosets in `𝓗(Γ₀(N), Δ₀(N))` (literature hypothesis `hUchi`, Shimura §3.3, transported; locator not
verified). -/
theorem nebRelations_U (p : ℕ) (hp : p.Prime) (_hpN : (p : ℤ) ∣ N) (_h6 : IsUnit (6 : R))
    (cd : ∀ μ, RelTrans (Bpp (Γ0Q N) (Γ0Q N) (gP p μ hp)) (Γ0Q N)) (τ : ∀ μ, TwistedChainDatum ΓQ (gP p μ hp) Y)
    (n : ℕ)
    (hUchi : ∀ μ, 1 ≤ μ → nebTprim ρ χ Y p hp cd τ n 1 * nebTprim ρ χ Y p hp cd τ n μ =
      nebTprim ρ χ Y p hp cd τ n (μ + 1)) (ν : ℕ) :
    nebTprim ρ χ Y p hp cd τ n ν = nebTprim ρ χ Y p hp cd τ n 1 ^ ν := by
  induction ν with
  | zero => simp [nebTprim]
  | succ k ih =>
    rcases Nat.eq_zero_or_pos k with rfl | hk
    · simp
    · rw [← hUchi k hk, ih, pow_succ']

end NebRelations

end GL2.M4

end
