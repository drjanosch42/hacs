import Mathlib.RepresentationTheory.Homological.GroupCohomology.Basic
import Mathlib.RepresentationTheory.Rep.Iso
import Mathlib.CategoryTheory.Preadditive.Projective.Basic
import Mathlib.RepresentationTheory.Rep.Res
import Mathlib.GroupTheory.Coset.Card

/-!
# G4 (L18-7): the projective reduction

[GT] `lem:permproj`, `thm:projred`, mirroring [C] v1.17.0 `rmk:orbifold` (the relative-projectivity half).

**Permutation modules with invertible stabiliser order are projective.**  For a group `G`, a finite subgroup `S ≤ G`
with `|S|` a unit of the commutative ring `k`, the permutation representation `k[G/S]` is a retract of the left
regular representation `k[G]` (`permRetract`): `r : g ↦ gS`, `i : xS ↦ |S|⁻¹ ∑_{s ∈ S} x s`.  Hence it is projective
in `Rep k G` (`projective_perm`), and so is every finite biproduct of such.

**The reduction.**  A resolution of the trivial representation `k` by projectives computes group cohomology
(mathlib's `groupCohomologyIso`, via `groupCohomologyIsoExt`); `projRed` packages this for a complex whose terms are
projective, the form [C] `rmk:orbifold` uses with the permutation terms above.
-/

noncomputable section

namespace GL2.Cone.ProjRed

open CategoryTheory MonoidAlgebra

universe u

variable {k G : Type u} [CommRing k] [Group G]

/-- `∑_{s ∈ S} e_{a s}` depends only on the coset `aS`. -/
lemma sum_single_coset (S : Subgroup G) [Fintype S] {a b : G}
    (h : (QuotientGroup.mk a : G ⧸ S) = QuotientGroup.mk b) :
    (∑ s : S, single (a * s) (1 : k) : k[G]) = ∑ s : S, single (b * s) 1 := by
  rw [QuotientGroup.eq] at h
  let t : S := ⟨a⁻¹ * b, h⟩
  rw [← Equiv.sum_comp (Equiv.mulLeft t) (fun s : S => (single (a * s) (1 : k) : k[G]))]
  refine Finset.sum_congr rfl fun s _ => ?_
  simp [t, mul_assoc]

variable (k) in
/-- The averaging section `k[G/S] → k[G]`, `xS ↦ c ∑_{s ∈ S} x s` (`c = |S|⁻¹`). -/
def avgL (S : Subgroup G) [Fintype S] (c : k) : k[G ⧸ S] →ₗ[k] k[G] :=
  Finsupp.lift k[G] k (G ⧸ S) (fun x => c • ∑ s : S, single (x.out * s) (1 : k)) ∘ₗ
    (coeffLinearEquiv k).toLinearMap

lemma avgL_single (S : Subgroup G) [Fintype S] (c : k) (x : G ⧸ S) (r : k) :
    avgL k S c (single x r) = r • c • ∑ s : S, single (x.out * s) (1 : k) := by
  simp [avgL]

lemma avgL_single' (S : Subgroup G) [Fintype S] (c : k) (a : G) (r : k) :
    avgL k S c (single (QuotientGroup.mk a) r) = r • c • ∑ s : S, single (a * s) (1 : k) := by
  rw [avgL_single, sum_single_coset S (QuotientGroup.out_eq' (QuotientGroup.mk a : G ⧸ S))]

/-- The averaging section as a morphism of representations. -/
def avg (S : Subgroup G) [Fintype S] (c : k) : Rep.ofMulAction k G (G ⧸ S) ⟶ Rep.leftRegular k G :=
  Rep.ofHom ⟨avgL k S c, fun g => by
    ext x
    induction x using QuotientGroup.induction_on with
    | H a =>
      simp only [LinearMap.coe_comp, Function.comp_apply, lsingle_apply, Representation.ofMulAction_single]
      rw [show g • (QuotientGroup.mk a : G ⧸ S) = QuotientGroup.mk (g * a) from rfl, avgL_single', avgL_single',
        map_smul, map_smul, map_sum]
      simp [mul_assoc]⟩

/-- The projection `k[G] → k[G/S]`, `g ↦ gS`. -/
def proj (S : Subgroup G) : Rep.leftRegular k G ⟶ Rep.ofMulAction k G (G ⧸ S) :=
  Rep.leftRegularHom (Rep.ofMulAction k G (G ⧸ S)) (single (QuotientGroup.mk 1) 1)

/-- **`k[G/S]` is a retract of `k[G]`** when `|S|` is a unit of `k`. -/
def permRetract (S : Subgroup G) [Fintype S] (hS : IsUnit (Fintype.card S : k)) :
    Retract (Rep.ofMulAction k G (G ⧸ S)) (Rep.leftRegular k G) where
  i := avg S (↑hS.unit⁻¹ : k)
  r := proj S
  retract := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext
    apply MonoidAlgebra.lhom_ext'; intro x
    induction x using QuotientGroup.induction_on with
    | H a =>
      apply LinearMap.ext_ring
      change (proj S).hom (avgL k S (↑hS.unit⁻¹ : k) (single (QuotientGroup.mk a) 1)) = single (QuotientGroup.mk a) 1
      rw [avgL_single', one_smul, map_smul, map_sum]
      have h1 : ∀ s : S, (proj (k := k) S).hom (single (a * s) 1) = single (QuotientGroup.mk a : G ⧸ S) 1 := by
        intro s
        rw [proj, Rep.leftRegularHom_hom_single, one_smul]
        simp only [Representation.ofMulAction_single]
        congr 1
        rw [show (a * (s : G)) • (QuotientGroup.mk 1 : G ⧸ S) = QuotientGroup.mk (a * s * 1) from rfl, mul_one,
          QuotientGroup.eq]
        simp
      simp only [h1, Finset.sum_const, Finset.card_univ, ← Nat.cast_smul_eq_nsmul k, smul_smul]
      rw [IsUnit.val_inv_mul, one_smul]

/-- **`lem:permproj` (the quotient form).** `k[G/S]` is projective when `|S|` is a unit of `k`. -/
theorem projective_perm (S : Subgroup G) [Fintype S] (hS : IsUnit (Fintype.card S : k)) :
    Projective (Rep.ofMulAction k G (G ⧸ S)) :=
  (permRetract S hS).projective

/-! ### Any permutation module with finite stabilisers of invertible order -/

section General

variable {X : Type u} [MulAction G X]

open MulAction

/-- The orbit representative `o x` (constant on orbits). -/
def orep (x : X) : X := (Quotient.mk (orbitRel G X) x).out

lemma orep_smul (g : G) (x : X) : orep (G := G) (g • x) = orep (G := G) x := by
  unfold orep; congr 1; exact Quotient.sound (by exact ⟨g, rfl⟩)

lemma exists_smul_orep (x : X) : ∃ w : G, w • orep (G := G) x = x := by
  have : (Quotient.mk (orbitRel G X) x).out ∈ orbit G x :=
    Quotient.exact (Quotient.out_eq (Quotient.mk (orbitRel G X) x))
  obtain ⟨w, hw⟩ := this
  exact ⟨w⁻¹, by rw [orep, ← hw, inv_smul_smul]⟩

/-- A chosen `w x` with `w x • o x = x`. -/
def wit (x : X) : G := (exists_smul_orep (G := G) x).choose

lemma wit_spec (x : X) : wit (G := G) x • orep (G := G) x = x := (exists_smul_orep x).choose_spec

variable (hfin : ∀ x : X, Finite (stabilizer G x))

/-- The stabiliser of `y`, as a finite type. -/
@[instance_reducible] def stabFt (y : X) : Fintype (stabilizer G y) := @Fintype.ofFinite _ (hfin _)

variable (k) in
/-- `∑_{s ∈ G_y} e_{(y, a s)}` in the free representation on `X`. -/
def sumAt (y : X) (a : G) : X →₀ k[G] :=
  @Finset.sum _ _ _ (@Finset.univ _ (stabFt hfin y)) fun s => Finsupp.single y (single (a * s) (1 : k))

/-- `sumAt y a` depends only on `a · y`. -/
lemma sumAt_eq {y : X} {a b : G} (h : a • y = b • y) : sumAt k hfin y a = sumAt k hfin y b := by
  let _ := stabFt hfin y
  have hm : a⁻¹ * b ∈ stabilizer G y := by rw [mem_stabilizer_iff, mul_smul, ← h, inv_smul_smul]
  let t : stabilizer G y := ⟨a⁻¹ * b, hm⟩
  unfold sumAt
  rw [← Equiv.sum_comp (Equiv.mulLeft t)]
  refine Finset.sum_congr rfl fun s _ => ?_
  simp [t, mul_assoc]

lemma free_sumAt (g : G) (y : X) (a : G) :
    Representation.free k G X g (sumAt k hfin y a) = sumAt k hfin y (g * a) := by
  unfold sumAt
  rw [map_sum]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [Representation.free_single_single, mul_assoc]

/-- The projection `⊕_X k[G] → k[X]`, `e_{(x, g)} ↦ g x`. -/
def projX : Rep.free k G X ⟶ Rep.ofMulAction k G X :=
  Rep.freeLift (k := k) (G := G) (A := Rep.ofMulAction k G X) (fun x => single x (1 : k))

lemma projX_sumAt (y : X) (a : G) :
    (projX (k := k) (G := G) (X := X)).hom (sumAt k hfin y a) = (Nat.card (stabilizer G y) : k) • single (a • y) 1 := by
  let _ := stabFt hfin y
  unfold sumAt
  rw [map_sum]
  have : ∀ s : stabilizer G y, (projX (k := k) (G := G) (X := X)).hom (Finsupp.single y (single (a * s) 1)) =
      single (a • y) (1 : k) := by
    intro s
    simp only [projX, Rep.hom_ofHom, Representation.freeLift_single_single, one_smul, 
      Representation.ofMulAction_single, mul_smul]
    rw [show (s : G) • y = y from s.2]
  simp only [this, Finset.sum_const, Finset.card_univ, Nat.card_eq_fintype_card, ← Nat.cast_smul_eq_nsmul k]

variable (hunit : ∀ x : X, IsUnit (Nat.card (stabilizer G x) : k))

variable (k) in
/-- The section `k[X] → ⊕_X k[G]`, `x ↦ |G_{o x}|⁻¹ ∑_{s ∈ G_{o x}} e_{(o x, w x · s)}`. -/
def secL : k[X] →ₗ[k] (X →₀ k[G]) :=
  Finsupp.lift (X →₀ k[G]) k X (fun x => (↑(hunit (orep (G := G) x)).unit⁻¹ : k) •
      sumAt k hfin (orep (G := G) x) (wit (G := G) x)) ∘ₗ (coeffLinearEquiv k).toLinearMap

lemma secL_single (x : X) (r : k) :
    secL k hfin hunit (single x r) =
      r • (↑(hunit (orep (G := G) x)).unit⁻¹ : k) • sumAt k hfin (orep (G := G) x) (wit (G := G) x) := by
  simp [secL]

/-- The section as a morphism of representations. -/
def secX : Rep.ofMulAction k G X ⟶ Rep.free k G X :=
  Rep.ofHom ⟨secL k hfin hunit, fun g => by
    apply MonoidAlgebra.lhom_ext'; intro x
    apply LinearMap.ext_ring
    simp only [LinearMap.coe_comp, Function.comp_apply, lsingle_apply, Representation.ofMulAction_single]
    rw [secL_single, secL_single, map_smul, map_smul, free_sumAt]
    have ho := orep_smul (G := G) g x
    congr 2
    · rw [ho]
    · rw [sumAt_eq hfin (y := orep (G := G) (g • x)) (b := g * wit (G := G) x)]
      · rw [ho]
      · rw [wit_spec, ho, mul_smul, wit_spec]⟩

/-- **`k[X]` is a retract of the free representation on `X`** when every stabiliser is finite of invertible order. -/
def permRetractX : Retract (Rep.ofMulAction k G X) (Rep.free k G X) where
  i := secX hfin hunit
  r := projX
  retract := by
    apply Rep.hom_ext; apply Representation.IntertwiningMap.ext
    apply MonoidAlgebra.lhom_ext'; intro x
    apply LinearMap.ext_ring
    change (projX (k := k) (G := G) (X := X)).hom (secL k hfin hunit (single x 1)) = single x 1
    rw [secL_single, one_smul, map_smul, projX_sumAt, wit_spec, smul_smul, IsUnit.val_inv_mul, one_smul]

include hfin hunit in
/-- **`lem:permproj`.** A permutation representation `k[X]` whose stabilisers are all finite of order invertible in
`k` is projective in `Rep k G`. -/
theorem projective_ofMulAction : Projective (Rep.ofMulAction k G X) :=
  (permRetractX hfin hunit).projective

end General

/-! ### The reduction: a projective resolution computes group cohomology -/

section Reduction

variable (C : ChainComplex (Rep k G) ℕ) (π : C ⟶ (ChainComplex.single₀ (Rep k G)).obj (Rep.trivial k G k))
  [QuasiIso π]

/-- A complex with projective terms and a quasi-isomorphism to `k` is a projective resolution of `k`. -/
def projRes (hP : ∀ n, Projective (C.X n)) : ProjectiveResolution (Rep.trivial k G k) where
  complex := C
  projective := hP
  π := π

/-- **`thm:projred`.** If every term of `C` is projective and `π : C → k` is a quasi-isomorphism, then for every
representation `A`, `H^n(Hom_G(C, A)) ≅ H^n(G; A)` (mathlib's `groupCohomologyIso`, i.e. `groupCohomologyIsoExt` and
`ProjectiveResolution.isoExt`). -/
def projRed (hP : ∀ n, Projective (C.X n)) (A : Rep k G) (n : ℕ) :
    groupCohomology A n ≅ (C.linearYonedaObj k A).homology n :=
  groupCohomologyIso A n (projRes C π hP)

/-- **`thm:projred`, the permutation form** ([C] `rmk:orbifold`): if every term of `C` is (isomorphic to) a
permutation representation `k[X_n]` whose stabilisers are all finite of order invertible in `k`, and `C → k` is a
quasi-isomorphism, then `H^n(Hom_G(C, A)) ≅ H^n(G; A)`. -/
def projRedPerm (X : ℕ → Type u) [∀ n, MulAction G (X n)] (e : ∀ n, C.X n ≅ Rep.ofMulAction k G (X n))
    (hfin : ∀ n (x : X n), Finite (MulAction.stabilizer G x))
    (hunit : ∀ n (x : X n), IsUnit (Nat.card (MulAction.stabilizer G x) : k)) (A : Rep k G) (n : ℕ) :
    groupCohomology A n ≅ (C.linearYonedaObj k A).homology n :=
  projRed C π (fun m => by
    have := projective_ofMulAction (hfin m) (hunit m)
    exact Projective.of_iso (e m).symm this) A n

end Reduction

/-! ### Restriction along an injective homomorphism -/

section Res

variable {H : Type u} [Group H]

/-- A divisor of a unit-valued natural number is a unit. -/
lemma isUnit_natCast_of_dvd {m n : ℕ} (h : m ∣ n) (hn : IsUnit (n : k)) : IsUnit (m : k) := by
  obtain ⟨c, rfl⟩ := h
  rw [Nat.cast_mul] at hn
  exact isUnit_of_mul_isUnit_left hn

/-- Restricting a permutation representation with finite stabilisers of invertible order along an **injective**
homomorphism `f : H → G` keeps it projective: the `H`-stabilisers are `f⁻¹(G_x)`, finite of order dividing `|G_x|`. -/
theorem projective_res_perm (f : H →* G) (hf : Function.Injective f) {X : Type u} [MulAction G X]
    (hfin : ∀ x : X, Finite (MulAction.stabilizer G x))
    (hunit : ∀ x : X, IsUnit (Nat.card (MulAction.stabilizer G x) : k)) :
    Projective ((Rep.resFunctor f).obj (Rep.ofMulAction k G X)) := by
  let _ : MulAction H X := MulAction.compHom X f
  have hinj : ∀ x : X, Function.Injective ((f.subgroupComap (MulAction.stabilizer G x)) :
      (MulAction.stabilizer G x).comap f → MulAction.stabilizer G x) := by
    intro x a b hab
    exact Subtype.ext (hf (congrArg Subtype.val hab))
  have hfinH : ∀ x : X, Finite (MulAction.stabilizer H x) := fun x => by
    have := hfin x
    exact Finite.of_injective _ (hinj x)
  have hunitH : ∀ x : X, IsUnit (Nat.card (MulAction.stabilizer H x) : k) := fun x => by
    have := hfin x
    exact isUnit_natCast_of_dvd (Subgroup.card_dvd_of_injective _ (hinj x)) (hunit x)
  exact projective_ofMulAction (k := k) hfinH hunitH

/-- **The restricted resolution.** A permutation resolution `P` of `k` over `G` restricts along an injective
`f : H → G` to a projective resolution over `H` (of the restriction of `k`, which is `k`). -/
def resProjRes (f : H →* G) (hf : Function.Injective f) (P : ProjectiveResolution (Rep.trivial k G k))
    (X : ℕ → Type u) [∀ n, MulAction G (X n)] (e : ∀ n, P.complex.X n ≅ Rep.ofMulAction k G (X n))
    (hfin : ∀ n (x : X n), Finite (MulAction.stabilizer G x))
    (hunit : ∀ n (x : X n), IsUnit (Nat.card (MulAction.stabilizer G x) : k)) :
    ProjectiveResolution ((Rep.resFunctor f).obj (Rep.trivial k G k)) where
  complex := ((Rep.resFunctor f).mapHomologicalComplex _).obj P.complex
  projective n := by
    have := projective_res_perm (k := k) f hf (hfin n) (hunit n)
    exact Projective.of_iso ((Rep.resFunctor f).mapIso (e n)).symm this
  π := ((Rep.resFunctor f).mapHomologicalComplex _).map P.π ≫
    (HomologicalComplex.singleMapHomologicalComplex _ _ _).hom.app _

end Res

/-! ### Existence and homotopy uniqueness of augmentation-compatible chain maps (`lem:tauexists`) -/

section Lift

variable {H : Type u} [Group H] {Z : Rep k H} (P Q : ProjectiveResolution Z)

/-- **Existence** (the abstract half of `lem:tauexists`): a chain map `P → Q` over `id_Z` exists. -/
theorem lift_exists : ∃ f : P.complex ⟶ Q.complex, f ≫ Q.π = P.π :=
  ⟨ProjectiveResolution.lift (𝟙 _) P Q, by simp⟩

/-- **Homotopy uniqueness** (the abstract form of `lem:tauexists`(2)): any two chain maps `P → Q` compatible with
the augmentations are chain homotopic. -/
theorem homotopic_of_aug (f g : P.complex ⟶ Q.complex) (hf : f ≫ Q.π = P.π) (hg : g ≫ Q.π = P.π) :
    Nonempty (Homotopy f g) :=
  ⟨ProjectiveResolution.liftHomotopy (𝟙 _) f g (by simp [hf]) (by simp [hg])⟩

end Lift

/-! ### `lem:tauexists` in the twisted form -/

section Twisted

variable {H : Type u} [Group H] (P : ProjectiveResolution (Rep.trivial k G k))
    (X : ℕ → Type u) [∀ n, MulAction G (X n)] (e : ∀ n, P.complex.X n ≅ Rep.ofMulAction k G (X n))
    (hfin : ∀ n (x : X n), Finite (MulAction.stabilizer G x))
    (hunit : ∀ n (x : X n), IsUnit (Nat.card (MulAction.stabilizer G x) : k))
    (f₁ f₂ : H →* G) (hf₁ : Function.Injective f₁) (hf₂ : Function.Injective f₂)

/-- **`lem:tauexists`, twisted form, over `k` with the stabiliser orders invertible.** For a permutation resolution
`P` of `k` over `G` and two injective homomorphisms `f₁, f₂ : H → G` (for a `g`-twisted chain map: `f₁` the inclusion
`Γ'' ≤ Γ`, `f₂ = h ↦ g⁻¹hg`), an `H`-chain map `res_{f₁} P → res_{f₂} P` over `id_k` exists, and any two such are
chain homotopic. -/
theorem twisted_exists_unique_upto_homotopy :
    (∃ τ : (resProjRes f₁ hf₁ P X e hfin hunit).complex ⟶ (resProjRes f₂ hf₂ P X e hfin hunit).complex,
        τ ≫ (resProjRes f₂ hf₂ P X e hfin hunit).π = (resProjRes f₁ hf₁ P X e hfin hunit).π) ∧
    ∀ τ τ' : (resProjRes f₁ hf₁ P X e hfin hunit).complex ⟶ (resProjRes f₂ hf₂ P X e hfin hunit).complex,
      τ ≫ (resProjRes f₂ hf₂ P X e hfin hunit).π = (resProjRes f₁ hf₁ P X e hfin hunit).π →
      τ' ≫ (resProjRes f₂ hf₂ P X e hfin hunit).π = (resProjRes f₁ hf₁ P X e hfin hunit).π →
      Nonempty (Homotopy τ τ') :=
  ⟨lift_exists _ _, fun τ τ' h h' => homotopic_of_aug _ _ τ τ' h h'⟩

end Twisted

end GL2.Cone.ProjRed

end
