import GL2.MP.Cover

/-!
# MP-L1, row L2 (slot 22): the twisted cochain complex of the MP set, `d ∘ d = 0`

[GT] `gt_part_MP.tex`, rows `def:mp-cochains`, `def:mp-ecoch`, `thm:mp-cochequiv`, `thm:mp-dd`,
`lem:mp-cochains-natural`, backing [C] `def:mpset` (the MP complex) and [C] `thm:mpcomparison` (3) (the on-the-nose
identification with the equivariant cochains on the labelled Čech nerve).  See `GL2.MP.Cover` for the conventions.

For `A : Rep k G` (any commutative ring `k`, any representation; nothing is assumed invertible):

* `Coch C A n = (MPSet n → A)`, `dC`: `(df)(s) = γ_1 · f(d_0 s) + ∑_{i=1}^{n+1} (-1)^i f(d_i s)` with `γ_1 = g_1`;
* `ECoch C A n`: the `G`-equivariant functions `𝔉'_n → A`, with the untwisted alternating differential `dE`;
* `cochEquiv`: `ECoch C A n ≃ₗ Coch C A n` (restriction to the normal forms `g_0 = 1`), intertwining `dE` and `dC`
  (`cochEquiv_dE`);
* `dC_dC`: `d ∘ d = 0`; `mpComplex C A : CochainComplex (ModuleCat k) ℕ`; `dC_natural`, `mpComplexMap`: naturality in `A`.

These are the unnormalised cochains (all strings); [C] `def:mpset` uses the normalised subcomplex (functions vanishing on
degenerate strings), chain homotopy equivalent to this one ([C] `thm:mpcomparison`, proof of (2)).
-/

namespace GL2.MP

namespace Cover

open CategoryTheory

universe u

variable {G : Type u} [Group G] (C : Cover G) {k : Type u} [CommRing k] (A : Rep.{u} k G)

/-- **The MP cochains** `C^n(S; A) = (S_n → A)`. -/
abbrev Coch (n : ℕ) := C.MPSet n → A

/-- **The twisted differential** `(df)(s) = γ_1 · f(d_0 s) + ∑_{i ≥ 1} (-1)^i f(d_i s)` ([C] `def:mpset`). -/
def dC (n : ℕ) : C.Coch A n →ₗ[k] C.Coch A (n + 1) where
  toFun f s := A.ρ ((s.1 1).1) (f (face 0 s)) + ∑ i : Fin (n + 1), ((-1 : k) ^ ((i : ℕ) + 1)) • f (face i.succ s)
  map_add' f g := by
    funext s
    simp only [Pi.add_apply, map_add, smul_add, Finset.sum_add_distrib]
    abel
  map_smul' r f := by
    funext s
    simp only [Pi.smul_apply, map_smul, RingHom.id_apply, smul_add, Finset.smul_sum]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [smul_comm]

lemma dC_apply (n : ℕ) (f : C.Coch A n) (s : C.MPSet (n + 1)) :
    C.dC A n f s = A.ρ ((s.1 1).1) (f (face 0 s)) + ∑ i : Fin (n + 1), ((-1 : k) ^ ((i : ℕ) + 1)) • f (face i.succ s) :=
  rfl

/-! ### Equivariant cochains on the labelled Čech nerve -/

/-- **The equivariant cochains** `Hom_G(k[𝔉'_n], A) = {F : 𝔉'_n → A | F(g x) = g F(x)}`. -/
def ECoch (n : ℕ) : Submodule k (C.F' n → A) where
  carrier := {F | ∀ (g : G) x, F (g • x) = A.ρ g (F x)}
  add_mem' {F F'} hF hF' g x := by simp only [Pi.add_apply, hF g x, hF' g x, map_add]
  zero_mem' g x := by simp
  smul_mem' r F hF g x := by simp only [Pi.smul_apply, hF g x, map_smul]

/-- The untwisted alternating differential `(dF)(x) = ∑_i (-1)^i F(d_i x)` on functions on `𝔉'`. -/
def dE (n : ℕ) : (C.F' n → A) →ₗ[k] (C.F' (n + 1) → A) where
  toFun F x := ∑ i : Fin (n + 2), ((-1 : k) ^ (i : ℕ)) • F (F'.del i x)
  map_add' F F' := by funext x; simp [Finset.sum_add_distrib, smul_add]
  map_smul' r F := by
    funext x
    simp only [Pi.smul_apply, RingHom.id_apply, Finset.smul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [smul_comm]

lemma dE_apply (n : ℕ) (F : C.F' n → A) (x : C.F' (n + 1)) :
    C.dE A n F x = ∑ i : Fin (n + 2), ((-1 : k) ^ (i : ℕ)) • F (F'.del i x) := rfl

lemma dE_mem (n : ℕ) {F : C.F' n → A} (hF : F ∈ C.ECoch A n) : C.dE A n F ∈ C.ECoch A (n + 1) := by
  intro g x
  simp only [dE_apply, F'.del_smul, hF g, map_sum, map_smul]

lemma neg_one_pow_smul_smul {M : Type*} [AddCommGroup M] [Module k M] (a b : ℕ) (m : M) :
    ((-1 : k) ^ a) • ((-1 : k) ^ b) • m = ((-1 : ℤ) ^ a * (-1) ^ b) • m := by
  rw [mul_smul, ← Int.cast_smul_eq_zsmul k ((-1 : ℤ) ^ a), ← Int.cast_smul_eq_zsmul k ((-1 : ℤ) ^ b)]
  push_cast; rfl

/-- **`d ∘ d = 0` on `𝔉'`** (`alt_sum_sq`). -/
theorem dE_dE (n : ℕ) (F : C.F' n → A) : C.dE A (n + 1) (C.dE A n F) = 0 := by
  classical
  funext x
  simp only [dE_apply, Finset.smul_sum, Pi.zero_apply]
  let Φ : C.Lab n → A := fun y => if h : C.Cov y then F ⟨y, h⟩ else 0
  have hΦ : ∀ (i : Fin (n + 3)) (j : Fin (n + 2)),
      F (F'.del j (F'.del i x)) = Φ (GL2.MP.del j (GL2.MP.del i x.1)) := fun i j => by
    simp only [Φ]; rw [dif_pos ((x.2.del i).del j)]; rfl
  simp_rw [neg_one_pow_smul_smul, hΦ]
  exact alt_sum_sq x.1 Φ

/-! ### The on-the-nose identification ([C] `thm:mpcomparison` (3)) -/

/-- **`Hom_G(k[𝔉'_n], A) = C^n(S; A)`**: restriction to the normal forms; inverse `F(x) = g_0 · f(nf x)`. -/
def cochEquiv (n : ℕ) : C.ECoch A n ≃ₗ[k] C.Coch A n where
  toFun F s := F.1 s.toF'
  map_add' F F' := rfl
  map_smul' r F := rfl
  invFun f := ⟨fun x => A.ρ ((x.1 0).1) (f (F'.norm x)), fun g x => by
    show A.ρ (g * (x.1 0).1) (f (F'.norm (g • x))) = A.ρ g (A.ρ ((x.1 0).1) (f (F'.norm x)))
    rw [F'.norm_smul, map_mul, Module.End.mul_apply]⟩
  left_inv F := by
    apply Subtype.ext
    funext x
    show A.ρ ((x.1 0).1) (F.1 (F'.norm x).toF') = F.1 x
    rw [← F.2 _ _, F'.smul_norm]
  right_inv f := by
    funext s
    show A.ρ ((s.1 0).1) (f (F'.norm s.toF')) = f s
    rw [MPSet.norm_toF', s.2.2, map_one, Module.End.one_apply]

lemma cochEquiv_apply (n : ℕ) (F : C.ECoch A n) (s : C.MPSet n) : C.cochEquiv A n F s = F.1 s.toF' := rfl

/-- **The identification intertwines the differentials**: the twist `γ_1` of `dC` is the re-normalisation of `d_0`. -/
theorem cochEquiv_dE (n : ℕ) (F : C.ECoch A n) :
    C.cochEquiv A (n + 1) ⟨C.dE A n F.1, C.dE_mem A n F.2⟩ = C.dC A n (C.cochEquiv A n F) := by
  funext s
  rw [cochEquiv_apply, dC_apply]
  show ∑ i : Fin (n + 2), ((-1 : k) ^ (i : ℕ)) • F.1 (F'.del i s.toF') = _
  rw [Fin.sum_univ_succ]
  congr 1
  · rw [Fin.val_zero, pow_zero, one_smul, del_zero_toF', F.2]; rfl
  · refine Finset.sum_congr rfl fun i _ => ?_
    have : F'.del i.succ s.toF' = (face i.succ s).toF' := Subtype.ext (face_succ_val i s).symm
    rw [this, Fin.val_succ]; rfl

/-- **`d ∘ d = 0`** for the twisted MP cochains ([C] `def:mpset`). -/
theorem dC_dC (n : ℕ) (f : C.Coch A n) : C.dC A (n + 1) (C.dC A n f) = 0 := by
  obtain ⟨F, rfl⟩ := (C.cochEquiv A n).surjective f
  rw [← cochEquiv_dE, ← cochEquiv_dE]
  have : (⟨C.dE A (n + 1) (C.dE A n F.1), C.dE_mem A (n + 1) (C.dE_mem A n F.2)⟩ : C.ECoch A (n + 2)) = 0 :=
    Subtype.ext (C.dE_dE A n F.1)
  rw [this, map_zero]

/-- **The MP complex** `C^•(S; A)` ([C] `def:mpset`, unnormalised) as a cochain complex of `k`-modules. -/
noncomputable def mpComplex : CochainComplex (ModuleCat.{u} k) ℕ :=
  CochainComplex.of (fun n => ModuleCat.of k (C.Coch A n)) (fun n => ModuleCat.ofHom (C.dC A n)) fun n => by
    ext f : 2
    simp only [ModuleCat.hom_comp, ModuleCat.hom_ofHom, LinearMap.coe_comp, Function.comp_apply, ModuleCat.hom_zero,
      LinearMap.zero_apply]
    exact C.dC_dC A n f

variable {A}

/-- Post-composition with a morphism of representations. -/
def cochMap {B : Rep.{u} k G} (φ : A ⟶ B) (n : ℕ) : C.Coch A n →ₗ[k] C.Coch B n where
  toFun f s := φ.hom (f s)
  map_add' f g := by funext s; simp [map_add]
  map_smul' r f := by funext s; simp [map_smul]

/-- **Naturality in the coefficients**: `φ ∘ d = d ∘ φ` for every morphism `φ : A ⟶ B` of representations. -/
theorem dC_natural {B : Rep.{u} k G} (φ : A ⟶ B) (n : ℕ) (f : C.Coch A n) :
    C.cochMap φ (n + 1) (C.dC A n f) = C.dC B n (C.cochMap φ n f) := by
  funext s
  simp only [cochMap, LinearMap.coe_mk, AddHom.coe_mk, dC_apply, map_add, map_sum, map_smul, Rep.hom_comm_apply]

/-- **The induced map of MP complexes.** -/
noncomputable def mpComplexMap {B : Rep.{u} k G} (φ : A ⟶ B) : C.mpComplex A ⟶ C.mpComplex B :=
  CochainComplex.ofHom (fun n => ModuleCat.ofHom (C.cochMap φ n)) fun n => by
    simp only [mpComplex, CochainComplex.of_d]
    ext f : 2
    exact (C.dC_natural φ n f).symm

end Cover

end GL2.MP
