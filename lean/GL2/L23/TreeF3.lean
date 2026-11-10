/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.TreeF2

/-!
# LV-2 (slot 22): `dim H^q(SL₂(ℤ); A)` over a field of characteristic `≠ 2`, from the tree complex

`L23_master_plan.md` §4.3, `LV1_audit.md` §5.2 (re-derived in `LV2_predictions.md` §C), on LR-Y's `treeCoch`
(`model7TreeCochCoh`: `H^q(SL₂(ℤ); A) ≅ H^q(A² → A³ → A³ → ⋯)`). Let `K` be a field with `2 ≠ 0`, `A` a finite-dimensional
`K`-representation, `ε = A.ρ (−1)`; `S² = U³ = −I` in `SL₂(ℤ)`.

* **`−I` acts by `−1`** (`finrank_H_eq_zero_of_negOne`): then `σ = (1 − S, 1 − U)` is injective (`Sx = x` gives
  `x = S²x = −x`), `N_S = N_U = 0`, `e_m = 2` for even `m` and `0` for odd `m`; every differential `d^{2j+1}` has rank
  `dim A` (onto through the edge coordinate) and every `d^{2j}` has rank `2 dim A`, so **`H^q = 0` for every `q`**. This is
  [C] `lem:centralminusone`'s "`H^q(Γ; V_R) = 0` when `2 ∈ R^×` and `−I` acts by `−1`" on the tree complex.
* **`−I` acts trivially** (`finrank_H_of_trivNegOne`): `N_S = 2(1 + S)`, `N_U = 2(1 + U + U²)`, `e_m = 0` (`m` even), `2`
  (`m` odd). With `s = rank σ`, `r_S⁺ = rank (1 + S)`, `r_S⁻ = rank (1 − S)`, `r₃ = rank (1 + U + U²)`, `r_U⁻ = rank (1 − U)`:
  * `dim H⁰ + s = dim A`;
  * `dim H¹ + s + r_S⁺ + r₃ = 2 dim A`;
  * `dim H^{q+2} + r₃ + r_U⁻ = dim A` for every `q` (using `r_S⁺ + r_S⁻ = dim A`, `finrank_range_add_of_sq_one`).

  **Degree 1 differs from degrees `≥ 2`.**
-/

noncomputable section

namespace GL2.L23.TreeF3

open CategoryTheory GL2.MP Module
open scoped MatrixGroups

/-! ### Linear algebra -/

section LinAlg

variable {K : Type} [Field K]

/-- `p.prod q ≃ p × q`. -/
def prodSubEquiv {M N : Type} [AddCommGroup M] [Module K M] [AddCommGroup N] [Module K N] (p : Submodule K M)
    (q : Submodule K N) : (p.prod q) ≃ₗ[K] p × q where
  toFun x := (⟨x.1.1, (Submodule.mem_prod.mp x.2).1⟩, ⟨x.1.2, (Submodule.mem_prod.mp x.2).2⟩)
  invFun y := ⟨(y.1.1, y.2.1), Submodule.mem_prod.mpr ⟨y.1.2, y.2.2⟩⟩
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  left_inv _ := rfl
  right_inv _ := rfl

lemma finrank_range_prodMap {M N P Q : Type} [AddCommGroup M] [Module K M] [AddCommGroup N] [Module K N]
    [AddCommGroup P] [Module K P] [AddCommGroup Q] [Module K Q] [FiniteDimensional K M] [FiniteDimensional K N]
    (f : M →ₗ[K] P) (g : N →ₗ[K] Q) :
    finrank K (LinearMap.range (f.prodMap g)) = finrank K (LinearMap.range f) + finrank K (LinearMap.range g) := by
  rw [LinearMap.range_prodMap, (prodSubEquiv _ _).finrank_eq, finrank_prod]

lemma finrank_range_comp_fstK {M N P : Type} [AddCommGroup M] [Module K M] [AddCommGroup N] [Module K N]
    [AddCommGroup P] [Module K P] (f : M →ₗ[K] P) :
    LinearMap.range (f ∘ₗ LinearMap.fst K M N) = LinearMap.range f :=
  LinearMap.range_comp_of_range_eq_top _ (LinearMap.range_eq_top.mpr LinearMap.fst_surjective)

lemma finrank_range_comp_inj {M N P : Type} [AddCommGroup M] [Module K M] [AddCommGroup N] [Module K N]
    [AddCommGroup P] [Module K P] (f : M →ₗ[K] N) (E : N →ₗ[K] P) (hE : Function.Injective E) :
    finrank K (LinearMap.range (E ∘ₗ f)) = finrank K (LinearMap.range f) := by
  rw [LinearMap.range_comp]; exact (Submodule.equivMapOfInjective _ hE _).finrank_eq.symm

/-- **`rank (1 + T) + rank (1 − T) = dim`** when `T² = 1` and `2 ≠ 0`. -/
theorem finrank_range_add_of_sq_one {M : Type} [AddCommGroup M] [Module K M] [FiniteDimensional K M] (h2 : (2 : K) ≠ 0)
    (T : M →ₗ[K] M) (hT : T ∘ₗ T = 1) :
    finrank K (LinearMap.range (1 + T)) + finrank K (LinearMap.range (1 - T)) = finrank K M := by
  have hTT : ∀ x, T (T x) = x := fun x => by simpa using LinearMap.congr_fun hT x
  have hrk : LinearMap.range (1 - T) = LinearMap.ker (1 + T) := by
    apply le_antisymm
    · rintro _ ⟨x, rfl⟩
      simp [hTT]
    · intro x hx
      have hx' : x + T x = 0 := by simpa using hx
      refine ⟨(2 : K)⁻¹ • x, ?_⟩
      have : T x = -x := eq_neg_of_add_eq_zero_right hx'
      rw [LinearMap.sub_apply, Module.End.one_apply, map_smul, this, smul_neg, sub_neg_eq_add,
        ← two_smul K ((2 : K)⁻¹ • x), smul_smul, mul_inv_cancel₀ h2, one_smul]
  rw [hrk]; exact LinearMap.finrank_range_add_finrank_ker _

end LinAlg

/-! ### The tree complex over a field -/

section Tree

variable {K : Type} [Field K] (A : Rep.{0} K SL(2, ℤ))

/-- `σ = (1 − S, 1 − U) : A → A × A`. -/
def sigmaK : A →ₗ[K] A × A := (difA A sS).prod (difA A sU)

lemma treeD0_applyK (x y : A) : treeD0 A (x, y) = ((x - A.ρ sS x, y - A.ρ sU y), x - y) := by
  simp [treeD0, vtxA, phiA, difA]

lemma sigmaK_apply (x : A) : sigmaK A x = (x - A.ρ sS x, x - A.ρ sU x) := rfl

/-- `ker treeD0 ≃ ker σ`, `(x, x) ↤ x`. -/
def kerD0EquivK : LinearMap.ker (sigmaK A) ≃ₗ[K] LinearMap.ker (treeD0 A) where
  toFun x := ⟨(x.1, x.1), by
    have h := x.2
    rw [LinearMap.mem_ker, sigmaK_apply, Prod.mk_eq_zero] at h
    rw [LinearMap.mem_ker, treeD0_applyK, h.1, h.2, sub_self]; rfl⟩
  invFun v := ⟨v.1.1, by
    have h := v.2
    rw [LinearMap.mem_ker, show v.1 = (v.1.1, v.1.2) from rfl, treeD0_applyK, Prod.mk_eq_zero, Prod.mk_eq_zero] at h
    rw [LinearMap.mem_ker, sigmaK_apply, Prod.mk_eq_zero]
    refine ⟨h.1.1, ?_⟩
    rw [sub_eq_zero.mp h.2]; exact h.1.2⟩
  map_add' _ _ := rfl
  map_smul' _ _ := rfl
  left_inv _ := rfl
  right_inv v := by
    have h := v.2
    rw [LinearMap.mem_ker, show v.1 = (v.1.1, v.1.2) from rfl, treeD0_applyK, Prod.mk_eq_zero, Prod.mk_eq_zero] at h
    apply Subtype.ext
    show (v.1.1, v.1.1) = v.1
    exact Prod.ext rfl (sub_eq_zero.mp h.2)

lemma finrank_cohK (q : ℕ) : finrank K (groupCohomology A q) = finrank K ((treeCoch A).homology q) :=
  (model7TreeCochCoh A q).toLinearEquiv.finrank_eq

lemma finrank_range_tcDK (m : ℕ) : finrank K (LinearMap.range ((treeCoch A).d m (m + 1)).hom) =
    match m with
    | 0 => finrank K (LinearMap.range (treeD0 A))
    | m + 1 => finrank K (LinearMap.range (treeDS A m)) := by
  rw [treeCoch_d]
  cases m <;> rfl

variable [FiniteDimensional K A]

lemma finrank_range_treeD0K :
    finrank K (LinearMap.range (treeD0 A)) = finrank K A + finrank K (LinearMap.range (sigmaK A)) := by
  have h1 := LinearMap.finrank_range_add_finrank_ker (treeD0 A)
  have h2 := LinearMap.finrank_range_add_finrank_ker (sigmaK A)
  rw [← (kerD0EquivK A).finrank_eq, finrank_prod] at h1
  omega

include A in
lemma finiteDimensional_XK (n : ℕ) : FiniteDimensional K ((treeCoch A).X n) := by
  cases n
  · show FiniteDimensional K (A × A); infer_instance
  · show FiniteDimensional K ((A × A) × A); infer_instance

/-- `dim H⁰ + rank σ = dim A` (any `A`). -/
theorem finrank_H_zeroK :
    finrank K (groupCohomology A 0) + finrank K (LinearMap.range (sigmaK A)) = finrank K A := by
  have : FiniteDimensional K ((treeCoch A).sc' 0 0 1).X₂ := finiteDimensional_XK A 0
  rw [finrank_cohK, ((treeCoch A).homologyIsoSc' 0 0 1 (by simp) (by simp)).toLinearEquiv.finrank_eq]
  have h := GL2.L23.TreeF2.finrank_homology_add ((treeCoch A).sc' 0 0 1)
  have hf : finrank K (LinearMap.range ((treeCoch A).sc' 0 0 1).f.hom) = 0 := by
    rw [show ((treeCoch A).sc' 0 0 1).f = 0 from (treeCoch A).shape 0 0 (by simp)]; simp
  have hg : finrank K (LinearMap.range ((treeCoch A).sc' 0 0 1).g.hom) = finrank K (LinearMap.range (treeD0 A)) :=
    finrank_range_tcDK A 0
  have hX : finrank K ((treeCoch A).sc' 0 0 1).X₂ = finrank K (A × A) := rfl
  rw [hf, hg, hX, finrank_range_treeD0K, finrank_prod] at h
  omega

/-- `dim H^{q+1} + rank d^q + rank d^{q+1} = 3 dim A` (any `A`). -/
theorem finrank_H_succK (q : ℕ) :
    finrank K (groupCohomology A (q + 1)) + finrank K (LinearMap.range ((treeCoch A).d q (q + 1)).hom) +
      finrank K (LinearMap.range ((treeCoch A).d (q + 1) (q + 1 + 1)).hom) = 3 * finrank K A := by
  have : FiniteDimensional K ((treeCoch A).sc' q (q + 1) (q + 2)).X₂ := finiteDimensional_XK A (q + 1)
  rw [finrank_cohK, ((treeCoch A).homologyIsoSc' q (q + 1) (q + 2) (by simp) (by simp)).toLinearEquiv.finrank_eq]
  have h := GL2.L23.TreeF2.finrank_homology_add ((treeCoch A).sc' q (q + 1) (q + 2))
  have hX : finrank K ((treeCoch A).sc' q (q + 1) (q + 2)).X₂ = finrank K ((A × A) × A) := rfl
  rw [hX, finrank_prod, finrank_prod] at h
  have e : 3 * finrank K A = finrank K A + finrank K A + finrank K A := by ring
  rw [e]; exact h

end Tree

/-! ### The norms and edges for `ε = ±1` -/

section Eps

variable {K : Type} [Field K] {A : Rep.{0} K SL(2, ℤ)}

lemma rho_sS_two : A.ρ (sS ^ 2) = A.ρ (-1) := by rw [sS_sq]
lemma rho_sS_three : A.ρ (sS ^ 3) = A.ρ (-1) * A.ρ sS := by rw [pow_succ, map_mul, rho_sS_two]
lemma rho_sU_three : A.ρ (sU ^ 3) = A.ρ (-1) := by rw [sU_cube]
lemma rho_sU_four : A.ρ (sU ^ 4) = A.ρ (-1) * A.ρ sU := by rw [pow_succ, map_mul, rho_sU_three]
lemma rho_sU_five : A.ρ (sU ^ 5) = A.ρ (-1) * A.ρ (sU ^ 2) := by
  rw [show 5 = 3 + 2 from rfl, pow_add, map_mul, rho_sU_three]

lemma nrmA_sS_four_eq : nrmA A sS 4 = (1 + A.ρ (-1)) * nrmA A sS 2 := by
  simp only [nrmA, Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one, map_one]
  rw [rho_sS_two, rho_sS_three]; noncomm_ring

lemma nrmA_sU_six_eq : nrmA A sU 6 = (1 + A.ρ (-1)) * nrmA A sU 3 := by
  simp only [nrmA, Finset.sum_range_succ, Finset.sum_range_zero, zero_add, pow_zero, pow_one, map_one]
  rw [rho_sU_three, rho_sU_four, rho_sU_five]; noncomm_ring

lemma edgA_even (m : ℕ) (hm : Even m) : edgA A m = 1 - A.ρ (-1) := by
  rw [edgA, if_pos hm]; rfl

lemma edgA_odd (m : ℕ) (hm : ¬ Even m) : edgA A m = 1 + A.ρ (-1) := by
  rw [edgA, if_neg hm]
  simp [nrmA, Finset.sum_range_succ]

lemma vtxA_even' (m : ℕ) (hm : Even m) : vtxA A m = (difA A sS).prodMap (difA A sU) := by rw [vtxA, if_pos hm]
lemma vtxA_odd' (m : ℕ) (hm : ¬ Even m) : vtxA A m = (nrmA A sS 4).prodMap (nrmA A sU 6) := by rw [vtxA, if_neg hm]
lemma phiA_even' (m : ℕ) (hm : Even m) : phiA A m = LinearMap.fst K A A - LinearMap.snd K A A := by rw [phiA, if_pos hm]
lemma phiA_odd' (m : ℕ) (hm : ¬ Even m) :
    phiA A m = nrmA A sS 2 ∘ₗ LinearMap.fst K A A - nrmA A sU 3 ∘ₗ LinearMap.snd K A A := by rw [phiA, if_neg hm]

lemma treeDS_apply (m : ℕ) (x y z : A) :
    treeDS A m ((x, y), z) = (vtxA A (m + 1) (x, y), phiA A (m + 1) (x, y) - edgA A m z) := rfl

end Eps

/-! ### `−I` acts by `−1`: every `H^q` vanishes -/

section NegOne

variable {K : Type} [Field K] (A : Rep.{0} K SL(2, ℤ)) [FiniteDimensional K A]

lemma sigmaK_injective (h2 : (2 : K) ≠ 0) (hN : A.ρ (-1) = -1) : Function.Injective (sigmaK A) := by
  rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
  intro x hx
  rw [sigmaK_apply, Prod.mk_eq_zero, sub_eq_zero, sub_eq_zero] at hx
  have h1 : A.ρ (sS ^ 2) x = x := by rw [pow_two, map_mul, Module.End.mul_apply, ← hx.1, ← hx.1]
  rw [rho_sS_two, hN] at h1
  have : (2 : K) • x = 0 := by
    rw [two_smul]; nth_rewrite 1 [← h1]; simp
  exact (smul_eq_zero.mp this).resolve_left h2

/-- **`−I` acting by `−1` kills `H^q` in every degree** (`2 ≠ 0`). -/
theorem finrank_H_eq_zero_of_negOne (h2 : (2 : K) ≠ 0) (hN : A.ρ (-1) = -1) (q : ℕ) :
    finrank K (groupCohomology A q) = 0 := by
  have hs : finrank K (LinearMap.range (sigmaK A)) = finrank K A := LinearMap.finrank_range_of_inj (sigmaK_injective A h2 hN)
  have hD0 : finrank K (LinearMap.range (treeD0 A)) = 2 * finrank K A := by rw [finrank_range_treeD0K, hs]; ring
  have hε : 1 + A.ρ (-1) = 0 := by rw [hN]; abel
  -- even `m`: `((x, y), z) ↦ ((0, 0), φ(x, y) − 2z)`, rank `dim A`
  have hE : ∀ m, Even m → finrank K (LinearMap.range (treeDS A m)) = finrank K A := by
    intro m hm
    have hm1 : ¬ Even (m + 1) := by rw [Nat.even_add_one]; exact not_not.mpr hm
    let g : (A × A) × A →ₗ[K] A := phiA A (m + 1) ∘ₗ LinearMap.fst K (A × A) A - edgA A m ∘ₗ LinearMap.snd K (A × A) A
    have hT : treeDS A m = LinearMap.inr K (A × A) A ∘ₗ g := by
      refine LinearMap.ext fun v => ?_
      obtain ⟨⟨x, y⟩, z⟩ := v
      rw [treeDS_apply, vtxA_odd' _ hm1, nrmA_sS_four_eq, nrmA_sU_six_eq, hε]
      simp [g]
    have hg : Function.Surjective g := by
      intro w
      refine ⟨((0, 0), -((2 : K)⁻¹ • w)), ?_⟩
      have e : g ((0, 0), -((2 : K)⁻¹ • w)) = phiA A (m + 1) 0 - edgA A m (-((2 : K)⁻¹ • w)) := rfl
      rw [e, map_zero, zero_sub, edgA_even _ hm, hN, map_neg, neg_neg]
      simp only [LinearMap.sub_apply, LinearMap.add_apply, LinearMap.neg_apply, Module.End.one_apply, sub_neg_eq_add]
      rw [← two_smul K ((2 : K)⁻¹ • w), smul_smul, mul_inv_cancel₀ h2, one_smul]
    rw [hT, finrank_range_comp_inj _ _ LinearMap.inr_injective, LinearMap.range_eq_top.mpr hg, finrank_top]
  -- odd `m`: `treeD0 ∘ fst`, rank `2 dim A`
  have hO : ∀ m, ¬ Even m → finrank K (LinearMap.range (treeDS A m)) = 2 * finrank K A := by
    intro m hm
    have hm1 : Even (m + 1) := by rw [Nat.even_add_one]; exact hm
    have hT : treeDS A m = treeD0 A ∘ₗ LinearMap.fst K (A × A) A := by
      refine LinearMap.ext fun v => ?_
      obtain ⟨⟨x, y⟩, z⟩ := v
      rw [treeDS_apply, vtxA_even' _ hm1, phiA_even' _ hm1, edgA_odd _ hm, hε]
      simp [treeD0, vtxA, phiA]
    rw [hT, finrank_range_comp_fstK, hD0]
  cases q with
  | zero => have := finrank_H_zeroK A; rw [hs] at this; omega
  | succ q =>
    have h := finrank_H_succK A q
    rw [finrank_range_tcDK, finrank_range_tcDK] at h
    cases q with
    | zero => simp only at h; rw [hD0, hE 0 (by decide)] at h; omega
    | succ q =>
      simp only at h
      by_cases hq : Even q
      · rw [hE q hq, hO (q + 1) (by rw [Nat.even_add_one]; exact not_not.mpr hq)] at h; omega
      · rw [hO q hq, hE (q + 1) (by rw [Nat.even_add_one]; exact hq)] at h; omega

end NegOne

/-! ### `−I` acts trivially: three ranks per degree -/

section Triv

variable {K : Type} [Field K] (A : Rep.{0} K SL(2, ℤ)) [FiniteDimensional K A]

/-- **`dim H^q` for `−I` trivial, `2 ≠ 0`**: `H⁰`, `H¹` and `H^{q+2}` from `rank σ`, `rank (1 + S)`, `rank (1 + U + U²)`,
`rank (1 − U)`. -/
theorem finrank_H_of_trivNegOne (h2 : (2 : K) ≠ 0) (hI : A.ρ (-1) = 1) :
    finrank K (groupCohomology A 0) + finrank K (LinearMap.range (sigmaK A)) = finrank K A ∧
    finrank K (groupCohomology A 1) + finrank K (LinearMap.range (sigmaK A)) +
        finrank K (LinearMap.range (nrmA A sS 2)) + finrank K (LinearMap.range (nrmA A sU 3)) = 2 * finrank K A ∧
    ∀ q, finrank K (groupCohomology A (q + 2)) + finrank K (LinearMap.range (nrmA A sU 3)) +
        finrank K (LinearMap.range (difA A sU)) = finrank K A := by
  have hε0 : 1 - A.ρ (-1) = 0 := by rw [hI]; abel
  have hε2 : 1 + A.ρ (-1) = (2 : K) • (1 : A →ₗ[K] A) := by rw [hI, two_smul]
  have h2u : IsUnit (2 : K) := isUnit_iff_ne_zero.mpr h2
  -- even `m`: `((x, y), z) ↦ ((2(1+S)x, 2N₃y), (1+S)x − N₃y)`, rank `r_S⁺ + r₃`
  let E : A × A →ₗ[K] (A × A) × A :=
    ((2 : K) • (LinearMap.id : A × A →ₗ[K] A × A)).prod (LinearMap.fst K A A - LinearMap.snd K A A)
  have hEi : Function.Injective E := by
    rw [← LinearMap.ker_eq_bot, LinearMap.ker_eq_bot']
    intro v hv
    have := congrArg Prod.fst hv
    simp only [E, LinearMap.prod_apply, LinearMap.smul_apply, LinearMap.id_apply, Prod.fst_zero] at this
    exact (smul_eq_zero.mp this).resolve_left h2
  have hE : ∀ m, Even m → finrank K (LinearMap.range (treeDS A m)) =
      finrank K (LinearMap.range (nrmA A sS 2)) + finrank K (LinearMap.range (nrmA A sU 3)) := by
    intro m hm
    have hm1 : ¬ Even (m + 1) := by rw [Nat.even_add_one]; exact not_not.mpr hm
    have hT : treeDS A m = E ∘ₗ ((nrmA A sS 2).prodMap (nrmA A sU 3)) ∘ₗ LinearMap.fst K (A × A) A := by
      refine LinearMap.ext fun v => ?_
      obtain ⟨⟨x, y⟩, z⟩ := v
      rw [treeDS_apply, vtxA_odd' _ hm1, phiA_odd' _ hm1, edgA_even _ hm, hε0, nrmA_sS_four_eq, nrmA_sU_six_eq, hε2]
      simp [E]
    rw [hT, finrank_range_comp_inj _ _ hEi, finrank_range_comp_fstK, finrank_range_prodMap]
  -- odd `m`: `((x, y), z) ↦ (((1−S)x, (1−U)y), x − y − 2z)`, rank `r_S⁻ + r_U⁻ + dim A`
  have hO : ∀ m, ¬ Even m → finrank K (LinearMap.range (treeDS A m)) =
      finrank K (LinearMap.range (difA A sS)) + finrank K (LinearMap.range (difA A sU)) + finrank K A := by
    intro m hm
    have hm1 : Even (m + 1) := by rw [Nat.even_add_one]; exact hm
    have hR : LinearMap.range (treeDS A m) = (LinearMap.range ((difA A sS).prodMap (difA A sU))).prod ⊤ := by
      apply le_antisymm
      · rintro _ ⟨⟨⟨x, y⟩, z⟩, rfl⟩
        rw [treeDS_apply, vtxA_even' _ hm1]
        exact Submodule.mem_prod.mpr ⟨⟨(x, y), rfl⟩, Submodule.mem_top⟩
      · rintro ⟨v, w⟩ hv
        obtain ⟨⟨x, y⟩, hxy⟩ := (Submodule.mem_prod.mp hv).1
        refine ⟨((x, y), (2 : K)⁻¹ • (x - y - w)), ?_⟩
        rw [treeDS_apply, vtxA_even' _ hm1, hxy, phiA_even' _ hm1, edgA_odd _ hm, hε2]
        simp only [LinearMap.sub_apply, LinearMap.fst_apply, LinearMap.snd_apply, LinearMap.smul_apply,
          Module.End.one_apply, smul_smul, mul_inv_cancel₀ h2, one_smul, sub_sub_cancel]
    rw [hR, (prodSubEquiv _ _).finrank_eq, finrank_prod, finrank_top, LinearMap.range_prodMap,
      (prodSubEquiv _ _).finrank_eq, finrank_prod]
  -- `r_S⁺ + r_S⁻ = dim A`
  have hS2 : A.ρ sS ∘ₗ A.ρ sS = 1 := by
    rw [← Module.End.mul_eq_comp, ← map_mul, ← pow_two, rho_sS_two, hI]
  have hpm := finrank_range_add_of_sq_one h2 (A.ρ sS) hS2
  have hP : nrmA A sS 2 = 1 + A.ρ sS := by simp [nrmA, Finset.sum_range_succ]
  have hM : difA A sS = 1 - A.ρ sS := rfl
  rw [← hP, ← hM] at hpm
  have hD0 := finrank_range_treeD0K A
  refine ⟨finrank_H_zeroK A, ?_, fun q => ?_⟩
  · have h := finrank_H_succK A 0
    rw [finrank_range_tcDK, finrank_range_tcDK] at h
    simp only at h
    change finrank K (groupCohomology A 1) + _ + _ = _ at h
    rw [hD0, hE 0 (by decide)] at h; omega
  · have h := finrank_H_succK A (q + 1)
    rw [finrank_range_tcDK, finrank_range_tcDK] at h
    simp only at h
    change finrank K (groupCohomology A (q + 2)) + _ + _ = _ at h
    by_cases hq : Even q
    · rw [hE q hq, hO (q + 1) (by rw [Nat.even_add_one]; exact not_not.mpr hq)] at h; omega
    · rw [hO q hq, hE (q + 1) (by rw [Nat.even_add_one]; exact hq)] at h; omega

end Triv

end GL2.L23.TreeF3
