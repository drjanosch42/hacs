/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.OddExponent
import GL2.L23.TreeF2

/-!
# LG-2 (slot 23): Theorem O(2)–(3) as far as it is general — `H⁰ = 0` integrally, and the mod-2 dimensions for odd `n`

`L23_master_plan.md` v0.4 §1 Theorem O(2)–(3), §4.3 (row LG-2); [C] `prop:oddnltwo` (1)–(3).
[GT] `gt_part_MP.tex`, rows `thm:lg-h0-zero`, `thm:lg-odd-f2-general`, `thm:lg-odd-f2-dims`.

* **`H⁰(Γ; X) = 0`** if `−I` acts by `−1` and `X` has no `2`-torsion (`H0_eq_zero_of_negOne`); hence
  `H⁰(SL₂(ℤ); Sym^n k²) = 0` for odd `n` over any domain with `2 ≠ 0`, `ℤ` included (`sym_odd_H0_eq_zero`).
* **Characteristic 2, `−I` trivial, general** (`finrank_kerU_eq`, `finrank_H_succ_of_free`): if `M^S = (1 + S)M`
  (`M` free over `F₂[⟨S⟩]`) and `S` normalises `⟨U⟩` on `M` (`S U S = U²`), then
  `c_U := dim M^U = 2 · dim H⁰` and `dim H^{q+1} + dim (1 + S)M = dim M` for every `q`.  The proof is [C]
  `prop:oddnltwo`(2)'s argument made linear: `π_U = 1 + U + U²` projects onto `M^U` (`range_piU`), commutes with
  `N = 1 + S`, so `N` maps `W = M^U` into `W ∩ ker N`, onto it (`M^S = N M`, `π_U` fixes `W`), and rank–nullity on `N|_W`
  gives `dim W = 2 dim (W ∩ ker N) = 2 dim M^Γ`.  No Dickson theorem and no eigenvalue count over `F₄` is used.
* **`Sym^n F₂²`, `n` odd** (`V2_S_apply`, `V2_free_S`, `V2_conj`, `V2_odd_dims`, `V2_odd_cU`):
  `ρ_n(S) mod 2` reverses the monomials (`x^{n−t}y^t ↦ x^t y^{n−t}`), which is fixed-point free for odd `n`; so for every odd
  `n` and every `q`, **`dim_{F₂} H^{q+1}(SL₂(ℤ); Sym^n F₂²) = (n + 1)/2`** and **`dim M^U = 2 h⁰`**, `h⁰ = dim H⁰`.

**Not here** (gaps of LG-2, `L23_master_plan.md` §4.3): the closed form `h⁰ = ⌊(⌊n/3⌋ + 1)/2⌋` for general `n` (it needs the
eigenvalue count of `U` over `F₄` or Dickson's theorem for `GL₂(F₂)`; per cell: `GL2.L23.OddDimsCells`); the integral groups
`H^q(Γ; Sym^n ℤ²) ≅ F₂^{h⁰}, F₂^{m}` (they need the Bockstein long exact sequence of `0 → V → V → V/2 → 0` in
`groupCohomology` and the change of rings `ℤ → F₂`); the `H⁰` Hecke formulas (`T_ℓ² = 0`, `T₂ m = π(m) d_n`).
-/

noncomputable section

namespace GL2.L23.OddDims

open CategoryTheory GL2.MP GL2.L23.OddExp GL2.L23.TreeF2 Module groupCohomology
open scoped MatrixGroups

/-! ### `H⁰ = 0` when `−I` acts by `−1` on a module without `2`-torsion -/

section H0

variable {k : Type} [CommRing k]

/-- **`H⁰(SL₂(ℤ); A) = 0`** if `−I` acts by `−1` and `A` has no `2`-torsion. -/
theorem H0_eq_zero_of_negOne {A : Rep.{0} k SL(2, ℤ)} (hI : NegOneActs A) (h2 : ∀ a : A, a + a = 0 → a = 0)
    (x : groupCohomology A 0) : x = 0 := by
  apply (H0Iso A).toLinearEquiv.injective
  rw [map_zero]
  set v := (H0Iso A).toLinearEquiv x
  have hv : A.ρ (-1) (v : A) = v := v.2 (-1)
  rw [hI] at hv
  apply Subtype.ext
  show (v : A) = 0
  apply h2
  nth_rewrite 1 [← hv]
  exact neg_add_cancel _

/-- `Sym^n k²` has no `2`-torsion over a domain with `2 ≠ 0`. -/
lemma symN_two_torsion [IsDomain k] (h2 : (2 : k) ≠ 0) (n : ℕ) (a : GL2.SymN k n) (ha : a + a = 0) : a = 0 := by
  apply Subtype.ext
  show (a : MvPolynomial (Fin 2) k) = 0
  ext m
  have h := congrArg (fun p : GL2.SymN k n => MvPolynomial.coeff m (p : MvPolynomial (Fin 2) k)) ha
  simp only [Submodule.coe_add, MvPolynomial.coeff_add, Submodule.coe_zero, MvPolynomial.coeff_zero] at h
  rw [← two_mul] at h
  rw [MvPolynomial.coeff_zero]
  exact (mul_eq_zero.mp h).resolve_left h2

/-- **`H⁰(SL₂(ℤ); Sym^n k²) = 0` for odd `n`**, over any domain with `2 ≠ 0` (e.g. `ℤ`, `ℤ₂`, `ℚ`). -/
theorem sym_odd_H0_eq_zero [IsDomain k] (h2 : (2 : k) ≠ 0) {n : ℕ} (hn : Odd n) (x : groupCohomology (VK k n) 0) :
    x = 0 :=
  H0_eq_zero_of_negOne (VK_negOneActs hn) (symN_two_torsion h2 n) x

end H0

/-! ### Characteristic `2`, `−I` trivial: `c_U = 2 h⁰` and `dim H^{q+1} = dim M − dim (1 + S)M` -/

section Char2

variable {A : Rep.{0} (ZMod 2) SL(2, ℤ)}

lemma add_self_zero {M : Type} [AddCommGroup M] [Module (ZMod 2) M] (x : M) : x + x = 0 := by
  rw [← two_smul (ZMod 2) x, show (2 : ZMod 2) = 0 from rfl, zero_smul]

/-- In characteristic `2`, `N_S = 1 + S` is `1 − S`. -/
lemma nrm_eq_dif : nrmA A sS 2 = difA A sS := by
  refine LinearMap.ext fun x => ?_
  simp [nrmA_apply, Finset.sum_range_succ, difA_apply, sub_eq_add_zmod2]

lemma rS_rS (hI : TrivNegOne A) (x : A) : A.ρ sS (A.ρ sS x) = x := by
  have h := congrArg (fun f => f x) (rho_sS_sq hI)
  simpa [pow_two, Module.End.mul_apply] using h

lemma rU_rU_rU (hI : TrivNegOne A) (x : A) : A.ρ sU (A.ρ sU (A.ρ sU x)) = x := by
  have h := congrArg (fun f => f x) (rho_sU_cube hI)
  simpa [pow_succ, Module.End.mul_apply] using h

lemma nrmS_apply (x : A) : nrmA A sS 2 x = x + A.ρ sS x := by
  simp [nrmA_apply, Finset.sum_range_succ]

lemma nrmU_apply (x : A) : nrmA A sU 3 x = x + A.ρ sU x + A.ρ sU (A.ρ sU x) := by
  simp [nrmA_apply, Finset.sum_range_succ, pow_two, Module.End.mul_apply]

/-- `N_S² = 0`. -/
lemma nrmS_nrmS (hI : TrivNegOne A) (x : A) : nrmA A sS 2 (nrmA A sS 2 x) = 0 := by
  rw [nrmS_apply, nrmS_apply, map_add, rS_rS hI]
  rw [show x + A.ρ sS x + (A.ρ sS x + x) = (x + x) + (A.ρ sS x + A.ρ sS x) by abel, add_self_zero, add_self_zero,
    add_zero]

/-- **`π_U = 1 + U + U²` projects onto `M^U`**: `range π_U = ker (1 − U)`. -/
theorem range_piU (hI : TrivNegOne A) : LinearMap.range (nrmA A sU 3) = LinearMap.ker (difA A sU) := by
  apply le_antisymm
  · rintro _ ⟨x, rfl⟩
    rw [LinearMap.mem_ker, difA_apply, nrmU_apply, map_add, map_add, rU_rU_rU hI, sub_eq_zero]
    abel
  · intro x hx
    rw [LinearMap.mem_ker, difA_apply, sub_eq_zero] at hx
    refine ⟨x, ?_⟩
    rw [nrmU_apply, ← hx, ← hx, add_self_zero, zero_add]

/-- `range ψ = range N_S ⊔ range π_U`. -/
theorem range_psiA : LinearMap.range (psiA (A := A)) =
    LinearMap.range (nrmA A sS 2) ⊔ LinearMap.range (nrmA A sU 3) := by
  have hψ : ∀ x y : A, psiA (A := A) (x, y) = nrmA A sS 2 x - nrmA A sU 3 y := fun x y => by
    simp [psiA, phiA]
  apply le_antisymm
  · rintro _ ⟨⟨x, y⟩, rfl⟩
    rw [hψ]
    exact Submodule.sub_mem _ (Submodule.mem_sup_left ⟨x, rfl⟩) (Submodule.mem_sup_right ⟨y, rfl⟩)
  · refine sup_le ?_ ?_
    · rintro _ ⟨x, rfl⟩; exact ⟨(x, 0), by rw [hψ, map_zero, sub_zero]⟩
    · rintro _ ⟨y, rfl⟩; exact ⟨(0, -y), by rw [hψ, map_zero, map_neg, zero_sub, neg_neg]⟩

/-- `ker σ = ker (1 − S) ⊓ ker (1 − U)`. -/
lemma ker_sigmaA : LinearMap.ker (sigmaA (A := A)) = LinearMap.ker (difA A sS) ⊓ LinearMap.ker (difA A sU) :=
  LinearMap.ker_prod _ _

/-- The commutation of `N_S` and `π_U`, from `S U S = U²` (`S` normalises `⟨U⟩`). -/
lemma nrm_comm (hI : TrivNegOne A) (hC : ∀ x : A, A.ρ sS (A.ρ sU (A.ρ sS x)) = A.ρ sU (A.ρ sU x)) (x : A) :
    nrmA A sU 3 (nrmA A sS 2 x) = nrmA A sS 2 (nrmA A sU 3 x) := by
  -- `S U = U² S` (hence `S U² = U⁴ S = U S`)
  have hSU : ∀ y : A, A.ρ sS (A.ρ sU y) = A.ρ sU (A.ρ sU (A.ρ sS y)) := fun y => by
    rw [← hC, rS_rS hI]
  rw [nrmS_apply, nrmU_apply, nrmU_apply, nrmS_apply]
  simp only [map_add, hSU, rU_rU_rU hI]
  abel

/-- `π_U` fixes `M^U`. -/
lemma piU_fix {y : A} (hy : y ∈ LinearMap.ker (difA A sU)) : nrmA A sU 3 y = y := by
  rw [LinearMap.mem_ker, difA_apply, sub_eq_zero] at hy
  rw [nrmU_apply, ← hy, ← hy, add_self_zero, zero_add]

variable [FiniteDimensional (ZMod 2) A]

/-- **`c_U = 2 h⁰`**: `dim M^U = 2 dim H⁰(Γ; M)`, if `M^S = (1 + S)M` and `S U S = U²` on `M` ([C] `prop:oddnltwo`(2)). -/
theorem finrank_kerU_eq (hI : TrivNegOne A)
    (hS : LinearMap.range (nrmA A sS 2) = LinearMap.ker (difA A sS))
    (hC : ∀ x : A, A.ρ sS (A.ρ sU (A.ρ sS x)) = A.ρ sU (A.ρ sU x)) :
    finrank (ZMod 2) (LinearMap.ker (difA A sU)) = 2 * finrank (ZMod 2) (groupCohomology A 0) := by
  -- `N(W) = W ⊓ ker N`, `W = M^U`, `N = 1 + S`
  have hNW : Submodule.map (nrmA A sS 2) (LinearMap.ker (difA A sU)) =
      LinearMap.ker (difA A sU) ⊓ LinearMap.ker (nrmA A sS 2) := by
    apply le_antisymm
    · rintro _ ⟨w, hw, rfl⟩
      have hw' : w ∈ LinearMap.ker (difA A sU) := hw
      refine Submodule.mem_inf.mpr ⟨?_, nrmS_nrmS hI w⟩
      rw [← piU_fix hw', ← nrm_comm hI hC, ← range_piU hI]
      exact ⟨nrmA A sS 2 w, rfl⟩
    · intro w hw
      obtain ⟨hw', hk'⟩ := Submodule.mem_inf.mp hw
      have hk2 : w ∈ LinearMap.ker (difA A sS) := by rw [← nrm_eq_dif]; exact hk'
      rw [← hS] at hk2
      obtain ⟨v, rfl⟩ := hk2
      refine ⟨nrmA A sU 3 v, ?_, ?_⟩
      · show nrmA A sU 3 v ∈ LinearMap.ker (difA A sU)
        rw [← range_piU hI]; exact ⟨v, rfl⟩
      · show nrmA A sS 2 (nrmA A sU 3 v) = nrmA A sS 2 v
        rw [← nrm_comm hI hC, piU_fix hw']
  -- rank–nullity for `N ∘ W.subtype`
  have hrn := LinearMap.finrank_range_add_finrank_ker (nrmA A sS 2 ∘ₗ (LinearMap.ker (difA A sU)).subtype)
  have hr : finrank (ZMod 2) (LinearMap.range (nrmA A sS 2 ∘ₗ (LinearMap.ker (difA A sU)).subtype)) =
      finrank (ZMod 2) ↥(LinearMap.ker (difA A sU) ⊓ LinearMap.ker (nrmA A sS 2)) := by
    rw [LinearMap.range_comp, Submodule.range_subtype, hNW]
  have hk : finrank (ZMod 2) (LinearMap.ker (nrmA A sS 2 ∘ₗ (LinearMap.ker (difA A sU)).subtype)) =
      finrank (ZMod 2) ↥(LinearMap.ker (difA A sU) ⊓ LinearMap.ker (nrmA A sS 2)) := by
    rw [LinearMap.ker_comp, (Submodule.equivMapOfInjective _ (Submodule.injective_subtype _) _).finrank_eq,
      Submodule.map_comap_subtype]
  -- `W ⊓ ker N = ker σ`, of dimension `h⁰`
  have hinf : LinearMap.ker (difA A sU) ⊓ LinearMap.ker (nrmA A sS 2) = LinearMap.ker (sigmaA (A := A)) := by
    rw [ker_sigmaA, nrm_eq_dif, inf_comm]
  have h0 := finrank_H_zero (A := A)
  have hσ := LinearMap.finrank_range_add_finrank_ker (sigmaA (A := A))
  rw [hinf] at hr hk
  omega

/-- **`dim H^{q+1} + dim (1 + S)M = dim M`**, every `q`, if `M^S = (1 + S)M` and `S U S = U²` on `M`. -/
theorem finrank_H_succ_of_free (hI : TrivNegOne A)
    (hS : LinearMap.range (nrmA A sS 2) = LinearMap.ker (difA A sS))
    (hC : ∀ x : A, A.ρ sS (A.ρ sU (A.ρ sS x)) = A.ρ sU (A.ρ sU x)) (q : ℕ) :
    finrank (ZMod 2) (groupCohomology A (q + 1)) + finrank (ZMod 2) (LinearMap.range (nrmA A sS 2)) =
      finrank (ZMod 2) A := by
  have hH := finrank_H_succ hI q
  have h0 := finrank_H_zero (A := A)
  have hc := finrank_kerU_eq hI hS hC
  have hsup := Submodule.finrank_sup_add_finrank_inf_eq (LinearMap.range (nrmA A sS 2))
    (LinearMap.range (nrmA A sU 3))
  rw [← range_psiA, range_piU hI, hS, ← ker_sigmaA, ← hS] at hsup
  have hσ := LinearMap.finrank_range_add_finrank_ker (sigmaA (A := A))
  omega

end Char2

/-! ### `Sym^n F₂²` for odd `n` -/

section V2

open GL2.L23.RankF2

/-- `ρ_n(S) mod 2` reverses the coordinates: `(S v)_t = v_{n−t}`. -/
lemma V2_S_apply (n : ℕ) (v : V2 n) (t : Fin (n + 1)) : ((V2 n).ρ sS v : Fin (n + 1) → ZMod 2) t = v (Fin.rev t) := by
  rw [V2_rho, mod2_sS, Matrix.toLin'_apply, Matrix.mulVec, dotProduct]
  have hA : mS2 = !![((0 : ℕ) : ZMod 2), ((1 : ℕ) : ZMod 2); ((1 : ℕ) : ZMod 2), ((0 : ℕ) : ZMod 2)] := by
    ext i j; fin_cases i <;> fin_cases j <;> rfl
  have he : ∀ j : Fin (n + 1), GL2.rhoMat n mS2 t j * v j = if j = Fin.rev t then v j else 0 := fun j => by
    rw [rhoMat_entry n mS2 0 1 1 0 hA, rhoEntry_S n j t (Nat.lt_succ_iff.mp j.2)]
    by_cases h : j = Fin.rev t
    · have : (t : ℕ) = n - j := by rw [h, Fin.val_rev]; omega
      rw [if_pos this, if_pos h]; simp
    · have : ¬ ((t : ℕ) = n - j) := fun h' => h (Fin.ext (by rw [Fin.val_rev]; have := j.2; omega))
      rw [if_neg this, if_neg h]; simp
  rw [Finset.sum_congr rfl (fun j _ => he j), Finset.sum_ite_eq' Finset.univ (Fin.rev t)]
  simp

/-- **`M^S = (1 + S)M`** on `Sym^n F₂²` for odd `n` (the monomials come in `(n + 1)/2` free `S`-orbits). -/
theorem V2_free_S {n : ℕ} (hn : Odd n) :
    LinearMap.range (nrmA (V2 n) sS 2) = LinearMap.ker (difA (V2 n) sS) := by
  apply le_antisymm
  · rintro _ ⟨x, rfl⟩
    rw [LinearMap.mem_ker, ← nrm_eq_dif]
    exact nrmS_nrmS (V2_trivNegOne n) x
  · intro x hx
    rw [LinearMap.mem_ker, difA_apply, sub_eq_zero] at hx
    have hsym : ∀ t, (x : Fin (n + 1) → ZMod 2) (Fin.rev t) = x t := fun t => by
      have := congrFun (congrArg (fun y : V2 n => (y : Fin (n + 1) → ZMod 2)) hx) t
      rw [V2_S_apply] at this
      exact this.symm
    obtain ⟨m, rfl⟩ := hn
    let w : V2 (2 * m + 1) := fun t => if (t : ℕ) ≤ m then x t else 0
    refine ⟨w, ?_⟩
    funext t
    show (nrmA (V2 (2 * m + 1)) sS 2 w : Fin (2 * m + 1 + 1) → ZMod 2) t = x t
    rw [nrmS_apply]
    show w t + ((V2 (2 * m + 1)).ρ sS w : Fin (2 * m + 1 + 1) → ZMod 2) t = x t
    rw [V2_S_apply]
    have hr : ((Fin.rev t : Fin (2 * m + 1 + 1)) : ℕ) = 2 * m + 1 - t := by rw [Fin.val_rev]; omega
    by_cases ht : (t : ℕ) ≤ m
    · have : ¬ (((Fin.rev t : Fin (2 * m + 1 + 1)) : ℕ) ≤ m) := by rw [hr]; omega
      show (if (t : ℕ) ≤ m then x t else 0) + (if ((Fin.rev t : Fin _) : ℕ) ≤ m then x (Fin.rev t) else 0) = x t
      rw [if_pos ht, if_neg this, add_zero]
    · have : ((Fin.rev t : Fin (2 * m + 1 + 1)) : ℕ) ≤ m := by rw [hr]; have := t.2; omega
      show (if (t : ℕ) ≤ m then x t else 0) + (if ((Fin.rev t : Fin _) : ℕ) ≤ m then x (Fin.rev t) else 0) = x t
      rw [if_neg ht, if_pos this, zero_add, hsym]

lemma mS2_mU2_mS2 : mS2 * mU2 * mS2 = mUU2 := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [mS2, mU2, mUU2, Matrix.mul_apply, Fin.sum_univ_two]

/-- **`S U S = U²`** on `Sym^n F₂²` (in `SL₂(F₂) ≅ S₃`, `⟨Ū⟩ = A₃` is normal). -/
theorem V2_conj (n : ℕ) (x : V2 n) : (V2 n).ρ sS ((V2 n).ρ sU ((V2 n).ρ sS x)) = (V2 n).ρ sU ((V2 n).ρ sU x) := by
  have h : (V2 n).ρ (sS * sU * sS) = (V2 n).ρ (sU ^ 2) := by
    rw [V2_rho, V2_rho, mod2_mul, mod2_mul, mod2_sS, mod2_sU, mod2_sU_sq, mS2_mU2_mS2]
  have hx := congrArg (fun f => f x) h
  simpa [map_mul, pow_two, Module.End.mul_apply] using hx

/-- `dim (1 + S)M = (n + 1)/2` for odd `n`. -/
theorem V2_finrank_rangeS {n : ℕ} (hn : Odd n) :
    2 * finrank (ZMod 2) (LinearMap.range (nrmA (V2 n) sS 2)) = n + 1 := by
  have hrn := LinearMap.finrank_range_add_finrank_ker (nrmA (V2 n) sS 2)
  have hk : LinearMap.ker (nrmA (V2 n) sS 2) = LinearMap.range (nrmA (V2 n) sS 2) := by
    rw [V2_free_S hn, nrm_eq_dif]
  rw [hk, finrank_V2] at hrn
  omega

/-- **Theorem O(3) mod 2**: `dim_{F₂} H^{q+1}(SL₂(ℤ); Sym^n F₂²) = (n + 1)/2` for every odd `n` and every `q`. -/
theorem V2_odd_dims {n : ℕ} (hn : Odd n) (q : ℕ) :
    finrank (ZMod 2) (groupCohomology (V2 n) (q + 1)) = (n + 1) / 2 := by
  have h := finrank_H_succ_of_free (V2_trivNegOne n) (V2_free_S hn) (V2_conj n) q
  have h2 := V2_finrank_rangeS hn
  rw [finrank_V2] at h
  omega

/-- **`c_U = 2 h⁰`** on `Sym^n F₂²` for every odd `n`. -/
theorem V2_odd_cU {n : ℕ} (hn : Odd n) :
    finrank (ZMod 2) (LinearMap.ker (difA (V2 n) sU)) = 2 * finrank (ZMod 2) (groupCohomology (V2 n) 0) :=
  finrank_kerU_eq (V2_trivNegOne n) (V2_free_S hn) (V2_conj n)

end V2

end GL2.L23.OddDims
