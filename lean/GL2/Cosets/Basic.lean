import Mathlib
import GL2.Conventions.SymPow
import GL2.Cosets.RowClass

/-!
# G1 (L17-3): the correspondence datum ([C] `prop:cosets`, `def:cosetreps`, `lem:cocycle`)

`Γ = SL(2, ℤ)`, `g = diag(1, N)` with `N = p^ν`, `p` prime, `ν ≥ 1`.  Matrices act on column vectors
from the left.  The canonical transporters `s_k` of [C] `def:cosetreps` are indexed 0-based here
(`k = j - 1`):
* `s_k = [[1, 0], [k, 1]]` for `0 ≤ k < N` (class `(k : 1)`),
* `s_{N + c} = [[0, -1], [1, p c]]` for `0 ≤ c < p^(ν-1)` (class `(1 : p c)`).

`Γ'' = Γ₀(N)` is mathlib's `CongruenceSubgroup.Gamma0`; `Γ' = Γ⁰(N)` is `Gamma0Up` below.
-/

namespace GL2.Cosets

open Matrix CongruenceSubgroup
open scoped MatrixGroups

/-- A `2 × 2` integer matrix of determinant one, as an element of `SL(2, ℤ)`. -/
def mkSL (a b c d : ℤ) (h : a * d - b * c = 1) : SL(2, ℤ) :=
  ⟨!![a, b; c, d], by rw [Matrix.det_fin_two_of]; exact h⟩

@[simp] lemma mkSL_coe (a b c d : ℤ) (h : a * d - b * c = 1) :
    ((mkSL a b c d h : SL(2, ℤ)) : Mat2 ℤ) = !![a, b; c, d] := rfl

/-- `g = diag(1, N)`. -/
def gMat (N : ℤ) : Mat2 ℤ := !![1, 0; 0, N]

lemma coe_inv_apply (A : SL(2, ℤ)) :
    ((A⁻¹ : SL(2, ℤ)) : Mat2 ℤ) = !![A 1 1, -A 0 1; -A 1 0, A 0 0] := by
  rw [Matrix.SpecialLinearGroup.coe_inv, Matrix.adjugate_fin_two]

lemma det_SL (A : SL(2, ℤ)) : A 0 0 * A 1 1 - A 0 1 * A 1 0 = 1 := by
  have := A.2; rw [Matrix.det_fin_two] at this; exact this

lemma mul_inv_coe (A : SL(2, ℤ)) : (A : Mat2 ℤ) * ((A⁻¹ : SL(2, ℤ)) : Mat2 ℤ) = 1 := by
  rw [← Matrix.SpecialLinearGroup.coe_mul, mul_inv_cancel, Matrix.SpecialLinearGroup.coe_one]

lemma inv_mul_coe (A : SL(2, ℤ)) : ((A⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * (A : Mat2 ℤ) = 1 := by
  rw [← Matrix.SpecialLinearGroup.coe_mul, inv_mul_cancel, Matrix.SpecialLinearGroup.coe_one]

lemma mul_inv_coe_cancel_left (A : SL(2, ℤ)) (X : Mat2 ℤ) :
    (A : Mat2 ℤ) * (((A⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * X) = X := by
  rw [← Matrix.mul_assoc, mul_inv_coe, Matrix.one_mul]

lemma inv_mul_coe_cancel_left (A : SL(2, ℤ)) (X : Mat2 ℤ) :
    ((A⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * ((A : Mat2 ℤ) * X) = X := by
  rw [← Matrix.mul_assoc, inv_mul_coe, Matrix.one_mul]

/-- The bottom-left entry of `γ γ'⁻¹`. -/
lemma mul_inv_apply_10 (γ γ' : SL(2, ℤ)) : (γ * γ'⁻¹) 1 0 = γ 1 0 * γ' 1 1 - γ 1 1 * γ' 1 0 := by
  rw [Matrix.SpecialLinearGroup.coe_mul, coe_inv_apply, Matrix.mul_apply, Fin.sum_univ_two]
  simp; ring

lemma mem_Gamma0_iff_dvd (N : ℕ) (A : SL(2, ℤ)) : A ∈ Gamma0 N ↔ (N : ℤ) ∣ A 1 0 := by
  rw [Gamma0_mem, ZMod.intCast_zmod_eq_zero_iff_dvd]

/-- The bottom row of an element of `SL(2, ℤ)` is coprime. -/
lemma isCoprime_bottom (A : SL(2, ℤ)) : IsCoprime (A 1 0) (A 1 1) :=
  ⟨-A 0 1, A 0 0, by have := det_SL A; linarith⟩

/-- The top row of an element of `SL(2, ℤ)` is coprime. -/
lemma isCoprime_top (A : SL(2, ℤ)) : IsCoprime (A 0 0) (A 0 1) :=
  ⟨A 1 1, -A 1 0, by have := det_SL A; linarith⟩

/-! ## `Γ⁰(N)` -/

/-- `Γ⁰(N) = {γ ∈ Γ : N ∣ b}` ([C] `prop:cosets`(3)). -/
def Gamma0Up (N : ℤ) : Subgroup SL(2, ℤ) where
  carrier := {γ | N ∣ γ 0 1}
  mul_mem' := by
    intro a b ha hb
    simp only [Set.mem_ofPred_eq, Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_apply,
      Fin.sum_univ_two] at *
    exact dvd_add (dvd_mul_of_dvd_right hb _) (dvd_mul_of_dvd_left ha _)
  one_mem' := by simp
  inv_mem' := by
    intro a ha
    simp only [Set.mem_ofPred_eq, coe_inv_apply] at *
    simpa using ha

lemma mem_Gamma0Up (N : ℤ) (γ : SL(2, ℤ)) : γ ∈ Gamma0Up N ↔ N ∣ γ 0 1 := Iff.rfl

/-! ## The canonical transporters -/

variable (p ν : ℕ)

/-- The canonical transporter `s_k` ([C] `def:cosetreps`, `j = k + 1`). -/
def sMat (k : ℕ) : Mat2 ℤ :=
  if k < p ^ ν then !![1, 0; (k : ℤ), 1] else !![0, -1; 1, (p : ℤ) * ((k : ℤ) - (p : ℤ) ^ ν)]

/-- `s_k` as an element of `SL(2, ℤ)`. -/
def sSL (k : ℕ) : SL(2, ℤ) :=
  ⟨sMat p ν k, by unfold sMat; split_ifs <;> simp [Matrix.det_fin_two_of]⟩

lemma sSL_coe (k : ℕ) : ((sSL p ν k : SL(2, ℤ)) : Mat2 ℤ) = sMat p ν k := rfl

/-- The class of `γ` modulo `Γ₀(N)` is the class of `s_k` iff `RowRel` holds on the bottom row. -/
theorem mul_sInv_mem_Gamma0_iff (γ : SL(2, ℤ)) (k : ℕ) :
    γ * (sSL p ν k)⁻¹ ∈ Gamma0 (p ^ ν) ↔ RowRel p ν k (γ 1 0) (γ 1 1) := by
  rw [mem_Gamma0_iff_dvd, mul_inv_apply_10, sSL_coe]
  unfold RowRel sMat
  push_cast
  split_ifs <;> simp

variable {p ν}

/-- **[C] `prop:cosets`(4) with `def:cosetreps`: the `s_k`, `k < m`, are a transversal of `Γ₀(N)\Γ`.**
Every `γ ∈ Γ` lies in `Γ₀(N) s_k` for exactly one `k < m = p^ν + p^(ν-1)`. -/
theorem cosets_transversal (hp : p.Prime) (hν : 1 ≤ ν) (γ : SL(2, ℤ)) :
    ∃! k, k < mIdx p ν ∧ γ * (sSL p ν k)⁻¹ ∈ Gamma0 (p ^ ν) := by
  simp_rw [mul_sInv_mem_Gamma0_iff]
  exact rowRel_existsUnique hp hν (isCoprime_bottom γ)

/-- `s_k` has class `k`. -/
lemma sSL_class (k : ℕ) : sSL p ν k * (sSL p ν k)⁻¹ ∈ Gamma0 (p ^ ν) := by
  rw [mul_inv_cancel]; exact one_mem _

/-- The class map `γ ↦ k` with `γ ∈ Γ₀(N) s_k` ([C] `cls`, read through `def:cosetreps`). -/
noncomputable def cls (p ν : ℕ) (γ : SL(2, ℤ)) : ℕ := by
  classical exact
  if h : ∃ k, k < mIdx p ν ∧ γ * (sSL p ν k)⁻¹ ∈ Gamma0 (p ^ ν) then Nat.find h else 0

lemma cls_spec (hp : p.Prime) (hν : 1 ≤ ν) (γ : SL(2, ℤ)) :
    cls p ν γ < mIdx p ν ∧ γ * (sSL p ν (cls p ν γ))⁻¹ ∈ Gamma0 (p ^ ν) := by
  classical
  have h := (cosets_transversal hp hν γ).exists
  unfold cls; rw [dif_pos h]; exact Nat.find_spec h

lemma cls_eq (hp : p.Prime) (hν : 1 ≤ ν) {γ : SL(2, ℤ)} {k : ℕ} (hk : k < mIdx p ν)
    (h : γ * (sSL p ν k)⁻¹ ∈ Gamma0 (p ^ ν)) : cls p ν γ = k :=
  (cosets_transversal hp hν γ).unique (cls_spec hp hν γ) ⟨hk, h⟩

lemma cls_sSL (hp : p.Prime) (hν : 1 ≤ ν) {k : ℕ} (hk : k < mIdx p ν) : cls p ν (sSL p ν k) = k :=
  cls_eq hp hν hk (sSL_class k)

/-! ## The action and the cocycle ([C] `lem:cocycle`) -/

/-- `k · γ`: the index with `s_k γ ∈ Γ₀(N) s_{k·γ}`. -/
noncomputable def act (p ν : ℕ) (k : ℕ) (γ : SL(2, ℤ)) : ℕ := cls p ν (sSL p ν k * γ)

/-- The coset cocycle `h_k(γ) = s_k γ s_{k·γ}⁻¹` ([C] `eq:cocycle`). -/
noncomputable def hcoc (p ν : ℕ) (k : ℕ) (γ : SL(2, ℤ)) : SL(2, ℤ) :=
  sSL p ν k * γ * (sSL p ν (act p ν k γ))⁻¹

lemma act_lt (hp : p.Prime) (hν : 1 ≤ ν) (k : ℕ) (γ : SL(2, ℤ)) : act p ν k γ < mIdx p ν :=
  (cls_spec hp hν _).1

theorem hcoc_mem (hp : p.Prime) (hν : 1 ≤ ν) (k : ℕ) (γ : SL(2, ℤ)) :
    hcoc p ν k γ ∈ Gamma0 (p ^ ν) :=
  (cls_spec hp hν _).2

theorem hcoc_eq (k : ℕ) (γ : SL(2, ℤ)) : sSL p ν k * γ = hcoc p ν k γ * sSL p ν (act p ν k γ) := by
  unfold hcoc; group

theorem act_one (hp : p.Prime) (hν : 1 ≤ ν) {k : ℕ} (hk : k < mIdx p ν) : act p ν k 1 = k := by
  unfold act; rw [mul_one]; exact cls_sSL hp hν hk

theorem act_mul (hp : p.Prime) (hν : 1 ≤ ν) (k : ℕ) (γ₁ γ₂ : SL(2, ℤ)) :
    act p ν k (γ₁ * γ₂) = act p ν (act p ν k γ₁) γ₂ := by
  apply cls_eq hp hν (act_lt hp hν _ _)
  have h1 := hcoc_mem hp hν k γ₁
  have h2 := hcoc_mem hp hν (act p ν k γ₁) γ₂
  have : sSL p ν k * (γ₁ * γ₂) * (sSL p ν (act p ν (act p ν k γ₁) γ₂))⁻¹ =
      hcoc p ν k γ₁ * hcoc p ν (act p ν k γ₁) γ₂ := by unfold hcoc; group
  rw [this]; exact mul_mem h1 h2

/-- **[C] `lem:cocycle`: the cocycle identity.** -/
theorem hcoc_mul (hp : p.Prime) (hν : 1 ≤ ν) (k : ℕ) (γ₁ γ₂ : SL(2, ℤ)) :
    hcoc p ν k (γ₁ * γ₂) = hcoc p ν k γ₁ * hcoc p ν (act p ν k γ₁) γ₂ := by
  unfold hcoc; rw [act_mul hp hν]; group

theorem hcoc_one (hp : p.Prime) (hν : 1 ≤ ν) {k : ℕ} (hk : k < mIdx p ν) : hcoc p ν k 1 = 1 := by
  unfold hcoc; rw [act_one hp hν hk]; group

/-- **[C] `lem:cocycle`, the bound:** `|s_k| ≤ N` and `|s_k⁻¹| ≤ N` entrywise for `k < m`. -/
theorem sSL_bound (hν : 1 ≤ ν) (hp : 1 ≤ p) {k : ℕ} (hk : k < mIdx p ν) (i j : Fin 2) :
    |sSL p ν k i j| ≤ (p : ℤ) ^ ν ∧ |((sSL p ν k)⁻¹ : SL(2, ℤ)) i j| ≤ (p : ℤ) ^ ν := by
  have hN1 : (1 : ℤ) ≤ (p : ℤ) ^ ν := one_le_pow₀ (by exact_mod_cast hp)
  have hNM : (p : ℤ) ^ ν = p * (p : ℤ) ^ (ν - 1) := by rw [← pow_succ']; congr 1; omega
  have hm : (k : ℤ) < (p : ℤ) ^ ν + (p : ℤ) ^ (ν - 1) := by
    have : ((mIdx p ν : ℕ) : ℤ) = (p : ℤ) ^ ν + (p : ℤ) ^ (ν - 1) := by unfold mIdx; push_cast; ring
    rw [← this]; exact_mod_cast hk
  have hp0 : (0 : ℤ) ≤ p := by positivity
  rw [coe_inv_apply, sSL_coe]
  unfold sMat
  split_ifs with h
  · have hkI : (k : ℤ) < (p : ℤ) ^ ν := by exact_mod_cast h
    fin_cases i <;> fin_cases j <;> simp [abs_of_nonneg] <;> omega
  · have hkI : (p : ℤ) ^ ν ≤ (k : ℤ) := by exact_mod_cast Nat.le_of_not_lt h
    have hc : 0 ≤ (p : ℤ) * ((k : ℤ) - (p : ℤ) ^ ν) := mul_nonneg hp0 (by linarith)
    have hc' : (p : ℤ) * ((k : ℤ) - (p : ℤ) ^ ν) < (p : ℤ) ^ ν := by
      have : (k : ℤ) - (p : ℤ) ^ ν < (p : ℤ) ^ (ν - 1) := by linarith
      have hp1 : (0 : ℤ) < p := by exact_mod_cast hp
      have := mul_lt_mul_of_pos_left this hp1
      linarith
    fin_cases i <;> fin_cases j <;> simp [abs_of_nonneg hc] <;> linarith

end GL2.Cosets
