import Mathlib
import GL2.Conventions.SymPow
import GL2.Cosets.Boundary

/-!
# G1 (L17-3): the boundary characters ([C] `prop:bdrychar` (1), (2))

`R` a commutative ring of characteristic `ℓ` (prime), `p` a prime `≠ ℓ` (only `ℓ ∤ p` is used).  On
`R[x, y]` with the column convention of [C] `def:rhon` (`subst`, [GT] `conv:symconv`):
`T : x ↦ x, y ↦ x + y`.  `Nm = y^ℓ - x^(ℓ-1) y` (the `⟨T⟩`-orbit product of `y`, [C] `prop:bdrychar`'s
closed form) and `f_b = x^(n - bℓ) Nm^b`.

* (1) `f_b` is `T`-invariant, and on it the left-coset sum `ρ(δ^∂_∞) + Σ_{t<p} ρ(δ^∂_t)` (the sum over
  `B_p / Γ_∂`, [GT] `prop:bdrycosets`) is `χ_b(p) = p^b + p^(n+1-b)`;
* (2) the right-coset sum `M_β = Σ_{b'<p} ρ(β_{b'}) + ρ(β_∞)` is `χ_{b+1}(p)` on it, and `M_β` maps
  `V^T` into itself (via `T β_b = β_b T^p`, `T^p β_∞ = β_∞ T`, `V^{T^p} = V^T`).

Also here (L18-4): the identification of `V^T` with `R[x, Nm]_n` (`bdrychar_invariants`, proved by
dehomogenisation) and, with `bdrychar_invariants_indep`, its rank `⌊n/ℓ⌋ + 1`. What is *not* here: that the
left-coset sum is the operator `Θ^∂_g` on `H^0(C^•_∂)` ([C] `prop:bdryop`(2), G2).
-/

namespace GL2.Cosets

open MvPolynomial

variable {R : Type*} [CommRing R] (ℓ : ℕ) [hℓ : Fact ℓ.Prime] [CharP R ℓ]

local notation "x" => (X 0 : MvPolynomial (Fin 2) R)
local notation "y" => (X 1 : MvPolynomial (Fin 2) R)

/-- `Nm = y^ℓ - x^(ℓ-1) y`. -/
noncomputable def Nm : MvPolynomial (Fin 2) R := y ^ ℓ - x ^ (ℓ - 1) * y

/-- `f_b = x^(n - bℓ) Nm^b`. -/
noncomputable def fB (n b : ℕ) : MvPolynomial (Fin 2) R := x ^ (n - b * ℓ) * (Nm (R := R) ℓ) ^ b

omit [CharP R ℓ] in
/-- Fermat in `R`: `k^ℓ = k` for every natural `k` (`R` of characteristic `ℓ`). -/
lemma natCast_pow_char (R : Type*) [CommRing R] [CharP R ℓ] (k : ℕ) : ((k : R)) ^ ℓ = k := by
  have h := ZMod.pow_card (k : ZMod ℓ)
  have := congrArg (ZMod.castHom (dvd_refl ℓ) R) h
  rw [map_pow, map_natCast] at this; exact this

omit [CharP R ℓ] in
lemma natCast_pow_sub_one (R : Type*) [CommRing R] [CharP R ℓ] {p : ℕ} (hpl : ¬ ℓ ∣ p) :
    ((p : R)) ^ (ℓ - 1) = 1 := by
  have hne : (p : ZMod ℓ) ≠ 0 := by
    rw [Ne, ZMod.natCast_eq_zero_iff]; exact hpl
  have := congrArg (ZMod.castHom (dvd_refl ℓ) R) (ZMod.pow_card_sub_one_eq_one hne)
  rw [map_pow, map_natCast, map_one] at this; exact this

/-! ## The matrices (reduced into `R`) -/

/-- `T`. -/
def Tm : Mat2 R := !![1, 1; 0, 1]
/-- `T^k = [[1, k], [0, 1]]`. -/
def Tk (k : ℕ) : Mat2 R := !![1, (k : R); 0, 1]
/-- `δ^∂_∞ = diag(1, p)`. -/
def dInf (p : ℕ) : Mat2 R := !![1, 0; 0, (p : R)]
/-- `δ^∂_t = [[p, t], [0, 1]]`. -/
def dT (p t : ℕ) : Mat2 R := !![(p : R), (t : R); 0, 1]
/-- `β_b = [[1, b], [0, p]]`. -/
def bB (p b : ℕ) : Mat2 R := !![1, (b : R); 0, (p : R)]
/-- `β_∞ = diag(p, 1)`. -/
def bInf (p : ℕ) : Mat2 R := !![(p : R), 0; 0, 1]

omit [Fact ℓ.Prime] [CharP R ℓ] in
/-- The matrices are the reductions of the integral representatives of [GT] `prop:bdrycosets`. -/
lemma dbd_map (p t : ℕ) (ht : t ≤ p) :
    (dbd p t).map (Int.castRingHom R) = if t < p then dT p t else dInf p := by
  unfold dbd dT dInf
  split_ifs <;> ext i j <;> fin_cases i <;> fin_cases j <;> simp

omit [Fact ℓ.Prime] [CharP R ℓ] in
lemma betaMat_map (p b : ℕ) :
    (betaMat p b).map (Int.castRingHom R) = if b < p then bB p b else bInf p := by
  unfold betaMat bB bInf
  split_ifs <;> ext i j <;> fin_cases i <;> fin_cases j <;> simp

omit [Fact ℓ.Prime] [CharP R ℓ] in
@[simp] lemma subst_x (A : Mat2 R) : subst A x = C (A 0 0) * x + C (A 1 0) * y := by
  rw [subst_X, linForm_zero]

omit [Fact ℓ.Prime] [CharP R ℓ] in
@[simp] lemma subst_y (A : Mat2 R) : subst A y = C (A 0 1) * x + C (A 1 1) * y := by
  rw [subst_X, linForm_one]

/-! ## `Nm` under the matrices -/

local notation "MP" => MvPolynomial (Fin 2) R

lemma subst_Tm_Nm : subst Tm (Nm (R := R) ℓ) = Nm ℓ := by
  obtain ⟨l, hl⟩ : ∃ l, ℓ = l + 1 := ⟨ℓ - 1, by have := hℓ.out.one_lt; omega⟩
  unfold Nm Tm
  simp [linForm_zero, linForm_one, add_pow_char]
  subst hl; simp; ring

lemma subst_dInf_Nm (p : ℕ) : subst (dInf p) (Nm (R := R) ℓ) = C (p : R) * Nm ℓ := by
  have hpP := natCast_pow_char ℓ MP p
  unfold Nm dInf
  simp [linForm_zero, linForm_one, mul_pow, hpP]; ring

lemma subst_dT_Nm {p : ℕ} (hpl : ¬ ℓ ∣ p) (t : ℕ) : subst (dT p t) (Nm (R := R) ℓ) = Nm ℓ := by
  obtain ⟨l, hl⟩ : ∃ l, ℓ = l + 1 := ⟨ℓ - 1, by have := hℓ.out.one_lt; omega⟩
  have hp1 := natCast_pow_sub_one ℓ MP hpl
  have ht := natCast_pow_char ℓ MP t
  unfold Nm dT
  simp [linForm_zero, linForm_one, mul_pow, add_pow_char, hp1, ht]
  subst hl; simp; ring

lemma subst_bB_Nm (p b : ℕ) : subst (bB p b) (Nm (R := R) ℓ) = C (p : R) * Nm ℓ := by
  obtain ⟨l, hl⟩ : ∃ l, ℓ = l + 1 := ⟨ℓ - 1, by have := hℓ.out.one_lt; omega⟩
  have hp := natCast_pow_char ℓ MP p
  have hb := natCast_pow_char ℓ MP b
  unfold Nm bB
  simp [linForm_zero, linForm_one, mul_pow, add_pow_char, hp, hb]
  subst hl; simp; ring

lemma subst_bInf_Nm {p : ℕ} (hpl : ¬ ℓ ∣ p) : subst (bInf p) (Nm (R := R) ℓ) = Nm ℓ := by
  have hp1 := natCast_pow_sub_one ℓ MP hpl
  unfold Nm bInf
  simp [linForm_zero, linForm_one, mul_pow, hp1]

/-! ## (1) and (2): the eigenvalues -/

/-- `χ_b(p) = p^b + p^(n+1-b)`. -/
def chi (R : Type*) [CommRing R] (n p b : ℕ) : R := (p : R) ^ b + (p : R) ^ (n + 1 - b)

omit [CharP R ℓ] in
/-- `p^(n - bℓ) = p^(n - b)` in characteristic `ℓ`, `ℓ ∤ p`. -/
lemma pow_sub_mul_char (S : Type*) [CommRing S] [CharP S ℓ] {p : ℕ} (hpl : ¬ ℓ ∣ p) {n b : ℕ}
    (hb : b * ℓ ≤ n) : ((p : S)) ^ (n - b * ℓ) = (p : S) ^ (n - b) := by
  obtain ⟨l, hl⟩ : ∃ l, ℓ = l + 1 := ⟨ℓ - 1, by have := hℓ.out.one_lt; omega⟩
  have hp1 := natCast_pow_sub_one ℓ S hpl
  rw [hl, Nat.add_sub_cancel] at hp1
  have : n - b = (n - b * ℓ) + l * b := by rw [hl] at hb ⊢; rw [Nat.mul_add, mul_one] at hb ⊢; rw [mul_comm l b]; omega
  rw [this, pow_add, pow_mul, hp1, one_pow, mul_one]

lemma fB_T_invariant (n b : ℕ) : subst Tm (fB (R := R) ℓ n b) = fB ℓ n b := by
  unfold fB
  rw [map_mul, map_pow, map_pow, subst_Tm_Nm]
  simp [Tm, linForm_zero]

/-- **[C] `prop:bdrychar`(1), the eigenvalue:** on `f_b` (`bℓ ≤ n`), the left-coset sum
`ρ(δ^∂_∞) + Σ_{t<p} ρ(δ^∂_t)` acts by `χ_b(p)`. -/
theorem bdrychar_left {p : ℕ} (hpl : ¬ ℓ ∣ p) {n b : ℕ} (hb : b * ℓ ≤ n) :
    subst (dInf p) (fB (R := R) ℓ n b) + ∑ t ∈ Finset.range p, subst (dT p t) (fB ℓ n b) =
      C (chi R n p b) * fB ℓ n b := by
  have h1 : subst (dInf p) (fB (R := R) ℓ n b) = C ((p : R) ^ b) * fB ℓ n b := by
    unfold fB
    rw [map_mul, map_pow, map_pow, subst_dInf_Nm]
    simp [dInf, linForm_zero, mul_pow]; ring
  have h2 : ∀ t, subst (dT p t) (fB (R := R) ℓ n b) = C ((p : R) ^ (n - b)) * fB ℓ n b := by
    intro t
    unfold fB
    rw [map_mul, map_pow, map_pow, subst_dT_Nm ℓ hpl]
    simp [dT, linForm_zero, mul_pow, pow_sub_mul_char ℓ MP hpl hb]; ring
  rw [h1, Finset.sum_congr rfl (fun t _ => h2 t), Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hbn : b ≤ n := le_trans (Nat.le_mul_of_pos_right b hℓ.out.pos) hb
  have hn : n + 1 - b = (n - b) + 1 := by omega
  unfold chi
  rw [hn]
  simp only [map_add, map_mul, map_pow, map_natCast, pow_succ]
  ring

/-- **[C] `prop:bdrychar`(2), the eigenvalue:** with the right-coset representatives the same `f_b` has
eigenvalue `χ_{b+1}(p)`: the two operators differ. -/
theorem bdrychar_right {p : ℕ} (hpl : ¬ ℓ ∣ p) {n b : ℕ} (hb : b * ℓ ≤ n) :
    (∑ b' ∈ Finset.range p, subst (bB p b') (fB (R := R) ℓ n b)) + subst (bInf p) (fB ℓ n b) =
      C (chi R n p (b + 1)) * fB ℓ n b := by
  have h1 : ∀ b', subst (bB p b') (fB (R := R) ℓ n b) = C ((p : R) ^ b) * fB ℓ n b := by
    intro b'
    unfold fB
    rw [map_mul, map_pow, map_pow, subst_bB_Nm]
    simp [bB, linForm_zero, mul_pow]; ring
  have h2 : subst (bInf p) (fB (R := R) ℓ n b) = C ((p : R) ^ (n - b)) * fB ℓ n b := by
    unfold fB
    rw [map_mul, map_pow, map_pow, subst_bInf_Nm ℓ hpl]
    simp [bInf, linForm_zero, mul_pow, pow_sub_mul_char ℓ MP hpl hb]; ring
  rw [h2, Finset.sum_congr rfl (fun t _ => h1 t), Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hn : n + 1 - (b + 1) = n - b := by omega
  unfold chi
  rw [hn]
  simp only [map_add, map_mul, map_pow, map_natCast, pow_succ]
  ring

/-! ## (2): `M_β` preserves `V^T` -/

omit [Fact ℓ.Prime] [CharP R ℓ] in
lemma Tk_add (a b : ℕ) : (Tk a : Mat2 R) * Tk b = Tk (a + b) := by
  unfold Tk; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]; ring

omit [Fact ℓ.Prime] [CharP R ℓ] in
lemma fixed_Tk_mul {v : MvPolynomial (Fin 2) R} {k : ℕ} (h : subst (Tk k) v = v) (a : ℕ) :
    subst (Tk (k * a)) v = v := by
  induction a with
  | zero =>
    have : (Tk (k * 0) : Mat2 R) = 1 := by unfold Tk; ext i j; fin_cases i <;> fin_cases j <;> simp
    rw [this, subst_one]; rfl
  | succ a ih =>
    rw [Nat.mul_succ, ← Tk_add, subst_mul, AlgHom.comp_apply, h, ih]

/-- `V^{T^p} = V^T` for `ℓ ∤ p`. -/
lemma fixed_Tp_iff {p : ℕ} (hpl : ¬ ℓ ∣ p) (v : MvPolynomial (Fin 2) R) :
    subst (Tk p) v = v ↔ subst Tm v = v := by
  have hTm : (Tm : Mat2 R) = Tk 1 := by unfold Tm Tk; simp
  constructor
  · intro h
    have hne : (p : ZMod ℓ) ≠ 0 := by rw [Ne, ZMod.natCast_eq_zero_iff]; exact hpl
    set a := ((p : ZMod ℓ)⁻¹).val
    have hpa : ((p * a : ℕ) : R) = 1 := by
      have : ((p * a : ℕ) : ZMod ℓ) = 1 := by
        push_cast; simp only [a, ZMod.natCast_val, ZMod.cast_id', id_eq]
        exact mul_inv_cancel₀ hne
      have := congrArg (ZMod.castHom (dvd_refl ℓ) R) this
      simpa using this
    have := fixed_Tk_mul h a
    have hT : (Tk (p * a) : Mat2 R) = Tm := by unfold Tk Tm; rw [hpa]
    rwa [hT] at this
  · intro h
    rw [hTm] at h
    have := fixed_Tk_mul h p
    rwa [one_mul] at this

/-- **[C] `prop:bdrychar`(2):** `M_β = Σ_{b'<p} ρ(β_{b'}) + ρ(β_∞)` maps `V^T` into itself. -/
theorem Mbeta_preserves {p : ℕ} (hpl : ¬ ℓ ∣ p) (v : MvPolynomial (Fin 2) R) (hv : subst Tm v = v) :
    subst Tm ((∑ b' ∈ Finset.range p, subst (bB p b') v) + subst (bInf p) v) =
      (∑ b' ∈ Finset.range p, subst (bB p b') v) + subst (bInf p) v := by
  have hvp : subst (Tk p) v = v := (fixed_Tp_iff ℓ hpl v).2 hv
  -- `T β_b = β_b T^p`
  have e1 : ∀ b', (Tm : Mat2 R) * bB p b' = bB p b' * Tk p := by
    intro b'; unfold Tm bB Tk; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
    ring
  -- `T^p β_∞ = β_∞ T`
  have e2 : (Tk p : Mat2 R) * bInf p = bInf p * Tm := by
    unfold Tm bInf Tk; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
  have h1 : ∀ b', subst Tm (subst (bB p b') v) = subst (bB p b') v := by
    intro b'
    rw [← AlgHom.comp_apply, ← subst_mul, e1, subst_mul, AlgHom.comp_apply, hvp]
  have h2 : subst Tm (subst (bInf p) v) = subst (bInf p) v := by
    rw [← fixed_Tp_iff ℓ hpl, ← AlgHom.comp_apply, ← subst_mul, e2, subst_mul, AlgHom.comp_apply, hv]
  rw [map_add, map_sum, Finset.sum_congr rfl (fun b' _ => h1 b'), h2]

/-! ## (1): `V^T = R[x, Nm]_n` -/

/-- **[C] `prop:bdrychar`(1), the easy inclusion:** `R[x, Nm]_n ⊆ V^T`. -/
theorem bdrychar_invariants_mem (n : ℕ) (a : ℕ → R) :
    subst Tm (∑ b ∈ Finset.range (n / ℓ + 1), C (a b) * fB ℓ n b) =
      ∑ b ∈ Finset.range (n / ℓ + 1), C (a b) * fB ℓ n b := by
  rw [map_sum]
  refine Finset.sum_congr rfl (fun b _ => ?_)
  rw [map_mul, fB_T_invariant, algHom_C, MvPolynomial.algebraMap_eq]

omit [Fact ℓ.Prime] [CharP R ℓ] in
/-- Powers of a monic polynomial of positive degree are linearly independent over any commutative ring. -/
theorem monic_pow_indep {P : Polynomial R} (hP : P.Monic) (hd : 1 ≤ P.natDegree)
    (k : ℕ) (a : ℕ → R) (h : ∑ b ∈ Finset.range k, Polynomial.C (a b) * P ^ b = 0) : ∀ b < k, a b = 0 := by
  induction k with
  | zero => intro b hb; omega
  | succ k ih =>
    rw [Finset.sum_range_succ] at h
    have hk : a k = 0 := by
      have hc := congrArg (fun q => q.coeff (k * P.natDegree)) h
      simp only [Polynomial.coeff_add, Polynomial.coeff_zero] at hc
      have h1 : (Polynomial.C (a k) * P ^ k).coeff (k * P.natDegree) = a k := by
        rw [Polynomial.coeff_C_mul, ← hP.natDegree_pow k, (hP.pow k).coeff_natDegree, mul_one]
      have h2 : (∑ b ∈ Finset.range k, Polynomial.C (a b) * P ^ b).coeff (k * P.natDegree) = 0 := by
        rw [Polynomial.finsetSum_coeff]
        apply Finset.sum_eq_zero
        intro b hb
        rw [Polynomial.coeff_C_mul, Polynomial.coeff_eq_zero_of_natDegree_lt, mul_zero]
        calc (P ^ b).natDegree ≤ b * P.natDegree := Polynomial.natDegree_pow_le
          _ < k * P.natDegree := Nat.mul_lt_mul_of_pos_right (Finset.mem_range.1 hb) (by omega)
      rw [h1, h2] at hc; simpa using hc
    rw [hk, map_zero, zero_mul, add_zero] at h
    intro b hb
    rcases Nat.lt_succ_iff_lt_or_eq.1 hb with hb | rfl
    · exact ih h b hb
    · exact hk

/-- Dehomogenisation `x ↦ 1`, `y ↦ Y`. -/
noncomputable def dehom : MvPolynomial (Fin 2) R →ₐ[R] Polynomial R :=
  MvPolynomial.aeval ![1, Polynomial.X]

omit hℓ [CharP R ℓ] in
lemma dehom_fB (n b : ℕ) : dehom (fB (R := R) ℓ n b) = (Polynomial.X ^ ℓ - Polynomial.X) ^ b := by
  simp [dehom, fB, Nm]

lemma monic_artinSchreier : (Polynomial.X ^ ℓ - Polynomial.X : Polynomial R).Monic ∧
    1 ≤ (Polynomial.X ^ ℓ - Polynomial.X : Polynomial R).natDegree := by
  have h2 := hℓ.out.two_le
  have hnt : Nontrivial R := CharP.nontrivial_of_char_ne_one hℓ.out.one_lt.ne'
  have hm : (Polynomial.X ^ ℓ - Polynomial.X : Polynomial R).Monic :=
    Polynomial.monic_X_pow_sub (by rw [Polynomial.degree_X]; exact_mod_cast (by omega : 1 < ℓ))
  refine ⟨hm, ?_⟩
  rw [Polynomial.natDegree_sub_eq_left_of_natDegree_lt] <;> simp <;> omega

/-- **[C] `prop:bdrychar`(1), independence:** the `f_b`, `b ≤ ⌊n/ℓ⌋`, are `R`-linearly independent. -/
theorem bdrychar_invariants_indep (n : ℕ) (a : ℕ → R)
    (h : ∑ b ∈ Finset.range (n / ℓ + 1), C (a b) * fB ℓ n b = 0) :
    ∀ b ≤ n / ℓ, a b = 0 := by
  have h' := congrArg (dehom (R := R)) h
  simp only [map_sum, map_mul, dehom_fB, map_zero] at h'
  have h'' : ∑ b ∈ Finset.range (n / ℓ + 1),
      Polynomial.C (a b) * (Polynomial.X ^ ℓ - Polynomial.X : Polynomial R) ^ b = 0 := by
    rw [← h']; refine Finset.sum_congr rfl (fun b _ => ?_); simp [dehom]
  intro b hb
  exact monic_pow_indep (monic_artinSchreier ℓ).1 (monic_artinSchreier ℓ).2 _ a h'' b (by omega)


/-! ## (1): `V^T ⊆ R[x, Nm]_n` (L18-4)

Route (L17-3): dehomogenise (`x ↦ 1`, `y ↦ Y`); `T` becomes `Y ↦ Y + 1`; divide by the monic `P = Y^ℓ − Y`, which is
`Y ↦ Y + 1`-invariant in characteristic `ℓ`; the remainder (degree `< ℓ`) is invariant, hence constant (its top
coefficient `c_d` gives `d · c_d = 0` in degree `d − 1`, and `d` is a unit for `0 < d < ℓ`); recurse on the quotient;
rehomogenise (a homogeneous polynomial of degree `n` is determined by its dehomogenisation). -/

omit [Fact ℓ.Prime] [CharP R ℓ] in
lemma dehom_monomial (d : Fin 2 →₀ ℕ) (a : R) :
    dehom (monomial d a) = Polynomial.C a * Polynomial.X ^ (d 1) := by
  simp [dehom, MvPolynomial.aeval_monomial, Finsupp.prod_fintype, Fin.prod_univ_two, Polynomial.C_mul']

omit [Fact ℓ.Prime] [CharP R ℓ] in
lemma deg_two {n : ℕ} {f : MvPolynomial (Fin 2) R} (hf : f.IsHomogeneous n) {d : Fin 2 →₀ ℕ}
    (hd : f.coeff d ≠ 0) : d 0 + d 1 = n := by
  by_contra h
  exact hd (hf.coeff_eq_zero (by rw [Finsupp.degree_eq_sum, Fin.sum_univ_two]; exact h))

omit [Fact ℓ.Prime] [CharP R ℓ] in
lemma dehom_coeff_of_homog {n : ℕ} {f : MvPolynomial (Fin 2) R} (hf : f.IsHomogeneous n) (m : Fin 2 →₀ ℕ)
    (hm : m 0 + m 1 = n) : (dehom f).coeff (m 1) = f.coeff m := by
  classical
  conv_lhs => rw [f.as_sum]
  rw [map_sum, Polynomial.finsetSum_coeff]
  simp only [dehom_monomial, Polynomial.coeff_C_mul_X_pow]
  rw [Finset.sum_eq_single m]
  · simp
  · intro d hd hdm
    rw [if_neg]
    intro h1
    apply hdm
    have h0 := deg_two hf (mem_support_iff.mp hd)
    ext i; fin_cases i
    · show d 0 = m 0; omega
    · exact h1.symm
  · intro hm'; rw [notMem_support_iff.mp hm', ite_self]

omit [Fact ℓ.Prime] [CharP R ℓ] in
/-- A homogeneous polynomial of degree `n` in two variables is determined by its dehomogenisation. -/
lemma homog_eq_of_dehom_eq {n : ℕ} {f f' : MvPolynomial (Fin 2) R} (hf : f.IsHomogeneous n)
    (hf' : f'.IsHomogeneous n) (h : dehom f = dehom f') : f = f' := by
  ext m
  by_cases hm : m 0 + m 1 = n
  · rw [← dehom_coeff_of_homog hf m hm, ← dehom_coeff_of_homog hf' m hm, h]
  · rw [hf.coeff_eq_zero, hf'.coeff_eq_zero] <;> (rw [Finsupp.degree_eq_sum, Fin.sum_univ_two]; exact hm)

omit [Fact ℓ.Prime] [CharP R ℓ] in
lemma dehom_subst_Tm (f : MvPolynomial (Fin 2) R) :
    dehom (subst Tm f) = (dehom f).comp (Polynomial.X + 1) := by
  rw [Polynomial.comp_eq_aeval, ← AlgHom.comp_apply, ← AlgHom.comp_apply]
  congr 1
  apply MvPolynomial.algHom_ext
  intro i
  fin_cases i <;> simp [dehom, Tm, linForm_zero, linForm_one] <;> ring

omit [Fact ℓ.Prime] [CharP R ℓ] in
lemma coeff_comp_X_add_one (r : Polynomial R) (e : ℕ) (hr : r.natDegree ≤ e + 1) :
    (r.comp (Polynomial.X + 1)).coeff e = r.coeff e + ((e + 1 : ℕ) : R) * r.coeff (e + 1) := by
  conv_lhs => rw [r.as_sum_range' (e + 1 + 1) (by omega)]
  rw [Polynomial.comp_eq_aeval, map_sum, Polynomial.finsetSum_coeff]
  simp only [Polynomial.aeval_monomial, Polynomial.algebraMap_eq, Polynomial.coeff_C_mul,
    Polynomial.coeff_X_add_one_pow]
  rw [Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_eq_zero]
  · simp [Nat.choose_succ_self_right]; ring
  · intro i hi
    rw [Nat.choose_eq_zero_of_lt (Finset.mem_range.1 hi), Nat.cast_zero, mul_zero]

/-- A `Y ↦ Y + 1`-invariant polynomial of degree `< ℓ` is constant. -/
lemma const_of_comp_invariant (r : Polynomial R) (hdeg : r.natDegree < ℓ)
    (hr : r.comp (Polynomial.X + 1) = r) : r = Polynomial.C (r.coeff 0) := by
  apply Polynomial.eq_C_of_natDegree_eq_zero
  by_contra hne
  obtain ⟨e, he⟩ : ∃ e, r.natDegree = e + 1 := ⟨r.natDegree - 1, by omega⟩
  have hc := coeff_comp_X_add_one r e he.le
  rw [hr] at hc
  replace hc : ((e + 1 : ℕ) : R) * r.coeff (e + 1) = 0 := by linear_combination -hc
  have hu : IsUnit (((e + 1 : ℕ)) : R) := by
    have hz : ((e + 1 : ℕ) : ZMod ℓ) ≠ 0 := by
      rw [Ne, ZMod.natCast_eq_zero_iff]
      exact Nat.not_dvd_of_pos_of_lt (by omega) (by omega)
    have := (IsUnit.mk0 _ hz).map (ZMod.castHom (dvd_refl ℓ) R)
    simpa using this
  have hlc : r.coeff (e + 1) = 0 := (hu.mul_right_eq_zero).1 hc
  have : r.leadingCoeff = 0 := by rw [Polynomial.leadingCoeff, he, hlc]
  rw [Polynomial.leadingCoeff_eq_zero] at this
  rw [this, Polynomial.natDegree_zero] at he
  omega

/-- `Y ↦ Y + 1`-invariant polynomials of degree `≤ N` are combinations of `(Y^ℓ − Y)^b`, `b ≤ N / ℓ`. -/
lemma comp_invariant_span (N : ℕ) : ∀ g : Polynomial R, g.natDegree ≤ N → g.comp (Polynomial.X + 1) = g →
    ∃ a : ℕ → R, g = ∑ b ∈ Finset.range (N / ℓ + 1),
      Polynomial.C (a b) * (Polynomial.X ^ ℓ - Polynomial.X : Polynomial R) ^ b := by
  induction N using Nat.strong_induction_on with
  | _ N ih =>
  intro g hg hinv
  set P : Polynomial R := Polynomial.X ^ ℓ - Polynomial.X with hPdef
  obtain ⟨hPm, -⟩ := monic_artinSchreier (R := R) ℓ
  have hPdeg : P.natDegree = ℓ := by
    have hnt : Nontrivial R := CharP.nontrivial_of_char_ne_one hℓ.out.one_lt.ne'
    have h2 := hℓ.out.two_le
    rw [hPdef, Polynomial.natDegree_sub_eq_left_of_natDegree_lt] <;> simp <;> omega
  have hPinv : P.comp (Polynomial.X + 1) = P := by
    rw [hPdef, Polynomial.sub_comp, Polynomial.X_pow_comp, Polynomial.X_comp, add_pow_char, one_pow]
    ring
  by_cases hN : N < ℓ
  · refine ⟨fun _ => g.coeff 0, ?_⟩
    rw [Nat.div_eq_of_lt hN, zero_add, Finset.sum_range_one, pow_zero, mul_one]
    exact const_of_comp_invariant ℓ g (by omega) hinv
  · push_neg at hN
    have hdiv := Polynomial.modByMonic_add_div g P
    have hP1' : P ≠ 1 := fun h => by rw [h, Polynomial.natDegree_one] at hPdeg; have := hℓ.out.one_lt; omega
    have hrdeg : (g %ₘ P).natDegree < ℓ := hPdeg ▸ Polynomial.natDegree_modByMonic_lt g hPm hP1'
    have hqdeg : (g /ₘ P).natDegree ≤ N - ℓ := by
      rw [Polynomial.natDegree_divByMonic g hPm, hPdeg]; omega
    obtain ⟨q, hq⟩ : ∃ q, g /ₘ P = q := ⟨_, rfl⟩
    obtain ⟨r, hr⟩ : ∃ r, g %ₘ P = r := ⟨_, rfl⟩
    rw [hq, hr] at hdiv
    rw [hr] at hrdeg
    rw [hq] at hqdeg
    have hrc : (r.comp (Polynomial.X + 1)).natDegree < ℓ := by
      calc (r.comp (Polynomial.X + 1)).natDegree ≤ r.natDegree * (Polynomial.X + 1 : Polynomial R).natDegree :=
            Polynomial.natDegree_comp_le
        _ ≤ r.natDegree * 1 := Nat.mul_le_mul_left _
            ((Polynomial.natDegree_add_le _ _).trans (max_le Polynomial.natDegree_X_le (by simp)))
        _ < ℓ := by omega
    have huniq := Polynomial.div_modByMonic_unique (f := g) (q.comp (Polynomial.X + 1)) (r.comp (Polynomial.X + 1))
      hPm ⟨by
        conv_rhs => rw [← hinv, ← hdiv]
        rw [Polynomial.add_comp, Polynomial.mul_comp, hPinv],
       Polynomial.degree_lt_degree (by rw [hPdeg]; exact hrc)⟩
    rw [hq, hr] at huniq
    have hqi : q.comp (Polynomial.X + 1) = q := huniq.1.symm
    have hri : r.comp (Polynomial.X + 1) = r := huniq.2.symm
    obtain ⟨a, ha⟩ := ih (N - ℓ) (by have := hℓ.out.one_lt; omega) q hqdeg hqi
    obtain ⟨c, hc⟩ : ∃ c, r = Polynomial.C c := ⟨_, const_of_comp_invariant ℓ r hrdeg hri⟩
    refine ⟨fun b => if b = 0 then c else a (b - 1), ?_⟩
    have hNl : N / ℓ + 1 = (N - ℓ) / ℓ + 1 + 1 := by
      conv_lhs => rw [← Nat.sub_add_cancel hN, Nat.add_div_right _ hℓ.out.pos]
    rw [hNl, Finset.sum_range_succ', ← hdiv, ha, hc, Finset.mul_sum]
    simp only [pow_zero, mul_one, Nat.add_one_ne_zero, if_false, Nat.add_sub_cancel]
    rw [add_comm]
    congr 1
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [pow_succ]; ring

/-- **[C] `prop:bdrychar`(1), the invariants** (proved, L18-4): every `T`-invariant homogeneous `f` of degree `n`
is an `R`-combination of the `f_b = x^(n - bℓ) Nm^b`, `b ≤ ⌊n/ℓ⌋`; with `bdrychar_invariants_indep`, `V^T` is free
of rank `⌊n/ℓ⌋ + 1`. -/
theorem bdrychar_invariants (n : ℕ) (f : MvPolynomial (Fin 2) R) (hf : f.IsHomogeneous n)
    (hT : subst Tm f = f) :
    ∃ a : ℕ → R, f = ∑ b ∈ Finset.range (n / ℓ + 1), C (a b) * fB ℓ n b := by
  have hinv : (dehom f).comp (Polynomial.X + 1) = dehom f := by rw [← dehom_subst_Tm, hT]
  have hdeg : (dehom f).natDegree ≤ n := by
    rw [Polynomial.natDegree_le_iff_coeff_eq_zero]
    intro k hk
    conv_lhs => rw [f.as_sum]
    rw [map_sum, Polynomial.finsetSum_coeff]
    refine Finset.sum_eq_zero fun d hd => ?_
    rw [dehom_monomial, Polynomial.coeff_C_mul_X_pow, if_neg]
    have := deg_two hf (mem_support_iff.mp hd)
    omega
  obtain ⟨a, ha⟩ := comp_invariant_span ℓ n (dehom f) hdeg hinv
  refine ⟨a, homog_eq_of_dehom_eq hf ?_ ?_⟩
  · refine IsHomogeneous.sum _ _ _ fun b hb => ?_
    have hb' : b * ℓ ≤ n := by
      have := Finset.mem_range.1 hb
      calc b * ℓ ≤ n / ℓ * ℓ := Nat.mul_le_mul_right _ (by omega)
        _ ≤ n := Nat.div_mul_le_self n ℓ
    have hNm : (Nm (R := R) ℓ).IsHomogeneous ℓ := by
      have h2 := (isHomogeneous_X_pow (R := R) (0 : Fin 2) (ℓ - 1)).mul (isHomogeneous_X (R := R) (1 : Fin 2))
      rw [Nat.sub_add_cancel hℓ.out.one_lt.le] at h2
      exact (isHomogeneous_X_pow (1 : Fin 2) ℓ).sub h2
    have h1 := (isHomogeneous_C (Fin 2) (a b)).mul
      ((isHomogeneous_X_pow (R := R) (0 : Fin 2) (n - b * ℓ)).mul (hNm.pow b))
    have he : 0 + ((n - b * ℓ) + ℓ * b) = n := by rw [mul_comm ℓ b]; omega
    rw [he] at h1
    exact h1
  · rw [ha, map_sum]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [map_mul, dehom_fB, dehom, MvPolynomial.aeval_C, Polynomial.algebraMap_eq]

end GL2.Cosets
