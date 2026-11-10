import Mathlib
import GL2.Cosets.Basic

/-!
# G1 (L17-3): the double coset `Γ g Γ` ([C] `prop:cosets` (1), (2), (3), (5))

* (3) `Γ ∩ g Γ g⁻¹ = Γ₀(N)` and `Γ ∩ g⁻¹ Γ g = Γ⁰(N)`, stated integrally (no inverse of `g`), with
  the conjugation formula;
* (1) `Γ g Γ` is the set of primitive integral matrices of determinant `N`, and the Hermite normal
  form (existence and uniqueness);
* (5) `δ_k = s_k⁻¹ g`, `k < m`, represent the left cosets `Γ g Γ / Γ`, each exactly once;
* (2) transposition maps `Γ g Γ` onto itself, so the `δ_kᵀ` represent the right cosets, again `m`.
-/

namespace GL2.Cosets

open Matrix CongruenceSubgroup
open scoped MatrixGroups

/-! ## (3) The two congruence subgroups -/

/-- **[C] `prop:cosets`(3), `Γ'' = Γ ∩ g Γ g⁻¹ = Γ₀(N)`:** `γ g ∈ g Γ` iff `N ∣ c`. -/
theorem conj_Gamma0_iff (N : ℤ) (γ : SL(2, ℤ)) :
    (∃ γ' : SL(2, ℤ), (γ : Mat2 ℤ) * gMat N = gMat N * (γ' : Mat2 ℤ)) ↔ N ∣ γ 1 0 := by
  constructor
  · rintro ⟨γ', h⟩
    have := congrFun (congrFun h 1) 0
    simp [gMat, Matrix.mul_apply, Fin.sum_univ_two] at this
    exact ⟨γ' 1 0, this⟩
  · rintro ⟨c', hc⟩
    refine ⟨mkSL (γ 0 0) (N * γ 0 1) c' (γ 1 1) (by have := det_SL γ; rw [hc] at this; linarith), ?_⟩
    ext i j; fin_cases i <;> fin_cases j <;>
      simp [gMat, Matrix.mul_apply, Fin.sum_univ_two, hc] <;> ring

/-- The same with mathlib's `Γ₀(N)`, `N = p^ν`. -/
theorem conj_Gamma0 (p ν : ℕ) (γ : SL(2, ℤ)) :
    γ ∈ Gamma0 (p ^ ν) ↔ ∃ γ' : SL(2, ℤ), (γ : Mat2 ℤ) * gMat ((p : ℤ) ^ ν) = gMat ((p : ℤ) ^ ν) * γ' := by
  rw [conj_Gamma0_iff, mem_Gamma0_iff_dvd]; push_cast; rfl

/-- **[C] `prop:cosets`(3), `Γ' = Γ ∩ g⁻¹ Γ g = Γ⁰(N)`:** `g γ ∈ Γ g` iff `N ∣ b`. -/
theorem conj_Gamma0Up_iff (N : ℤ) (γ : SL(2, ℤ)) :
    (∃ γ' : SL(2, ℤ), gMat N * (γ : Mat2 ℤ) = (γ' : Mat2 ℤ) * gMat N) ↔ γ ∈ Gamma0Up N := by
  rw [mem_Gamma0Up]
  constructor
  · rintro ⟨γ', h⟩
    have := congrFun (congrFun h 0) 1
    simp [gMat, Matrix.mul_apply, Fin.sum_univ_two] at this
    exact ⟨γ' 0 1, by rw [this]; ring⟩
  · rintro ⟨b', hb⟩
    refine ⟨mkSL (γ 0 0) b' (N * γ 1 0) (γ 1 1) (by have := det_SL γ; rw [hb] at this; linarith), ?_⟩
    ext i j; fin_cases i <;> fin_cases j <;>
      simp [gMat, Matrix.mul_apply, Fin.sum_univ_two, hb] <;> ring

/-- **[C] `prop:cosets`(3), the conjugation formula:** `g [[a, N b], [c, d]] g⁻¹ = [[a, b], [N c, d]]`,
written integrally. -/
theorem conj_formula (N a b c d : ℤ) :
    gMat N * !![a, N * b; c, d] = !![a, b; N * c, d] * gMat N := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [gMat, Matrix.mul_apply, Fin.sum_univ_two] <;> ring

/-! ## (1) Primitive matrices -/

/-- The double coset `Γ g Γ`, `g = diag(1, N)`. -/
def DC (N : ℤ) : Set (Mat2 ℤ) := {δ | ∃ γ₁ γ₂ : SL(2, ℤ), δ = (γ₁ : Mat2 ℤ) * gMat N * γ₂}

/-- The content of `δ` is one. -/
def IsPrimitive (δ : Mat2 ℤ) : Prop := ∀ e : ℤ, (∀ i j, e ∣ δ i j) → IsUnit e

lemma dvd_entries_mul_left {e : ℤ} {A : Mat2 ℤ} (h : ∀ i j, e ∣ A i j) (B : Mat2 ℤ) :
    ∀ i j, e ∣ (B * A) i j := by
  intro i j; rw [Matrix.mul_apply, Fin.sum_univ_two]
  exact dvd_add (dvd_mul_of_dvd_right (h _ _) _) (dvd_mul_of_dvd_right (h _ _) _)

lemma dvd_entries_mul_right {e : ℤ} {A : Mat2 ℤ} (h : ∀ i j, e ∣ A i j) (B : Mat2 ℤ) :
    ∀ i j, e ∣ (A * B) i j := by
  intro i j; rw [Matrix.mul_apply, Fin.sum_univ_two]
  exact dvd_add (dvd_mul_of_dvd_left (h _ _) _) (dvd_mul_of_dvd_left (h _ _) _)

/-- The content is `Γ`-biinvariant. -/
theorem isPrimitive_mul_iff (γ₁ γ₂ : SL(2, ℤ)) (δ : Mat2 ℤ) :
    IsPrimitive ((γ₁ : Mat2 ℤ) * δ * γ₂) ↔ IsPrimitive δ := by
  have back : ∀ (A : Mat2 ℤ) (u v : SL(2, ℤ)), IsPrimitive ((u : Mat2 ℤ) * A * v) → IsPrimitive A := by
    intro A u v h e he
    exact h e (dvd_entries_mul_right (dvd_entries_mul_left he _) _)
  constructor
  · exact back δ γ₁ γ₂
  · intro h
    apply back _ γ₁⁻¹ γ₂⁻¹
    have : ((γ₁⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * ((γ₁ : Mat2 ℤ) * δ * γ₂) * ((γ₂⁻¹ : SL(2, ℤ)) : Mat2 ℤ) = δ := by
      rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, inv_mul_coe, Matrix.one_mul, Matrix.mul_assoc,
        mul_inv_coe, Matrix.mul_one]
    rw [this]; exact h

lemma gMat_det (N : ℤ) : (gMat N).det = N := by simp [gMat, Matrix.det_fin_two_of]

lemma gMat_isPrimitive (N : ℤ) : IsPrimitive (gMat N) := by
  intro e he
  have := he 0 0
  simp [gMat] at this
  exact isUnit_of_dvd_one this

lemma DC_left_mul {N : ℤ} {δ : Mat2 ℤ} (h : δ ∈ DC N) (γ : SL(2, ℤ)) : (γ : Mat2 ℤ) * δ ∈ DC N := by
  obtain ⟨γ₁, γ₂, rfl⟩ := h
  exact ⟨γ * γ₁, γ₂, by simp [Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc]⟩

lemma DC_right_mul {N : ℤ} {δ : Mat2 ℤ} (h : δ ∈ DC N) (γ : SL(2, ℤ)) : δ * (γ : Mat2 ℤ) ∈ DC N := by
  obtain ⟨γ₁, γ₂, rfl⟩ := h
  exact ⟨γ₁, γ₂ * γ, by simp [Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc]⟩

lemma isUnit_of_dvd_ppow {p ν : ℕ} (hp : p.Prime) {x : ℤ} (hx : ¬ (p : ℤ) ∣ x) (hdvd : x ∣ (p : ℤ) ^ ν) :
    IsUnit x :=
  (isCoprime_pow_of_not_dvd hp hx ν).isUnit_of_dvd' (dvd_refl x) hdvd

lemma isCoprime_of_isUnit_right {x y : ℤ} (hy : IsUnit y) : IsCoprime x y := by
  rcases Int.isUnit_iff.1 hy with rfl | rfl
  · exact ⟨0, 1, by ring⟩
  · exact ⟨0, -1, by ring⟩

/-- A primitive upper-triangular matrix of determinant `p^ν` lies in `Γ g Γ` (explicit transporters). -/
theorem upper_mem_DC {p ν : ℕ} (hp : p.Prime) {a b d : ℤ} (had : a * d = (p : ℤ) ^ ν)
    (hprim : IsPrimitive !![a, b; 0, d]) : !![a, b; 0, d] ∈ DC ((p : ℤ) ^ ν) := by
  have hp' : Prime (p : ℤ) := Nat.prime_iff_prime_int.mp hp
  have key : IsCoprime a b ∨ IsCoprime b d := by
    by_cases ha : (p : ℤ) ∣ a
    · by_cases hd : (p : ℤ) ∣ d
      · left
        have hb : ¬ (p : ℤ) ∣ b := by
          intro hb
          apply hp'.not_isUnit
          apply hprim
          intro i j; fin_cases i <;> fin_cases j <;> simp [ha, hb, hd]
        exact ((isCoprime_pow_of_not_dvd hp hb ν).of_isCoprime_of_dvd_right ⟨d, had.symm⟩).symm
      · right
        exact isCoprime_of_isUnit_right (isUnit_of_dvd_ppow hp hd ⟨a, by rw [← had]; ring⟩)
    · left
      exact (isCoprime_of_isUnit_right (x := b) (isUnit_of_dvd_ppow hp ha ⟨d, had.symm⟩)).symm
  rcases key with ⟨u, v, huv⟩ | ⟨u, v, huv⟩
  · refine ⟨mkSL 1 0 (d * v) 1 (by ring), mkSL a b (-v) u (by linarith), ?_⟩
    ext i j; fin_cases i <;> fin_cases j <;> simp [gMat, Matrix.mul_apply, Fin.sum_univ_two]
    · linear_combination (-v) * had
    · linear_combination (-d) * huv + u * had
  · refine ⟨mkSL b (-v) d u (by linarith), mkSL (u * a) 1 (-1) 0 (by ring), ?_⟩
    ext i j; fin_cases i <;> fin_cases j <;> simp [gMat, Matrix.mul_apply, Fin.sum_univ_two]
    · linear_combination (-a) * huv + v * had
    · linear_combination (-u) * had

/-- Left multiplication by `Γ` makes the lower-left entry zero (Bézout on the first column). -/
theorem exists_lower_zero (δ : Mat2 ℤ) (hdet : δ.det ≠ 0) : ∃ γ : SL(2, ℤ), ((γ : Mat2 ℤ) * δ) 1 0 = 0 := by
  set a := δ 0 0 with hadef
  set c := δ 1 0 with hcdef
  have hne : a ≠ 0 ∨ c ≠ 0 := by
    by_contra h
    simp only [not_or, not_not] at h
    apply hdet; rw [Matrix.det_fin_two, ← hadef, ← hcdef, h.1, h.2]; ring
  have hpos : 0 < Int.gcd a c := Int.gcd_pos_iff.2 hne
  obtain ⟨a', ha'⟩ := Int.gcd_dvd_left a c
  obtain ⟨c', hc'⟩ := Int.gcd_dvd_right a c
  have hb := Int.gcd_eq_gcd_ab a c
  have he : ((Int.gcd a c : ℕ) : ℤ) ≠ 0 := by exact_mod_cast hpos.ne'
  have h1 : a' * Int.gcdA a c + c' * Int.gcdB a c = 1 := by
    apply mul_left_cancel₀ he
    have : ((Int.gcd a c : ℕ) : ℤ) = ((Int.gcd a c : ℕ) : ℤ) * a' * Int.gcdA a c +
        ((Int.gcd a c : ℕ) : ℤ) * c' * Int.gcdB a c := by
      conv_lhs => rw [hb]
      rw [← ha', ← hc']
    linear_combination -this
  refine ⟨mkSL (Int.gcdA a c) (Int.gcdB a c) (-c') a' (by linarith), ?_⟩
  simp only [mkSL_coe, Matrix.mul_apply, Fin.sum_univ_two]
  simp
  rw [← hadef, ← hcdef]
  nth_rewrite 1 [ha']; nth_rewrite 2 [hc']; ring

/-- **[C] `prop:cosets`(1): `Γ g Γ` is the set of primitive integral matrices of determinant `N`.** -/
theorem mem_DC_iff {p ν : ℕ} (hp : p.Prime) (δ : Mat2 ℤ) :
    δ ∈ DC ((p : ℤ) ^ ν) ↔ δ.det = (p : ℤ) ^ ν ∧ IsPrimitive δ := by
  constructor
  · rintro ⟨γ₁, γ₂, rfl⟩
    refine ⟨?_, (isPrimitive_mul_iff γ₁ γ₂ _).2 (gMat_isPrimitive _)⟩
    rw [Matrix.det_mul, Matrix.det_mul, Matrix.SpecialLinearGroup.det_coe,
      Matrix.SpecialLinearGroup.det_coe, gMat_det]; ring
  · rintro ⟨hdet, hprim⟩
    have hN : ((p : ℤ) ^ ν) ≠ 0 := pow_ne_zero _ (by exact_mod_cast hp.ne_zero)
    obtain ⟨γ, hγ⟩ := exists_lower_zero δ (by rw [hdet]; exact hN)
    set H := (γ : Mat2 ℤ) * δ with hH
    have hHform : H = !![H 0 0, H 0 1; 0, H 1 1] := by
      ext i j; fin_cases i <;> fin_cases j <;> simp [hγ]
    have hHdet : H 0 0 * H 1 1 = (p : ℤ) ^ ν := by
      have : H.det = (p : ℤ) ^ ν := by
        rw [hH, Matrix.det_mul, Matrix.SpecialLinearGroup.det_coe, hdet, one_mul]
      rw [Matrix.det_fin_two, hγ] at this; linarith
    have hHprim : IsPrimitive H := by
      have := (isPrimitive_mul_iff γ 1 δ).2 hprim
      simpa [hH] using this
    have hmem : H ∈ DC ((p : ℤ) ^ ν) := by
      rw [hHform]; rw [hHform] at hHprim; exact upper_mem_DC hp hHdet hHprim
    have : δ = ((γ⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * H := by
      rw [hH, inv_mul_coe_cancel_left]
    rw [this]; exact DC_left_mul hmem _

/-! ## The Hermite normal form -/

/-- **[C] `prop:cosets`(1), proof: Hermite reduction, existence.** Every integral `δ` with `det δ > 0`
lies in `Γ [[a, b], [0, d]]` with `a, d > 0` and `0 ≤ b < d`. -/
theorem hermite_exists (δ : Mat2 ℤ) (hdet : 0 < δ.det) :
    ∃ γ : SL(2, ℤ), ∃ a b d : ℤ, (γ : Mat2 ℤ) * δ = !![a, b; 0, d] ∧ 0 < a ∧ 0 < d ∧ 0 ≤ b ∧ b < d := by
  obtain ⟨γ, hγ⟩ := exists_lower_zero δ hdet.ne'
  set H := (γ : Mat2 ℤ) * δ with hH
  have hHdet : H 0 0 * H 1 1 = δ.det := by
    have : H.det = δ.det := by rw [hH, Matrix.det_mul, Matrix.SpecialLinearGroup.det_coe, one_mul]
    rw [Matrix.det_fin_two, hγ] at this; linarith
  obtain ⟨σ, a, b', d, hσ, ha, hd⟩ : ∃ σ : SL(2, ℤ), ∃ a b' d : ℤ,
      (σ : Mat2 ℤ) * H = !![a, b'; 0, d] ∧ 0 < a ∧ 0 < d := by
    rcases lt_or_gt_of_ne (show H 0 0 ≠ 0 by intro h; rw [h] at hHdet; linarith) with hneg | hpos
    · refine ⟨mkSL (-1) 0 0 (-1) (by ring), -H 0 0, -H 0 1, -H 1 1, ?_, by linarith, ?_⟩
      · ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two, hγ]
      · by_contra h; nlinarith
    · refine ⟨1, H 0 0, H 0 1, H 1 1, ?_, hpos, ?_⟩
      · ext i j; fin_cases i <;> fin_cases j <;> simp [hγ]
      · by_contra h; nlinarith
  refine ⟨mkSL 1 (-(b' / d)) 0 1 (by ring) * σ * γ, a, b' % d, d, ?_, ha, hd,
    Int.emod_nonneg _ hd.ne', Int.emod_lt_of_pos _ hd⟩
  rw [Matrix.SpecialLinearGroup.coe_mul, Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc,
    Matrix.mul_assoc, ← hH, hσ, Int.emod_def]
  ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two] <;> ring

/-- **Hermite reduction, uniqueness:** two Hermite forms in one right coset `Γ δ` coincide. -/
theorem hermite_unique (γ : SL(2, ℤ)) {a b d a' b' d' : ℤ}
    (h : (γ : Mat2 ℤ) * !![a, b; 0, d] = !![a', b'; 0, d'])
    (ha : 0 < a) (hd : 0 < d) (hb : 0 ≤ b) (hbd : b < d)
    (ha' : 0 < a') (hd' : 0 < d') (hb' : 0 ≤ b') (hbd' : b' < d') :
    a = a' ∧ b = b' ∧ d = d' := by
  have e := fun i j => congrFun (congrFun h i) j
  have e00 := e 0 0; have e01 := e 0 1; have e10 := e 1 0; have e11 := e 1 1
  simp [Matrix.mul_apply, Fin.sum_univ_two] at e00 e01 e10 e11
  have hdet := det_SL γ
  have hc : γ 1 0 = 0 := e10.resolve_right ha.ne'
  rw [hc] at hdet
  simp only [mul_zero, sub_zero] at hdet
  have hγ0 : 0 < γ 0 0 := by
    by_contra h; nlinarith
  have hγ1' : 0 < γ 1 1 := by nlinarith
  have hγ1 : γ 0 0 = 1 ∧ γ 1 1 = 1 := by
    constructor <;> nlinarith
  obtain ⟨h0, h1⟩ := hγ1
  have e11' : d = d' := by rw [hc, h1] at e11; linarith
  have e00' : a = a' := by rw [h0] at e00; linarith
  rw [h0] at e01
  refine ⟨e00', ?_, e11'⟩
  have hx : γ 0 1 = 0 := by
    by_contra hne
    rcases lt_or_gt_of_ne hne with hl | hl <;> nlinarith
  rw [hx] at e01; linarith

/-! ## (5) The left cosets, and (2) the right cosets -/

/-- The coset elements `δ_k = s_k⁻¹ g` ([C] `prop:cosets`(5)). -/
def deltaMat (p ν k : ℕ) : Mat2 ℤ := (((sSL p ν k)⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * gMat ((p : ℤ) ^ ν)

lemma deltaMat_mem_DC (p ν k : ℕ) : deltaMat p ν k ∈ DC ((p : ℤ) ^ ν) :=
  ⟨(sSL p ν k)⁻¹, 1, by simp [deltaMat]⟩

lemma deltaMat_det (p ν k : ℕ) : (deltaMat p ν k).det = (p : ℤ) ^ ν := by
  rw [deltaMat, Matrix.det_mul, Matrix.SpecialLinearGroup.det_coe, gMat_det, one_mul]

/-- **[C] `prop:cosets`(5): the `δ_k = s_k⁻¹ g`, `k < m`, represent `Γ g Γ / Γ`, each coset once.** -/
theorem leftCosets {p ν : ℕ} (hp : p.Prime) (hν : 1 ≤ ν) {δ : Mat2 ℤ} (hδ : δ ∈ DC ((p : ℤ) ^ ν)) :
    ∃! k, k < mIdx p ν ∧ ∃ γ : SL(2, ℤ), δ = deltaMat p ν k * γ := by
  obtain ⟨γ₁, γ₂, rfl⟩ := hδ
  obtain ⟨hk, hmem⟩ := cls_spec hp hν γ₁⁻¹
  set k := cls p ν γ₁⁻¹
  set h := γ₁⁻¹ * (sSL p ν k)⁻¹ with hh
  have hinv : h⁻¹ ∈ Gamma0 (p ^ ν) := inv_mem hmem
  obtain ⟨u, hu⟩ := (conj_Gamma0 p ν h⁻¹).1 hinv
  refine ⟨k, ⟨hk, u * γ₂, ?_⟩, ?_⟩
  · have hγ₁ : γ₁ = (sSL p ν k)⁻¹ * h⁻¹ := by rw [hh]; group
    rw [hγ₁, deltaMat, Matrix.SpecialLinearGroup.coe_mul, Matrix.SpecialLinearGroup.coe_mul,
      Matrix.mul_assoc, Matrix.mul_assoc, ← Matrix.mul_assoc _ (gMat _), hu]
    simp [Matrix.mul_assoc]
  · rintro k' ⟨hk', γ', hγ'⟩
    have heq : ((sSL p ν k' * γ₁ : SL(2, ℤ)) : Mat2 ℤ) * gMat ((p : ℤ) ^ ν) =
        gMat ((p : ℤ) ^ ν) * ((γ' * γ₂⁻¹ : SL(2, ℤ)) : Mat2 ℤ) := by
      have := congrArg (fun X => ((sSL p ν k' : SL(2, ℤ)) : Mat2 ℤ) * X * ((γ₂⁻¹ : SL(2, ℤ)) : Mat2 ℤ)) hγ'
      simp only [deltaMat, Matrix.mul_assoc, mul_inv_coe, Matrix.mul_one, mul_inv_coe_cancel_left] at this
      rw [Matrix.SpecialLinearGroup.coe_mul, Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc, this]
    have hmem' : sSL p ν k' * γ₁ ∈ Gamma0 (p ^ ν) := (conj_Gamma0 p ν _).2 ⟨_, heq⟩
    have : γ₁⁻¹ * (sSL p ν k')⁻¹ ∈ Gamma0 (p ^ ν) := by
      have := inv_mem hmem'; rwa [_root_.mul_inv_rev] at this
    exact (cls_eq hp hν hk' this).symm

/-- `Γ` is stable under transposition. -/
def trSL (γ : SL(2, ℤ)) : SL(2, ℤ) := ⟨(γ : Mat2 ℤ)ᵀ, by rw [Matrix.det_transpose]; exact γ.2⟩

@[simp] lemma trSL_coe (γ : SL(2, ℤ)) : ((trSL γ : SL(2, ℤ)) : Mat2 ℤ) = (γ : Mat2 ℤ)ᵀ := rfl

lemma gMat_transpose (N : ℤ) : (gMat N)ᵀ = gMat N := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [gMat]

/-- **[C] `prop:cosets`(2): transposition carries `Γ g Γ` onto itself.** -/
theorem transpose_mem_DC {N : ℤ} {δ : Mat2 ℤ} (h : δ ∈ DC N) : δᵀ ∈ DC N := by
  obtain ⟨γ₁, γ₂, rfl⟩ := h
  refine ⟨trSL γ₂, trSL γ₁, ?_⟩
  simp [Matrix.transpose_mul, gMat_transpose, Matrix.mul_assoc]

/-- **[C] `prop:cosets`(2) with (1): `Γ g Γ` is the disjoint union of exactly `m` right cosets, represented
by the transposes `δ_kᵀ`, `k < m`.** -/
theorem rightCosets {p ν : ℕ} (hp : p.Prime) (hν : 1 ≤ ν) {δ : Mat2 ℤ} (hδ : δ ∈ DC ((p : ℤ) ^ ν)) :
    ∃! k, k < mIdx p ν ∧ ∃ γ : SL(2, ℤ), δ = (γ : Mat2 ℤ) * (deltaMat p ν k)ᵀ := by
  have tr : ∀ k, (∃ γ : SL(2, ℤ), δ = (γ : Mat2 ℤ) * (deltaMat p ν k)ᵀ) ↔
      ∃ γ : SL(2, ℤ), δᵀ = deltaMat p ν k * γ := by
    intro k; constructor
    · rintro ⟨γ, rfl⟩; exact ⟨trSL γ, by simp [Matrix.transpose_mul]⟩
    · rintro ⟨γ, hγ⟩
      refine ⟨trSL γ, ?_⟩
      have := congrArg Matrix.transpose hγ
      simpa [Matrix.transpose_mul] using this
  simp_rw [tr]
  exact leftCosets hp hν (transpose_mem_DC hδ)

/-! ## `ν = 1`: the explicit right cosets `β_b`, `β_∞` -/

/-- `β_b = [[1, b], [0, p]]` for `b < p`, and `β_∞ = diag(p, 1)` at the index `b = p`. -/
def betaMat (p b : ℕ) : Mat2 ℤ := if b < p then !![1, (b : ℤ); 0, (p : ℤ)] else !![(p : ℤ), 0; 0, 1]

lemma betaMat_hermite (p b : ℕ) (hp : 0 < p) (hb : b ≤ p) :
    ∃ a b' d : ℤ, betaMat p b = !![a, b'; 0, d] ∧ 0 < a ∧ 0 < d ∧ 0 ≤ b' ∧ b' < d ∧
      ((b < p ∧ a = 1 ∧ b' = b) ∨ (b = p ∧ a = p)) := by
  unfold betaMat
  split_ifs with h
  · exact ⟨1, b, p, rfl, one_pos, by exact_mod_cast hp, by positivity, by exact_mod_cast h, Or.inl ⟨h, rfl, rfl⟩⟩
  · exact ⟨p, 0, 1, rfl, by exact_mod_cast hp, one_pos, le_refl _, one_pos, Or.inr ⟨by omega, rfl⟩⟩

/-- **[C] `prop:cosets`(1) at `ν = 1`:** `Γ g Γ = ⊔_{0 ≤ b < p} Γ β_b ⊔ Γ β_∞`, each coset once. -/
theorem rightCosets_nu_one {p : ℕ} (hp : p.Prime) {δ : Mat2 ℤ} (hδ : δ ∈ DC ((p : ℤ) ^ 1)) :
    ∃! b, b ≤ p ∧ ∃ γ : SL(2, ℤ), δ = (γ : Mat2 ℤ) * betaMat p b := by
  have hp0 : (0 : ℤ) < p := by exact_mod_cast hp.pos
  have hdet := ((mem_DC_iff hp δ).1 hδ).1
  rw [pow_one] at hdet
  obtain ⟨γ, a, b, d, hH, ha, hd, hb, hbd⟩ := hermite_exists δ (by rw [hdet]; exact hp0)
  have had : a * d = p := by
    have := congrArg Matrix.det hH
    rw [Matrix.det_mul, Matrix.SpecialLinearGroup.det_coe, hdet, one_mul, Matrix.det_fin_two_of] at this
    linarith
  have hδeq : δ = ((γ⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * !![a, b; 0, d] := by rw [← hH, inv_mul_coe_cancel_left]
  have hadvd : a.natAbs ∣ p := ⟨d.natAbs, by rw [← Int.natAbs_mul, had]; simp⟩
  have exist : ∃ b0, b0 ≤ p ∧ ∃ γ' : SL(2, ℤ), δ = (γ' : Mat2 ℤ) * betaMat p b0 := by
    rcases (Nat.dvd_prime hp).1 hadvd with h1 | h1
    · have ha1 : a = 1 := by omega
      subst ha1
      have hdp : d = p := by linarith
      subst hdp
      refine ⟨b.toNat, by omega, γ⁻¹, ?_⟩
      have hbp : b.toNat < p := by omega
      rw [hδeq, betaMat, if_pos hbp, Int.toNat_of_nonneg hb]
    · have hap : a = p := by omega
      subst hap
      have hd1 : d = 1 := by nlinarith
      subst hd1
      have hb0 : b = 0 := by omega
      subst hb0
      refine ⟨p, le_refl _, γ⁻¹, ?_⟩
      rw [hδeq, betaMat, if_neg (lt_irrefl _)]
  obtain ⟨b0, hb0, γ0, h0⟩ := exist
  refine ⟨b0, ⟨hb0, γ0, h0⟩, ?_⟩
  rintro b1 ⟨hb1, γ1, h1⟩
  obtain ⟨a₀, c₀, d₀, e₀, ha₀, hd₀, hc₀, hcd₀, k₀⟩ := betaMat_hermite p b0 hp.pos hb0
  obtain ⟨a₁, c₁, d₁, e₁, ha₁, hd₁, hc₁, hcd₁, k₁⟩ := betaMat_hermite p b1 hp.pos hb1
  have hrel : ((γ0⁻¹ * γ1 : SL(2, ℤ)) : Mat2 ℤ) * betaMat p b1 = betaMat p b0 := by
    rw [Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc, ← h1, h0, inv_mul_coe_cancel_left]
  rw [e₀, e₁] at hrel
  obtain ⟨r1, r2, r3⟩ := hermite_unique _ hrel ha₁ hd₁ hc₁ hcd₁ ha₀ hd₀ hc₀ hcd₀
  rcases k₀ with ⟨h0p, ha0, hb0'⟩ | ⟨h0p, ha0⟩ <;> rcases k₁ with ⟨h1p, ha1, hb1'⟩ | ⟨h1p, ha1⟩
  · have : (b1 : ℤ) = b0 := by rw [← hb1', ← hb0', r2]
    exact_mod_cast this
  · have : (p : ℤ) = 1 := by rw [← ha1, r1, ha0]
    have := hp.one_lt; omega
  · have : (p : ℤ) = 1 := by rw [← ha0, ← r1, ha1]
    have := hp.one_lt; omega
  · omega

end GL2.Cosets
