import GL2.Cosets.DoubleCoset

/-!
# M5-E (slot 22): the elementary-divisor enumeration of the double cosets of determinant `p^ν`

[GT] `gt_part_M5.tex`, rows `lem:m5-dc-classify`, `lem:m5-dc-unique`, `thm:m5-dc-enum`, backing [C] `eq:fullop`
(which keeps **no** `\LEAN` marker, owner ruling Q5-4) through [C] `prop:fullassembly`'s index-set paragraph:

* `dc_classify`: every `δ ∈ M_2(ℤ)` with `det δ = p^ν` is `γ₁ · p^j diag(1, p^{ν-2j}) · γ₂` with `2j ≤ ν`,
  `γ₁, γ₂ ∈ SL(2, ℤ)` (the content of `δ` is `p^j`; the primitive case is `GL2.Cosets.mem_DC_iff`, [GT] `prop:cosets` (1));
* `dc_unique`: `p^j diag(1, p^μ)` and `p^{j'} diag(1, p^{μ'})` are `SL(2, ℤ)`-doubly equivalent only if `j = j'`, `μ = μ'`
  (the content is `Γ`-biinvariant, the determinant fixes `μ`);
* `dc_enum`: the two together — the double cosets of determinant `p^ν` are exactly `Γ p^j diag(1, p^{ν-2j}) Γ`,
  `0 ≤ 2j ≤ ν`, pairwise distinct.
-/

namespace GL2.M5

open Matrix GL2.Cosets
open scoped MatrixGroups

/-- `p^j · diag(1, p^μ)`, the elementary-divisor representative of content `p^j` and determinant `p^{2j+μ}`. -/
def dMat (p j μ : ℕ) : Mat2 ℤ := ((p : ℤ) ^ j) • gMat ((p : ℤ) ^ μ)

/-- Two integral matrices lie in one `SL(2, ℤ)` double coset: `A = γ₁ B γ₂`. -/
def DCRel (A B : Mat2 ℤ) : Prop := ∃ γ₁ γ₂ : SL(2, ℤ), A = (γ₁ : Mat2 ℤ) * B * γ₂

lemma DCRel.refl (A : Mat2 ℤ) : DCRel A A := ⟨1, 1, by simp⟩

lemma DCRel.symm {A B : Mat2 ℤ} (h : DCRel A B) : DCRel B A := by
  obtain ⟨γ₁, γ₂, rfl⟩ := h
  refine ⟨γ₁⁻¹, γ₂⁻¹, ?_⟩
  rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, GL2.Cosets.inv_mul_coe, Matrix.one_mul,
    Matrix.mul_assoc, GL2.Cosets.mul_inv_coe, Matrix.mul_one]

lemma DCRel.trans {A B C : Mat2 ℤ} (h : DCRel A B) (h' : DCRel B C) : DCRel A C := by
  obtain ⟨γ₁, γ₂, rfl⟩ := h
  obtain ⟨γ₃, γ₄, rfl⟩ := h'
  exact ⟨γ₁ * γ₃, γ₄ * γ₂, by simp only [Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc]⟩

lemma dMat_zero (p μ : ℕ) : dMat p 0 μ = gMat ((p : ℤ) ^ μ) := by simp [dMat]

lemma dMat_succ (p j μ : ℕ) : dMat p (j + 1) μ = (p : ℤ) • dMat p j μ := by
  simp only [dMat, smul_smul, pow_succ, mul_comm]

lemma dMat_det (p j μ : ℕ) : (dMat p j μ).det = (p : ℤ) ^ (2 * j + μ) := by
  simp only [dMat, Matrix.det_smul, Fintype.card_fin, gMat_det]
  ring

lemma det_DCRel {A B : Mat2 ℤ} (h : DCRel A B) : A.det = B.det := by
  obtain ⟨γ₁, γ₂, rfl⟩ := h
  rw [Matrix.det_mul, Matrix.det_mul, Matrix.SpecialLinearGroup.det_coe, Matrix.SpecialLinearGroup.det_coe]; ring

lemma dvd_entries_DCRel {e : ℤ} {A B : Mat2 ℤ} (h : DCRel A B) (hB : ∀ i j, e ∣ B i j) : ∀ i j, e ∣ A i j := by
  obtain ⟨γ₁, γ₂, rfl⟩ := h
  exact dvd_entries_mul_right (dvd_entries_mul_left hB _) _

lemma dvd_entries_dMat (p j μ : ℕ) : ∀ a b, (p : ℤ) ^ j ∣ dMat p j μ a b := by
  intro a b
  simp only [dMat, Matrix.smul_apply, smul_eq_mul]
  exact dvd_mul_right _ _

/-- A matrix all of whose entries are divisible by `p` is `p •` a matrix. -/
lemma eq_smul_of_dvd {p : ℤ} {δ : Mat2 ℤ} (h : ∀ i j, p ∣ δ i j) :
    δ = p • Matrix.of (fun i j => δ i j / p) := by
  ext i j
  simp only [Matrix.smul_apply, Matrix.of_apply, smul_eq_mul]
  exact (Int.mul_ediv_cancel' (h i j)).symm

/-- A non-primitive matrix of determinant `p^ν` has all entries divisible by `p`. -/
lemma dvd_entries_of_not_primitive {p ν : ℕ} (hp : p.Prime) {δ : Mat2 ℤ} (hdet : δ.det = (p : ℤ) ^ ν)
    (hprim : ¬ GL2.Cosets.IsPrimitive δ) : ∀ i j, (p : ℤ) ∣ δ i j := by
  have hp' : Prime (p : ℤ) := Nat.prime_iff_prime_int.mp hp
  simp only [GL2.Cosets.IsPrimitive, not_forall] at hprim
  obtain ⟨e, he, hunit⟩ := hprim
  have hne : e.natAbs ≠ 1 := fun h => hunit (Int.isUnit_iff_natAbs_eq.2 h)
  obtain ⟨q, hq, hqe⟩ := Int.exists_prime_and_dvd hne
  have hq' : ∀ i j, q ∣ δ i j := fun i j => dvd_trans hqe (he i j)
  have hqq : q * q ∣ (p : ℤ) ^ ν := by
    rw [← hdet, Matrix.det_fin_two]
    exact dvd_sub (mul_dvd_mul (hq' 0 0) (hq' 1 1)) (mul_dvd_mul (hq' 0 1) (hq' 1 0))
  have hqp : q ∣ (p : ℤ) := hq.dvd_of_dvd_pow (dvd_trans (dvd_mul_right q q) hqq)
  have hpq : (p : ℤ) ∣ q := (hq.associated_of_dvd hp' hqp).symm.dvd
  exact fun i j => dvd_trans hpq (hq' i j)

/-- **M5-E (1), classification**: every integral `δ` with `det δ = p^ν` lies in `Γ p^j diag(1, p^{ν-2j}) Γ` for some
`j` with `2j ≤ ν`. -/
theorem dc_classify {p : ℕ} (hp : p.Prime) (ν : ℕ) :
    ∀ δ : Mat2 ℤ, δ.det = (p : ℤ) ^ ν → ∃ j, 2 * j ≤ ν ∧ DCRel δ (dMat p j (ν - 2 * j)) := by
  induction ν using Nat.strong_induction_on with
  | _ ν ih =>
    intro δ hdet
    by_cases hprim : GL2.Cosets.IsPrimitive δ
    · refine ⟨0, Nat.zero_le _, ?_⟩
      obtain ⟨γ₁, γ₂, h⟩ := (mem_DC_iff hp δ).2 ⟨hdet, hprim⟩
      refine ⟨γ₁, γ₂, ?_⟩
      rw [h, dMat_zero, Nat.mul_zero, Nat.sub_zero]
    · have hdvd := dvd_entries_of_not_primitive hp hdet hprim
      set δ' : Mat2 ℤ := Matrix.of (fun i j => δ i j / (p : ℤ)) with hδ'
      have hδ : δ = (p : ℤ) • δ' := eq_smul_of_dvd hdvd
      have hp0 : (p : ℤ) ≠ 0 := by exact_mod_cast hp.ne_zero
      have hdet' : (p : ℤ) ^ 2 * δ'.det = (p : ℤ) ^ ν := by
        rw [← hdet, hδ, Matrix.det_smul, Fintype.card_fin]
      have h2 : 2 ≤ ν := by
        have hd : (p : ℤ) ^ 2 ∣ (p : ℤ) ^ ν := ⟨δ'.det, hdet'.symm⟩
        have hd' : p ^ 2 ∣ p ^ ν := by exact_mod_cast hd
        exact (Nat.pow_dvd_pow_iff_le_right hp.one_lt).1 hd'
      have hdetδ' : δ'.det = (p : ℤ) ^ (ν - 2) := by
        apply mul_left_cancel₀ (pow_ne_zero 2 hp0)
        rw [hdet', ← pow_add]; congr 1; omega
      obtain ⟨j, hj, γ₁, γ₂, h⟩ := ih (ν - 2) (by omega) δ' hdetδ'
      refine ⟨j + 1, by omega, γ₁, γ₂, ?_⟩
      have hidx : ν - 2 * (j + 1) = ν - 2 - 2 * j := by omega
      rw [hδ, h, hidx, dMat_succ, Matrix.mul_smul, Matrix.smul_mul]

/-- **M5-E (2), distinctness**: `p^j diag(1, p^μ)` and `p^{j'} diag(1, p^{μ'})` in one double coset force `j = j'` and
`μ = μ'`. -/
theorem dc_unique {p : ℕ} (hp : p.Prime) {j μ j' μ' : ℕ} (h : DCRel (dMat p j μ) (dMat p j' μ')) :
    j = j' ∧ μ = μ' := by
  have hp0 : (p : ℤ) ≠ 0 := by exact_mod_cast hp.ne_zero
  have hpu : ¬ IsUnit (p : ℤ) := (Nat.prime_iff_prime_int.mp hp).not_isUnit
  have hle : ∀ {a b c d : ℕ}, DCRel (dMat p a b) (dMat p c d) → c ≤ a := by
    intro a b c d hr
    have h00 := dvd_entries_DCRel hr (dvd_entries_dMat p c d) 0 0
    simp only [dMat, gMat, Matrix.smul_apply, Matrix.of_apply, Matrix.cons_val', Matrix.cons_val_zero,
      Matrix.empty_val', Matrix.cons_val_fin_one, smul_eq_mul, mul_one] at h00
    exact (pow_dvd_pow_iff hp0 hpu).1 h00
  have hjj : j = j' := le_antisymm (hle h.symm) (hle h)
  refine ⟨hjj, ?_⟩
  have hd := det_DCRel h
  rw [dMat_det, dMat_det, hjj] at hd
  have h1 := (pow_dvd_pow_iff hp0 hpu).1 (dvd_of_eq hd)
  have h2 := (pow_dvd_pow_iff hp0 hpu).1 (dvd_of_eq hd.symm)
  omega

/-- **M5-E (3), the enumeration** ([C] `prop:fullassembly`, the index set of [C] `eq:fullop`): for `p` prime and every
`δ ∈ M_2(ℤ)` with `det δ = p^ν` there is exactly one `j` with `2j ≤ ν` and `δ ∈ Γ p^j diag(1, p^{ν-2j}) Γ`; each of these
representatives has determinant `p^ν`. -/
theorem dc_enum {p : ℕ} (hp : p.Prime) (ν : ℕ) :
    (∀ δ : Mat2 ℤ, δ.det = (p : ℤ) ^ ν → ∃! j, 2 * j ≤ ν ∧ DCRel δ (dMat p j (ν - 2 * j))) ∧
      ∀ j, 2 * j ≤ ν → (dMat p j (ν - 2 * j)).det = (p : ℤ) ^ ν := by
  refine ⟨fun δ hdet => ?_, fun j hj => ?_⟩
  · obtain ⟨j, hj, hr⟩ := dc_classify hp ν δ hdet
    refine ⟨j, ⟨hj, hr⟩, fun j' ⟨_, hr'⟩ => ?_⟩
    exact (dc_unique hp (hr'.symm.trans hr)).1
  · rw [dMat_det]; congr 1; omega

end GL2.M5
