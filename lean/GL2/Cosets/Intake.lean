import GL2.Cosets.DoubleCoset

/-!
# The G2 intake (L18-1, partial): `prop:cosets`(3) as a group isomorphism, `prop:cosets`(5) for any transversal

[GT] `prop:cosets-3-iso`, `prop:cosets-5-any`, mirroring [C] v1.17.0 `prop:cosets`(3) (L17-3 MINOR 2, group-isomorphism
clause) and `prop:cosets`(5) for an arbitrary transversal `s_1, …, s_m` (F3 finding 9).

* `conj_Gamma0_mulEquiv` -- `γ ↦ g⁻¹ γ g` is a group isomorphism `Γ'' = Γ₀(N) ≃* Γ' = Γ⁰(N)`, `g = diag(1, N)`,
  `N ≠ 0`, written integrally (`γ g = g φ(γ)`; no determinant-`N` inverse).
* `leftCosets_any` -- for **any** transversal `s : Fin m → SL(2, ℤ)` of `Γ₀(N)\Γ` (not only L17-3's `s_k`), the matrices
  `δ_i = s_i⁻¹ g` represent `Γ g Γ / Γ`, each coset exactly once.
-/

namespace GL2.Cosets

open Matrix CongruenceSubgroup
open scoped MatrixGroups

lemma gMat_mul_cancel {N : ℤ} (hN : N ≠ 0) {X Y : Mat2 ℤ} (h : gMat N * X = gMat N * Y) : X = Y := by
  ext i j
  have := congrFun (congrFun h i) j
  fin_cases i <;> simp [gMat, Matrix.mul_apply, Fin.sum_univ_two] at this ⊢
  · exact this
  · rcases this with h' | h'
    · exact h'
    · exact absurd h' hN

lemma mul_gMat_cancel {N : ℤ} (hN : N ≠ 0) {X Y : Mat2 ℤ} (h : X * gMat N = Y * gMat N) : X = Y := by
  ext i j
  have := congrFun (congrFun h i) j
  fin_cases j <;> simp [gMat, Matrix.mul_apply, Fin.sum_univ_two] at this ⊢
  · exact this
  · rcases this with h' | h'
    · exact h'
    · exact absurd h' hN

section Iso

variable (N : ℕ) (hN : N ≠ 0)

lemma conj_exists (A : Gamma0 N) :
    ∃ γ' : SL(2, ℤ), ((A : SL(2, ℤ)) : Mat2 ℤ) * gMat N = gMat N * (γ' : Mat2 ℤ) :=
  (conj_Gamma0_iff N A).2 ((mem_Gamma0_iff_dvd N A).1 A.2)

/-- `g⁻¹ A g` for `A ∈ Γ₀(N)`. -/
noncomputable def conjFun (A : Gamma0 N) : SL(2, ℤ) := Classical.choose (conj_exists N A)

lemma conjFun_spec (A : Gamma0 N) :
    ((A : SL(2, ℤ)) : Mat2 ℤ) * gMat N = gMat N * (conjFun N A : Mat2 ℤ) :=
  Classical.choose_spec (conj_exists N A)

lemma conjFun_mem (A : Gamma0 N) : conjFun N A ∈ Gamma0Up N :=
  (conj_Gamma0Up_iff N (conjFun N A)).1 ⟨A, (conjFun_spec N A).symm⟩

include hN in
lemma conjFun_eq {A : Gamma0 N} {B : SL(2, ℤ)} (h : ((A : SL(2, ℤ)) : Mat2 ℤ) * gMat N = gMat N * (B : Mat2 ℤ)) :
    conjFun N A = B :=
  Subtype.ext (gMat_mul_cancel (by exact_mod_cast hN) ((conjFun_spec N A).symm.trans h))

/-- The conjugation `Γ₀(N) → Γ⁰(N)` as a monoid homomorphism. -/
noncomputable def conjHom : Gamma0 N →* Gamma0Up N where
  toFun A := ⟨conjFun N A, conjFun_mem N A⟩
  map_one' := Subtype.ext (conjFun_eq N hN (by simp))
  map_mul' A B := by
    apply Subtype.ext
    apply conjFun_eq N hN
    simp only [Subgroup.coe_mul, Matrix.SpecialLinearGroup.coe_mul]
    rw [Matrix.mul_assoc, conjFun_spec N B, ← Matrix.mul_assoc, conjFun_spec N A, Matrix.mul_assoc]

/-- **`prop:cosets`(3), the group isomorphism**: `γ ↦ g⁻¹ γ g` is an isomorphism `Γ₀(N) ≃* Γ⁰(N)`
(`γ g = g φ(γ)`). -/
noncomputable def conj_Gamma0_mulEquiv : Gamma0 N ≃* Gamma0Up N :=
  MulEquiv.ofBijective (conjHom N hN) ⟨fun A B h => by
    have h' : conjFun N A = conjFun N B := congrArg Subtype.val h
    apply Subtype.ext; apply Subtype.ext
    apply mul_gMat_cancel (N := N) (by exact_mod_cast hN)
    rw [conjFun_spec, conjFun_spec, h'],
  fun B => by
    obtain ⟨γ', hγ'⟩ := (conj_Gamma0Up_iff N B).2 B.2
    have hmem : γ' ∈ Gamma0 N := (mem_Gamma0_iff_dvd N γ').2 ((conj_Gamma0_iff N γ').1 ⟨B, hγ'.symm⟩)
    exact ⟨⟨γ', hmem⟩, Subtype.ext (conjFun_eq N hN hγ'.symm)⟩⟩

theorem conj_Gamma0_mulEquiv_spec (A : Gamma0 N) :
    ((A : SL(2, ℤ)) : Mat2 ℤ) * gMat N = gMat N * (((conj_Gamma0_mulEquiv N hN A : Gamma0Up N) : SL(2, ℤ)) : Mat2 ℤ) :=
  conjFun_spec N A

end Iso

/-- **`prop:cosets`(5) for any transversal** (F3 finding 9): if `s : Fin m → SL(2, ℤ)` meets every right coset
`Γ₀(N) γ` exactly once, then every `δ ∈ Γ g Γ` lies in `s_i⁻¹ g Γ` for exactly one `i`. -/
theorem leftCosets_any {p ν m : ℕ} (s : Fin m → SL(2, ℤ))
    (hs : ∀ γ : SL(2, ℤ), ∃! i, γ * (s i)⁻¹ ∈ Gamma0 (p ^ ν)) {δ : Mat2 ℤ} (hδ : δ ∈ DC ((p : ℤ) ^ ν)) :
    ∃! i, ∃ γ : SL(2, ℤ), δ = (((s i)⁻¹ : SL(2, ℤ)) : Mat2 ℤ) * gMat ((p : ℤ) ^ ν) * γ := by
  obtain ⟨γ₁, γ₂, rfl⟩ := hδ
  obtain ⟨i, hmem, huniq⟩ := hs γ₁⁻¹
  set h := γ₁⁻¹ * (s i)⁻¹ with hh
  have hinv : h⁻¹ ∈ Gamma0 (p ^ ν) := inv_mem hmem
  obtain ⟨u, hu⟩ := (conj_Gamma0 p ν h⁻¹).1 hinv
  refine ⟨i, ⟨u * γ₂, ?_⟩, ?_⟩
  · have hγ₁ : γ₁ = (s i)⁻¹ * h⁻¹ := by rw [hh]; group
    rw [hγ₁, Matrix.SpecialLinearGroup.coe_mul, Matrix.SpecialLinearGroup.coe_mul,
      Matrix.mul_assoc, Matrix.mul_assoc, ← Matrix.mul_assoc _ (gMat _), hu]
    simp [Matrix.mul_assoc]
  · rintro i' ⟨γ', hγ'⟩
    have heq : ((s i' * γ₁ : SL(2, ℤ)) : Mat2 ℤ) * gMat ((p : ℤ) ^ ν) =
        gMat ((p : ℤ) ^ ν) * ((γ' * γ₂⁻¹ : SL(2, ℤ)) : Mat2 ℤ) := by
      have := congrArg (fun X => ((s i' : SL(2, ℤ)) : Mat2 ℤ) * X * ((γ₂⁻¹ : SL(2, ℤ)) : Mat2 ℤ)) hγ'
      simp only [Matrix.mul_assoc, mul_inv_coe, Matrix.mul_one, mul_inv_coe_cancel_left] at this
      rw [Matrix.SpecialLinearGroup.coe_mul, Matrix.SpecialLinearGroup.coe_mul, Matrix.mul_assoc, this]
    have hmem' : s i' * γ₁ ∈ Gamma0 (p ^ ν) := (conj_Gamma0 p ν _).2 ⟨_, heq⟩
    have : γ₁⁻¹ * (s i')⁻¹ ∈ Gamma0 (p ^ ν) := by
      have := inv_mem hmem'; rwa [_root_.mul_inv_rev] at this
    exact huniq i' this

end GL2.Cosets
