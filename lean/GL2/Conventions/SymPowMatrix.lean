import GL2.Conventions.SymPow

namespace GL2
open MvPolynomial

variable {R : Type*} [CommRing R]

/-- The exponent vector of `u_j = x^{n-j} y^j`. -/
noncomputable def mono (n j : ℕ) : Fin 2 →₀ ℕ := Finsupp.single 0 (n - j) + Finsupp.single 1 j

@[simp] lemma mono_zero (n j : ℕ) : mono n j 0 = n - j := by simp [mono]
@[simp] lemma mono_one (n j : ℕ) : mono n j 1 = j := by simp [mono]

lemma mono_degree {n j : ℕ} (hj : j ≤ n) : (mono n j).degree = n := by
  rw [Finsupp.degree_eq_sum, Fin.sum_univ_two, mono_zero, mono_one]; omega

lemma u_eq_monomial (n j : ℕ) : (u n j : MvPolynomial (Fin 2) R) = monomial (mono n j) 1 := by
  simp [u, mono, X_pow_eq_monomial, monomial_mul]

lemma symBasis_repr (n : ℕ) (v : SymN R n) (d : symIdx n) :
    (symBasis n).repr v d = coeff (d : Fin 2 →₀ ℕ) (v : MvPolynomial (Fin 2) R) := by
  rfl

lemma mono_injective {n i j : ℕ} (h : mono n i = mono n j) : i = j := by
  simpa using congrArg (fun d => d 1) h

lemma eq_mono_of_degree {n : ℕ} (d : Fin 2 →₀ ℕ) (hd : d.degree = n) : d = mono n (d 1) := by
  rw [Finsupp.degree_eq_sum, Fin.sum_univ_two] at hd
  ext t; fin_cases t <;> (simp; try omega)

/-- `symIdx n ≃ Fin (n+1)`, `d ↦ d 1` (the exponent of `y`). -/
noncomputable def symIdxEquiv (n : ℕ) : symIdx n ≃ Fin (n + 1) where
  toFun d := ⟨d.1 1, by
    have : (d.1).degree = n := d.2; rw [Finsupp.degree_eq_sum, Fin.sum_univ_two] at this; omega⟩
  invFun j := ⟨mono n j, mono_degree (Nat.lt_succ_iff.mp j.2)⟩
  left_inv d := Subtype.ext (eq_mono_of_degree d.1 d.2).symm
  right_inv j := Fin.ext (by simp)

/-- The basis `(u_0, …, u_n)` of `Sym^n(R²)`, `u_j = x^{n-j} y^j`, indexed by `Fin (n+1)`. -/
noncomputable def uBasis (n : ℕ) : Module.Basis (Fin (n + 1)) R (SymN R n) :=
  (symBasis n).reindex (symIdxEquiv n)

lemma uBasis_repr (n : ℕ) (v : SymN R n) (i : Fin (n + 1)) :
    (uBasis n).repr v i = coeff (mono n i) (v : MvPolynomial (Fin 2) R) := by
  rw [uBasis, Module.Basis.repr_reindex_apply, symBasis_repr]; rfl

lemma uBasis_coe (n : ℕ) (j : Fin (n + 1)) :
    ((uBasis n j : SymN R n) : MvPolynomial (Fin 2) R) = u n j := by
  rw [u_eq_monomial]
  ext d
  have h := uBasis_repr (R := R) n (uBasis n j)
  simp only [Module.Basis.repr_self] at h
  by_cases hd : d.degree = n
  · rw [eq_mono_of_degree d hd]
    have hk : d 1 < n + 1 := by
      rw [Finsupp.degree_eq_sum, Fin.sum_univ_two] at hd; omega
    rw [← h ⟨d 1, hk⟩, coeff_monomial, Finsupp.single_apply]
    by_cases e : j = ⟨d 1, hk⟩
    · subst e; simp
    · rw [if_neg e, if_neg]; intro h'; exact e (Fin.ext (mono_injective h'))
  · have hv := (mem_homogeneousSubmodule _ _).1 (uBasis (R := R) n j).2
    rw [coeff_monomial, if_neg, hv.coeff_eq_zero]
    · exact hd
    · rintro rfl; exact hd (mono_degree (Nat.lt_succ_iff.mp j.2))

/-- The coordinate matrix of `ρ_n(A)` in the basis `(u_j)`: column `j` holds the coefficients of
`(a x + c y)^{n-j} (b x + d y)^j` ([C] `eq:rhondef`). -/
noncomputable def rhoMat (n : ℕ) (A : Mat2 R) : Matrix (Fin (n + 1)) (Fin (n + 1)) R :=
  LinearMap.toMatrix (uBasis n) (uBasis n) (rho n A)

lemma rhoMat_apply (n : ℕ) (A : Mat2 R) (i j : Fin (n + 1)) :
    rhoMat n A i j = coeff (mono n i) (subst A (u n j)) := by
  rw [rhoMat, LinearMap.toMatrix_apply, uBasis_repr, rho_apply_coe, uBasis_coe]

theorem rhoMat_mul (n : ℕ) (A B : Mat2 R) : rhoMat n (A * B) = rhoMat n A * rhoMat n B := by
  rw [rhoMat, rho_mul, LinearMap.toMatrix_mul]; rfl

theorem rhoMat_one (n : ℕ) : rhoMat n (1 : Mat2 R) = 1 := by
  rw [rhoMat, rho_one, LinearMap.toMatrix_one]

theorem rhoMat_map {S : Type*} [CommRing S] (f : R →+* S) (n : ℕ) (A : Mat2 R) :
    rhoMat n (A.map f) = (rhoMat n A).map f := by
  ext i j
  rw [Matrix.map_apply, rhoMat_apply, rhoMat_apply, ← coeff_map, map_subst]
  congr 2
  simp [u]

theorem det_rho_eq_det_rhoMat (n : ℕ) (A : Mat2 R) : LinearMap.det (rho n A) = (rhoMat n A).det :=
  (LinearMap.det_toMatrix (uBasis n) (rho n A)).symm

end GL2
