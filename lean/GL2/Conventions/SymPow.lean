import Mathlib

/-!
# C1: the symmetric power `ρ_n`, the column convention ([C] `def:rhon`, `prop:symconv`)

[GT] `conv:symconv`, mirroring [C] v1.15.0 `def:rhon`, `eq:rhondef`, `prop:symconv`(2)(3),
`def:semigroup`.

`W = R²` with ordered basis `(x, y) = (X 0, X 1)`; `A = !![a, b; c, d]` acts in the **column
convention** `A·x = a x + c y`, `A·y = b x + d y`, i.e. `A·e_j = ∑_t A t j • e_t`.  `ρ_n(A)` is the
algebra substitution `x ↦ a x + c y`, `y ↦ b x + d y` restricted to the homogeneous polynomials of
degree `n` (= `Sym^n W`), with the basis `u_j = x^{n-j} y^j`.  No transpose, inverse, determinant
twist or contragredient is involved.

This is literally the C++ `ComputeSymmetricPowerMatrixFM` (column `j` = coefficients of
`(a x + c y)^{n-j} (b x + d y)^j`), and it is FLT's `HeckeEis.binarySubst` / `binaryFormRep`
(L16-1 survey §4.1).
-/

namespace GL2

open MvPolynomial

/-- `2 × 2` matrices over `R`. -/
abbrev Mat2 (R : Type*) := Matrix (Fin 2) (Fin 2) R

variable {R : Type*} [CommRing R]

/-- The image of the basis vector `e_j` under `A` in the column convention: `∑_t A t j • X t`. -/
noncomputable def linForm (A : Mat2 R) (j : Fin 2) : MvPolynomial (Fin 2) R :=
  ∑ t, C (A t j) * X t

/-- The substitution algebra map `x ↦ a x + c y`, `y ↦ b x + d y` on `R[x, y]`. -/
noncomputable def subst (A : Mat2 R) : MvPolynomial (Fin 2) R →ₐ[R] MvPolynomial (Fin 2) R :=
  bind₁ (linForm A)

@[simp] lemma subst_X (A : Mat2 R) (j : Fin 2) : subst A (X j) = linForm A j := by
  simp [subst]

lemma linForm_zero (A : Mat2 R) : linForm A 0 = C (A 0 0) * X 0 + C (A 1 0) * X 1 := by
  simp [linForm, Fin.sum_univ_two]

lemma linForm_one (A : Mat2 R) : linForm A 1 = C (A 0 1) * X 0 + C (A 1 1) * X 1 := by
  simp [linForm, Fin.sum_univ_two]

/-- `prop:symconv`(2), polynomial level: `subst (A * B) = subst A ∘ subst B`. -/
theorem subst_mul (A B : Mat2 R) : subst (A * B) = (subst A).comp (subst B) := by
  apply MvPolynomial.algHom_ext
  intro j
  simp only [AlgHom.comp_apply, subst_X]
  simp only [linForm, map_sum, map_mul, algHom_C, subst_X, Matrix.mul_apply, Finset.sum_mul,
    Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun s _ => Finset.sum_congr rfl fun t _ => ?_
  simp only [algebraMap_eq]
  ring

/-- `prop:symconv`(2): `subst 1 = id`. -/
theorem subst_one : subst (1 : Mat2 R) = AlgHom.id R _ := by
  apply MvPolynomial.algHom_ext
  intro j
  rw [subst_X, AlgHom.id_apply]
  fin_cases j <;> simp [linForm, Matrix.one_apply]

lemma linForm_isHomogeneous (A : Mat2 R) (j : Fin 2) : (linForm A j).IsHomogeneous 1 := by
  unfold linForm
  exact IsHomogeneous.sum _ _ _ fun t _ => isHomogeneous_C_mul_X _ _

lemma subst_isHomogeneous (A : Mat2 R) {n : ℕ} {φ : MvPolynomial (Fin 2) R}
    (h : φ.IsHomogeneous n) : (subst A φ).IsHomogeneous n := by
  have := h.aeval (linForm A) (linForm_isHomogeneous A)
  rw [one_mul] at this
  exact this

/-- `Sym^n(R²)`: the homogeneous polynomials of degree `n` in `x = X 0`, `y = X 1`. -/
abbrev SymN (R : Type*) [CommRing R] (n : ℕ) := homogeneousSubmodule (Fin 2) R n

/-- `ρ_n(A)` ([C] `def:rhon`): the restriction of `subst A` to `Sym^n`. -/
noncomputable def rho (n : ℕ) (A : Mat2 R) : SymN R n →ₗ[R] SymN R n :=
  (subst A).toLinearMap.restrict fun _ hφ =>
    (mem_homogeneousSubmodule _ _).2 (subst_isHomogeneous A ((mem_homogeneousSubmodule _ _).1 hφ))

@[simp] lemma rho_apply_coe (n : ℕ) (A : Mat2 R) (v : SymN R n) :
    ((rho n A v : SymN R n) : MvPolynomial (Fin 2) R) = subst A v := rfl

/-- **`prop:symconv`(2)**: `ρ_n` is a monoid homomorphism on all of `Mat₂(R)` (singular matrices
included): `ρ_n(A B) = ρ_n(A) ρ_n(B)`. -/
theorem rho_mul (n : ℕ) (A B : Mat2 R) : rho n (A * B) = rho n A * rho n B := by
  ext v
  simp [subst_mul]

/-- **`prop:symconv`(2)**: `ρ_n(I) = I`. -/
theorem rho_one (n : ℕ) : rho n (1 : Mat2 R) = 1 := by
  ext v
  simp [subst_one]

/-- `ρ_n` as a monoid homomorphism `Mat₂(R) →* End(Sym^n)`. -/
noncomputable def rhoHom (n : ℕ) : Mat2 R →* Module.End R (SymN R n) where
  toFun := rho n
  map_one' := rho_one n
  map_mul' := rho_mul n

/-- The basis monomial `u_j = x^{n-j} y^j`. -/
noncomputable def u (n j : ℕ) : MvPolynomial (Fin 2) R := X 0 ^ (n - j) * X 1 ^ j

lemma u_isHomogeneous {n j : ℕ} (hj : j ≤ n) : (u n j : MvPolynomial (Fin 2) R).IsHomogeneous n := by
  have := (isHomogeneous_X_pow (R := R) (0 : Fin 2) (n - j)).mul (isHomogeneous_X_pow (1 : Fin 2) j)
  simpa [u, Nat.sub_add_cancel hj] using this

/-- **`eq:rhondef`** (the defining formula, column `j`):
`ρ_n(A) u_j = (a x + c y)^{n-j} (b x + d y)^j`. -/
theorem subst_u (A : Mat2 R) (n j : ℕ) :
    subst A (u n j) = (C (A 0 0) * X 0 + C (A 1 0) * X 1) ^ (n - j) *
      (C (A 0 1) * X 0 + C (A 1 1) * X 1) ^ j := by
  simp [u, linForm_zero, linForm_one]

/-- A scalar matrix acts on `Sym^n` by `λ^n` (used by `prop:pairing`(4) and `prop:nebcorr`(3)). -/
theorem subst_smul_one_of_isHomogeneous (r : R) {n : ℕ} {φ : MvPolynomial (Fin 2) R}
    (h : φ.IsHomogeneous n) : subst (r • (1 : Mat2 R)) φ = r ^ n • φ := by
  have hlin : ∀ j, linForm (r • (1 : Mat2 R)) j = C r * X j := by
    intro j; fin_cases j <;> simp [linForm, Fin.sum_univ_two]
  rw [φ.as_sum]
  simp only [map_sum, Finset.smul_sum]
  refine Finset.sum_congr rfl fun d hd => ?_
  have hdeg : d.degree = n := by
    have := h (mem_support_iff.mp hd); rwa [Finsupp.degree_eq_weight_one]
  rw [monomial_eq, map_mul, algHom_C, Finsupp.prod, map_prod]
  simp only [map_pow, subst_X, hlin, mul_pow, Finset.prod_mul_distrib]
  rw [Finset.prod_pow_eq_pow_sum, smul_eq_C_mul, algebraMap_eq]
  have : (∑ i ∈ d.support, d i) = n := by rw [← hdeg, Finsupp.degree]; rfl
  rw [this, map_pow]
  ring

theorem rho_smul_one (n : ℕ) (r : R) : rho n (r • (1 : Mat2 R)) = r ^ n • (1 : Module.End R _) := by
  refine LinearMap.ext fun v => Subtype.ext ?_
  simp only [rho_apply_coe, LinearMap.smul_apply, Module.End.one_apply, Submodule.coe_smul]
  exact subst_smul_one_of_isHomogeneous r ((mem_homogeneousSubmodule _ _).1 v.2)

/-- Base change: `ρ_n` commutes with every ring map on the coefficients (the `ρ_n mod ℓ`, `mod ℓ^K`
reading of `prop:symconv`(1)). -/
theorem map_subst {S : Type*} [CommRing S] (f : R →+* S) (A : Mat2 R) (φ : MvPolynomial (Fin 2) R) :
    MvPolynomial.map f (subst A φ) = subst (A.map f) (MvPolynomial.map f φ) := by
  induction φ using MvPolynomial.induction_on with
  | C a => simp
  | add p q hp hq => simp [hp, hq]
  | mul_X p j hp =>
    simp only [map_mul, hp, subst_X, map_X]
    congr 1
    simp [linForm, Matrix.map_apply]

/-! ### The monomial basis and freeness -/

/-- The index set of the monomial basis: exponent vectors of degree `n` (in bijection with `u_j`, `j = d 1`). -/
abbrev symIdx (n : ℕ) := {d : Fin 2 →₀ ℕ | d.degree = n}

/-- The monomial basis of `Sym^n(R²)` (the `u_j`, indexed by exponent vectors). -/
noncomputable def symBasis (n : ℕ) : Module.Basis (symIdx n) R (SymN R n) :=
  (basisRestrictSupport R (symIdx n)).map
    (LinearEquiv.ofEq _ _ (homogeneousSubmodule_eq_finsupp_supported (Fin 2) R n).symm)

instance (n : ℕ) : Module.Free R (SymN R n) := Module.Free.of_basis (symBasis n)

instance (n : ℕ) : Finite (symIdx n) := (Finsupp.finite_of_degree_eq n).to_subtype

instance (n : ℕ) : Module.Finite R (SymN R n) := Module.Finite.of_basis (symBasis n)

/-! `lem:detsym` (`GL2.det_rho`) and the `prop:symconv`(3) bound (`GL2.rho_coeff_bound`) are proved in
`GL2.Conventions.SymPowDet` (L16-6), on the coordinate matrix `GL2.rhoMat` of `GL2.Conventions.SymPowMatrix`. -/

end GL2
