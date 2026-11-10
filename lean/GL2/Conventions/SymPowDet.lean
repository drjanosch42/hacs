import GL2.Conventions.SymPowMatrix

/-!
# `lem:detsym`: `det ρ_n(A) = (det A)^{n(n+1)/2}` (L16-6)

[GT] `lem:detsym`, mirroring [C] v1.15.0 `lem:detsym`.  The route (L16-2's plan, with the
transvection–diagonal factorisation written out by hand instead of `Matrix.Pivot`):

1. `rhoMat` (the matrix of `ρ_n` in the basis `u_j`) is multiplicative and commutes with ring maps
   (`GL2.rhoMat_mul`, `GL2.rhoMat_map`);
2. `ρ_n` of a lower unitriangular matrix is lower unitriangular (`det = 1`), `ρ_n` of a diagonal matrix
   is diagonal with entries `x^{n-j} y^j`, and the upper unitriangular case is its conjugate by the swap;
3. over a field, `A = L · D · U` whenever `a ≠ 0`;
4. descent: the generic matrix over `ℤ[a,b,c,d]` has `a ≠ 0` in the fraction field, and every matrix
   over every commutative ring is a specialisation of it.
-/

namespace GL2
open MvPolynomial Matrix

variable {R : Type*} [CommRing R]

/-- `!![1, 0; c, 1]`. -/
def lowerT (c : R) : Mat2 R := !![1, 0; c, 1]
/-- `!![1, c; 0, 1]`. -/
def upperT (c : R) : Mat2 R := !![1, c; 0, 1]
/-- `!![x, 0; 0, y]`. -/
def diagM (x y : R) : Mat2 R := !![x, 0; 0, y]
/-- `!![0, 1; 1, 0]`. -/
def swapM : Mat2 R := !![0, 1; 1, 0]

lemma coeff_lin_pow (c : R) (k : ℕ) :
    coeff (Finsupp.single 0 k) ((X 0 + C c * X 1 : MvPolynomial (Fin 2) R) ^ k) = 1 := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [pow_succ, mul_add, coeff_add, coeff_mul_X', ← mul_assoc, coeff_mul_X']
    have h1 : (Finsupp.single (0 : Fin 2) (k + 1) - Finsupp.single 0 1) = Finsupp.single 0 k := by
      ext t; fin_cases t <;> simp
    simp [ih]

lemma subst_lowerT_u (c : R) (n j : ℕ) :
    subst (lowerT c) (u n j) = (X 0 + C c * X 1) ^ (n - j) * monomial (Finsupp.single 1 j) 1 := by
  rw [subst_u, ← X_pow_eq_monomial]; simp [lowerT]

lemma rhoMat_lowerT_lower (n : ℕ) (c : R) : (rhoMat n (lowerT c)).IsLowerTriangular := by
  intro i j hij
  have hij' : i < j := hij
  rw [rhoMat_apply, subst_lowerT_u, coeff_mul_monomial', if_neg]
  rw [Finsupp.single_le_iff, mono_one]; exact not_le.mpr hij'

lemma rhoMat_lowerT_diag (n : ℕ) (c : R) (j : Fin (n + 1)) : rhoMat n (lowerT c) j j = 1 := by
  rw [rhoMat_apply, subst_lowerT_u, coeff_mul_monomial', if_pos (by simp [Finsupp.single_le_iff])]
  have h : mono n j - Finsupp.single 1 (j : ℕ) = Finsupp.single 0 (n - j) := by
    ext t; fin_cases t <;> simp
  rw [h, coeff_lin_pow, mul_one]

theorem det_rhoMat_lowerT (n : ℕ) (c : R) : (rhoMat n (lowerT c)).det = 1 := by
  rw [det_of_isLowerTriangular _ (rhoMat_lowerT_lower n c)]
  exact Finset.prod_eq_one fun j _ => rhoMat_lowerT_diag n c j

lemma upperT_eq (c : R) : upperT c = swapM * lowerT c * swapM := by
  ext i j; fin_cases i <;> fin_cases j <;> simp [upperT, lowerT, swapM, Matrix.mul_apply, Fin.sum_univ_two]

theorem det_rhoMat_upperT (n : ℕ) (c : R) : (rhoMat n (upperT c)).det = 1 := by
  have hs : (swapM : Mat2 R) * swapM = 1 := by
    ext i j; fin_cases i <;> fin_cases j <;> simp [swapM, Matrix.mul_apply, Fin.sum_univ_two]
  have hs' : (rhoMat n (swapM : Mat2 R)).det * (rhoMat n swapM).det = 1 := by
    rw [← det_mul, ← rhoMat_mul, hs, rhoMat_one, det_one]
  rw [upperT_eq, rhoMat_mul, rhoMat_mul, det_mul, det_mul, det_rhoMat_lowerT, mul_one, hs']

lemma subst_diagM_u (x y : R) (n j : ℕ) :
    subst (diagM x y) (u n j) = C (x ^ (n - j) * y ^ j) * monomial (mono n j) 1 := by
  rw [subst_u, ← u_eq_monomial]; simp [diagM, u, mul_pow, map_mul, map_pow]; ring

lemma rhoMat_diagM (n : ℕ) (x y : R) (i j : Fin (n + 1)) :
    rhoMat n (diagM x y) i j = if i = j then x ^ (n - (j : ℕ)) * y ^ (j : ℕ) else 0 := by
  rw [rhoMat_apply, subst_diagM_u, coeff_C_mul, coeff_monomial]
  by_cases h : i = j
  · subst h; simp
  · rw [if_neg, if_neg h, mul_zero]
    intro h'; exact h (Fin.ext (mono_injective h').symm)

lemma sum_fin_id (n : ℕ) : ∑ j : Fin (n + 1), (j : ℕ) = n * (n + 1) / 2 := by
  rw [Fin.sum_univ_eq_sum_range (fun j => j) (n + 1)]
  have := Finset.sum_range_id_mul_two (n + 1)
  rw [Nat.add_sub_cancel] at this
  exact (Nat.div_eq_of_eq_mul_left two_pos (by rw [this, Nat.mul_comm])).symm

lemma sum_fin_rev (n : ℕ) : ∑ j : Fin (n + 1), (n - (j : ℕ)) = n * (n + 1) / 2 := by
  rw [Fin.sum_univ_eq_sum_range (fun j => n - j) (n + 1), ← sum_fin_id,
    Fin.sum_univ_eq_sum_range (fun j => j) (n + 1)]
  have := Finset.sum_range_reflect (fun j => j) (n + 1)
  simpa using this

theorem det_rhoMat_diagM (n : ℕ) (x y : R) :
    (rhoMat n (diagM x y)).det = (x * y) ^ (n * (n + 1) / 2) := by
  rw [det_of_isUpperTriangular (M := rhoMat n (diagM x y))]
  · simp only [rhoMat_diagM, if_true]
    rw [Finset.prod_mul_distrib, Finset.prod_pow_eq_pow_sum, Finset.prod_pow_eq_pow_sum, sum_fin_id,
      sum_fin_rev, mul_pow]
  · intro i j hij
    rw [rhoMat_diagM, if_neg]
    exact fun h => (lt_irrefl _ (h ▸ hij))

/-- The `L D U` factorisation over a field, when `A₀₀ ≠ 0`. -/
lemma eq_lowerT_diagM_upperT {K : Type*} [Field K] (A : Mat2 K) (ha : A 0 0 ≠ 0) :
    A = lowerT (A 1 0 / A 0 0) * diagM (A 0 0) (A.det / A 0 0) * upperT (A 0 1 / A 0 0) := by
  ext i j; fin_cases i <;> fin_cases j <;>
    simp [lowerT, diagM, upperT, Matrix.mul_apply, Fin.sum_univ_two, Matrix.det_fin_two] <;>
    field_simp <;> ring

/-- Over a field, `det ρ_n(A) = (det A)^{n(n+1)/2}` when `A₀₀ ≠ 0` (the `L D U` factorisation). -/
theorem det_rhoMat_field {K : Type*} [Field K] (n : ℕ) (A : Mat2 K) (ha : A 0 0 ≠ 0) :
    (rhoMat n A).det = A.det ^ (n * (n + 1) / 2) := by
  have hA := eq_lowerT_diagM_upperT A ha
  rw [hA, rhoMat_mul, rhoMat_mul, det_mul, det_mul, det_rhoMat_lowerT, det_rhoMat_upperT,
    det_rhoMat_diagM, one_mul, mul_one, ← hA, mul_div_cancel₀ _ ha]

/-- The coefficient ring of the generic matrix, `ℤ[a, b, c, d]`. -/
abbrev GenRing := MvPolynomial (Fin 2 × Fin 2) ℤ

/-- The generic matrix `(X_{ij})` over `ℤ[a, b, c, d]`. -/
noncomputable def genMat : Mat2 GenRing := Matrix.of fun i j => X (i, j)

/-- Specialisation `X_{ij} ↦ A_{ij}`. -/
noncomputable def evalAt (A : Mat2 R) : GenRing →+* R :=
  eval₂Hom (Int.castRingHom R) fun p => A p.1 p.2

lemma genMat_map (A : Mat2 R) : genMat.map (evalAt A) = A := by
  ext i j; simp [genMat, evalAt]

theorem det_rhoMat_generic (n : ℕ) : (rhoMat n genMat).det = genMat.det ^ (n * (n + 1) / 2) := by
  have hinj : Function.Injective (algebraMap GenRing (FractionRing GenRing)) :=
    IsFractionRing.injective GenRing (FractionRing GenRing)
  apply hinj
  rw [RingHom.map_det, RingHom.mapMatrix_apply, ← rhoMat_map, map_pow, RingHom.map_det,
    RingHom.mapMatrix_apply]
  refine det_rhoMat_field n _ ?_
  rw [Matrix.map_apply]
  intro h
  exact X_ne_zero (R := ℤ) ((0 : Fin 2), (0 : Fin 2)) (hinj (by simpa [genMat] using h))

/-- **`lem:detsym`** ([C] v1.15.0): `det ρ_n(A) = (det A)^{n(n+1)/2}` for every `A ∈ Mat₂(R)`, over every
commutative ring `R`. -/
theorem det_rho (n : ℕ) (A : Mat2 R) : LinearMap.det (rho n A) = A.det ^ (n * (n + 1) / 2) := by
  rw [det_rho_eq_det_rhoMat, ← genMat_map A, rhoMat_map, ← RingHom.mapMatrix_apply, ← RingHom.map_det,
    det_rhoMat_generic, map_pow, RingHom.map_det, RingHom.mapMatrix_apply]

/-- [C] `lem:detsym`, second sentence (the ring-level content): `ρ_n(A)` is invertible as soon as
`det A` is a unit; conversely, for `n ≥ 1`, invertibility of `ρ_n(A)` forces `det A` to be a unit. -/
theorem isUnit_det_rho_iff (n : ℕ) (hn : 1 ≤ n) (A : Mat2 R) :
    IsUnit (LinearMap.det (rho n A)) ↔ IsUnit A.det := by
  rw [det_rho]
  have : n * (n + 1) / 2 ≠ 0 := by
    have : 2 ≤ n * (n + 1) := by nlinarith
    omega
  exact isUnit_pow_iff this

/-- [C] `lem:detsym`, the characteristic-`ℓ` reading (L16-5 minor 9): over a field of characteristic `ℓ`,
for `det A = p` with `ℓ`, `p` prime, `ρ_n(A)` is invertible iff `ℓ ≠ p` or `n = 0`. -/
theorem isUnit_det_rho_charP {F : Type*} [Field F] (ℓ : ℕ) [CharP F ℓ] (hℓ : ℓ.Prime) (n : ℕ) (A : Mat2 F)
    {p : ℕ} (hp : p.Prime) (hA : A.det = (p : F)) :
    IsUnit (LinearMap.det (rho n A)) ↔ ℓ ≠ p ∨ n = 0 := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · simp [det_rho]
  · rw [isUnit_det_rho_iff n hn A, hA, isUnit_iff_ne_zero, Ne, CharP.cast_eq_zero_iff F ℓ,
      Nat.prime_dvd_prime_iff_eq hℓ hp]
    simp [Nat.pos_iff_ne_zero.mp hn]

/-! ### `prop:symconv`(3): the coefficient bound -/

/-- The `ℓ¹`-norm of the coefficients of an integer polynomial in `x, y`. -/
def l1 (p : MvPolynomial (Fin 2) ℤ) : ℤ := ∑ d ∈ p.support, |coeff d p|

lemma l1_nonneg (p : MvPolynomial (Fin 2) ℤ) : 0 ≤ l1 p := Finset.sum_nonneg fun _ _ => abs_nonneg _

lemma l1_eq_of_subset (p : MvPolynomial (Fin 2) ℤ) {s : Finset (Fin 2 →₀ ℕ)} (h : p.support ⊆ s) :
    l1 p = ∑ d ∈ s, |coeff d p| :=
  Finset.sum_subset h fun d _ hd => by rw [MvPolynomial.notMem_support_iff.mp hd, abs_zero]

lemma abs_coeff_le_l1 (p : MvPolynomial (Fin 2) ℤ) (d : Fin 2 →₀ ℕ) : |coeff d p| ≤ l1 p := by
  by_cases hd : d ∈ p.support
  · exact Finset.single_le_sum (f := fun d => |coeff d p|) (fun _ _ => abs_nonneg _) hd
  · rw [MvPolynomial.notMem_support_iff.mp hd, abs_zero]; exact l1_nonneg p

lemma l1_zero : l1 0 = 0 := by simp [l1]

lemma l1_add (p q : MvPolynomial (Fin 2) ℤ) : l1 (p + q) ≤ l1 p + l1 q := by
  classical
  rw [l1_eq_of_subset (p + q) support_add, l1_eq_of_subset p Finset.subset_union_left,
    l1_eq_of_subset q Finset.subset_union_right, ← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun d _ => by rw [coeff_add]; exact abs_add_le _ _

lemma l1_monomial (d : Fin 2 →₀ ℕ) (r : ℤ) : l1 (monomial d r) = |r| := by
  classical
  rw [l1_eq_of_subset _ support_monomial_subset, Finset.sum_singleton, coeff_monomial, if_pos rfl]

lemma l1_sum {ι : Type*} (s : Finset ι) (f : ι → MvPolynomial (Fin 2) ℤ) :
    l1 (∑ i ∈ s, f i) ≤ ∑ i ∈ s, l1 (f i) :=
  Finset.le_sum_of_subadditive l1 l1_zero.le l1_add s f

/-- Submultiplicativity of the `ℓ¹`-norm. -/
lemma l1_mul (p q : MvPolynomial (Fin 2) ℤ) : l1 (p * q) ≤ l1 p * l1 q := by
  conv_lhs => rw [p.as_sum, q.as_sum, Finset.sum_mul_sum]
  refine (l1_sum _ _).trans ?_
  rw [l1, l1, Finset.sum_mul_sum]
  refine Finset.sum_le_sum fun a _ => (l1_sum _ _).trans (Finset.sum_le_sum fun b _ => ?_)
  rw [monomial_mul, l1_monomial, abs_mul]

lemma l1_pow (p : MvPolynomial (Fin 2) ℤ) (k : ℕ) : l1 (p ^ k) ≤ l1 p ^ k := by
  induction k with
  | zero => rw [pow_zero, pow_zero, ← C_1, C_apply, l1_monomial, abs_one]
  | succ k ih =>
    rw [pow_succ, pow_succ]
    exact (l1_mul _ _).trans (mul_le_mul_of_nonneg_right ih (l1_nonneg p))

lemma l1_linForm (A : Mat2 ℤ) (j : Fin 2) : l1 (linForm A j) ≤ |A 0 j| + |A 1 j| := by
  have h : linForm A j = monomial (Finsupp.single 0 1) (A 0 j) + monomial (Finsupp.single 1 1) (A 1 j) := by
    simp only [linForm, Fin.sum_univ_two, X, C_mul_monomial, mul_one]
  rw [h]
  refine (l1_add _ _).trans ?_
  rw [l1_monomial, l1_monomial]

/-- **`prop:symconv`(3), the bound** ([C] v1.15.0): for `A ∈ Mat₂(ℤ)` with `|A_{ts}| ≤ M` and `j ≤ n`, every
coefficient of `ρ_n(A) u_j = (a x + c y)^{n-j} (b x + d y)^j` is at most `(2M)^n` in absolute value.

The hypothesis `j ≤ n` is necessary and was missing from L16-2's `sorry` statement: for `j > n`, `u n j` is
`y^j` of degree `j`, and `A = !![1, 1; 1, 1]` gives the coefficient `C(j, ⌊j/2⌋) > 2^n` once `j ≥ n + 5`. -/
theorem rho_coeff_bound (A : Mat2 ℤ) (n j : ℕ) (hj : j ≤ n) (M : ℤ) (hM : ∀ t s, |A t s| ≤ M)
    (d : Fin 2 →₀ ℕ) : |coeff d (subst A (u n j))| ≤ (2 * M) ^ n := by
  have hlin : ∀ s, l1 (linForm A s) ≤ 2 * M := fun s => (l1_linForm A s).trans (by linarith [hM 0 s, hM 1 s])
  have hsub : subst A (u n j) = linForm A 0 ^ (n - j) * linForm A 1 ^ j := by simp [u]
  refine (abs_coeff_le_l1 _ d).trans ?_
  rw [hsub, ← Nat.sub_add_cancel hj, pow_add, Nat.add_sub_cancel]
  refine (l1_mul _ _).trans (mul_le_mul ((l1_pow _ _).trans (pow_le_pow_left₀ (l1_nonneg _) (hlin 0) _))
    ((l1_pow _ _).trans (pow_le_pow_left₀ (l1_nonneg _) (hlin 1) _)) (l1_nonneg _) ?_)
  exact pow_nonneg ((l1_nonneg _).trans (hlin 0)) _

end GL2
