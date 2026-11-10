import GL2.Conventions.SymPowDet
import GL2.Conventions.Contragredient

/-!
# `lem:symdual`: the integral invariant form `B_n` and `Φ_n : Sym^n → D_n` (L16-6)

[GT] `lem:symdual`, mirroring [C] v1.15.0 `lem:symdual`, `eq:BnGram`.

`B_n` is defined here by its Gram matrix `eq:BnGram` in the basis `u_j`,
`B_n(u_t, u_j) = (-1)^t t! (n-t)! δ_{t+j,n}` (`GL2.gram`); [C]'s permutation-sum definition on products of
linear forms is the motivation and is not formalized (its only use in [C] is to derive `eq:BnGram`).
The equivariance is proved as the matrix identity `ρ_n(g)ᵀ G = G ρ_n(adj g)` (`GL2.rhoMat_transpose_gram`):
for the generators `!![1,0;c,1]`, `diag(x,y)` and the swap by explicit computation, over a field by the
`L D U` factorisation of `GL2.Conventions.SymPowDet`, and over every commutative ring by descent from the
generic matrix.
-/

namespace GL2
open MvPolynomial Matrix

variable {R : Type*} [CommRing R]

/-! ### The matrix of `ρ_n(!![1,0;c,1])` -/

lemma coeff_lin_pow_gen (c : R) (k : ℕ) (d : Fin 2 →₀ ℕ) :
    coeff d ((X 0 + C c * X 1 : MvPolynomial (Fin 2) R) ^ k) =
      if d 0 + d 1 = k then (k.choose (d 1) : R) * c ^ (d 1) else 0 := by
  induction k generalizing d with
  | zero =>
    rw [pow_zero, coeff_one]
    by_cases h : d = 0
    · subst h; simp
    · rw [if_neg (Ne.symm h), if_neg]
      intro h'; apply h; ext t; fin_cases t <;> simp <;> omega
  | succ k ih =>
    rw [pow_succ, mul_add, coeff_add, coeff_mul_X', ← mul_assoc, mul_comm _ (C c), mul_assoc,
      coeff_C_mul, coeff_mul_X', ih, ih]
    simp only [Finsupp.mem_support_iff, Finsupp.coe_tsub, Pi.sub_apply, Finsupp.single_apply]
    simp only [Fin.isValue, ne_eq, zero_ne_one, one_ne_zero, if_true, if_false, Nat.sub_zero]
    rcases Nat.eq_zero_or_pos (d 0) with h0 | h0 <;> rcases Nat.eq_zero_or_pos (d 1) with h1 | h1
    · simp [h0, h1]
    · simp only [h0, h1.ne', not_true_eq_false, not_false_eq_true, if_false, if_true, zero_add]
      by_cases hk : d 1 = k + 1
      · rw [if_pos (by omega), if_pos hk, hk, Nat.add_sub_cancel, Nat.choose_self, Nat.choose_self]
        simp only [Nat.cast_one, one_mul]; ring
      · rw [if_neg (by omega), if_neg hk, mul_zero]
    · simp only [h1, h0.ne', not_true_eq_false, not_false_eq_true, if_false, if_true, add_zero]
      by_cases hk : d 0 = k + 1
      · rw [if_pos (by omega), if_pos hk]; simp
      · rw [if_neg (by omega), if_neg hk]; simp
    · simp only [h0.ne', h1.ne', not_false_eq_true, if_true]
      by_cases hk : d 0 + d 1 = k + 1
      · rw [if_pos (by omega), if_pos (by omega), if_pos hk]
        obtain ⟨b, hb⟩ : ∃ b, d 1 = b + 1 := ⟨d 1 - 1, by omega⟩
        rw [hb, Nat.add_sub_cancel, Nat.choose_succ_succ', Nat.cast_add, pow_succ]
        ring
      · rw [if_neg (by omega), if_neg (by omega), if_neg hk]; simp

lemma rhoMat_lowerT_apply (n : ℕ) (c : R) (i j : Fin (n + 1)) :
    rhoMat n (lowerT c) i j =
      if (j : ℕ) ≤ i then (((n - j).choose (i - j) : ℕ) : R) * c ^ ((i : ℕ) - j) else 0 := by
  rw [rhoMat_apply, subst_lowerT_u, coeff_mul_monomial']
  have hi := i.2
  by_cases h : (j : ℕ) ≤ i
  · rw [if_pos (by rw [Finsupp.single_le_iff, mono_one]; exact h), if_pos h, mul_one, coeff_lin_pow_gen]
    have h10 : ((1 : Fin 2) = 0) ↔ False := by decide
    simp only [Finsupp.coe_tsub, Pi.sub_apply, mono_zero, mono_one, Finsupp.single_apply, h10, if_false,
      if_true, Nat.sub_zero]
    rw [if_pos (by omega)]
  · rw [if_neg (by rw [Finsupp.single_le_iff, mono_one]; exact h), if_neg h]

lemma subst_swapM_u (n j : ℕ) (hj : j ≤ n) : subst (swapM : Mat2 R) (u n j) = u n (n - j) := by
  rw [subst_u]; simp [swapM, u, Nat.sub_sub_self hj, mul_comm]

lemma rhoMat_swapM_apply (n : ℕ) (i j : Fin (n + 1)) :
    rhoMat n (swapM : Mat2 R) i j = if i = Fin.rev j then 1 else 0 := by
  rw [rhoMat_apply, subst_swapM_u n j (Nat.lt_succ_iff.mp j.2), u_eq_monomial, coeff_monomial]
  have hrev : ((Fin.rev j : Fin (n + 1)) : ℕ) = n - j := by simp [Fin.val_rev]
  by_cases h : i = Fin.rev j
  · subst h; rw [if_pos (by rw [hrev]), if_pos rfl]
  · rw [if_neg, if_neg h]
    intro h'; apply h; apply Fin.ext; rw [hrev]; exact (mono_injective h').symm

/-! ### The Gram matrix `eq:BnGram` -/

/-- **`eq:BnGram`**: `G_{tj} = (-1)^t t! (n-t)! δ_{t+j,n}`, the Gram matrix of `B_n` in the basis `u_j`. -/
noncomputable def gram (n : ℕ) : Matrix (Fin (n + 1)) (Fin (n + 1)) R :=
  Matrix.of fun t j =>
    if (t : ℕ) + j = n then (-1) ^ (t : ℕ) * (((t : ℕ).factorial * (n - t).factorial : ℕ) : R) else 0

lemma gram_apply (n : ℕ) (t j : Fin (n + 1)) :
    (gram n : Matrix _ _ R) t j =
      if (t : ℕ) + j = n then (-1) ^ (t : ℕ) * (((t : ℕ).factorial * (n - t).factorial : ℕ) : R) else 0 := rfl

lemma gram_map {S : Type*} [CommRing S] (f : R →+* S) (n : ℕ) : (gram n : Matrix _ _ R).map f = gram n := by
  ext t j; rw [Matrix.map_apply, gram_apply, gram_apply]; split_ifs <;> simp

lemma val_rev (n : ℕ) (j : Fin (n + 1)) : ((Fin.rev j : Fin (n + 1)) : ℕ) = n - j := by
  simp [Fin.val_rev]

lemma transpose_mul_gram (n : ℕ) (M : Matrix (Fin (n + 1)) (Fin (n + 1)) R) (t j : Fin (n + 1)) :
    (Mᵀ * (gram n : Matrix _ _ R)) t j = M (Fin.rev j) t * gram n (Fin.rev j) j := by
  rw [mul_apply]
  refine Finset.sum_eq_single (Fin.rev j) (fun a _ ha => ?_) (by simp)
  rw [gram_apply, if_neg, mul_zero]
  intro h; apply ha; apply Fin.ext; rw [val_rev]; omega

lemma gram_mul_apply (n : ℕ) (M : Matrix (Fin (n + 1)) (Fin (n + 1)) R) (t j : Fin (n + 1)) :
    ((gram n : Matrix _ _ R) * M) t j = gram n t (Fin.rev t) * M (Fin.rev t) j := by
  rw [mul_apply]
  refine Finset.sum_eq_single (Fin.rev t) (fun a _ ha => ?_) (by simp)
  rw [gram_apply, if_neg, zero_mul]
  intro h; apply ha; apply Fin.ext; rw [val_rev]; omega

/-- The intertwining property `ρ_n(A)ᵀ G = G ρ_n(adj A)`, i.e. `B_n(ρ_n(A) v, w) = B_n(v, ρ_n(adj A) w)`. -/
def GramAdj (n : ℕ) (A : Mat2 R) : Prop := (rhoMat n A)ᵀ * gram n = gram n * rhoMat n A.adjugate

lemma gramAdj_mul (n : ℕ) {A B : Mat2 R} (hA : GramAdj n A) (hB : GramAdj n B) : GramAdj n (A * B) := by
  unfold GramAdj at *
  rw [rhoMat_mul, transpose_mul, Matrix.mul_assoc, hA, ← Matrix.mul_assoc, hB, Matrix.mul_assoc,
    ← rhoMat_mul, adjugate_mul_distrib]

lemma gramAdj_map {S : Type*} [CommRing S] (f : R →+* S) (n : ℕ) {A : Mat2 R} (h : GramAdj n A) :
    GramAdj n (A.map f) := by
  unfold GramAdj at *
  have := congrArg (fun M => M.map f) h
  simp only [Matrix.map_mul, transpose_map, gram_map] at this
  have hadj : (A.map f).adjugate = A.adjugate.map f := (RingHom.map_adjugate f A).symm
  rw [rhoMat_map, hadj, rhoMat_map]
  exact this

lemma gramAdj_diagM (n : ℕ) (x y : R) : GramAdj n (diagM x y) := by
  have hadj : (diagM x y).adjugate = diagM y x := by
    rw [diagM, adjugate_fin_two_of]; simp [diagM]
  unfold GramAdj
  rw [hadj]
  ext t j
  rw [transpose_mul_gram, gram_mul_apply, rhoMat_diagM, rhoMat_diagM]
  by_cases h : Fin.rev j = t
  · subst h
    rw [if_pos rfl, Fin.rev_rev, if_pos rfl, val_rev]
    have hj := j.2
    rw [show n - (n - (j : ℕ)) = j by omega]
    ring
  · rw [if_neg h, if_neg, zero_mul, mul_zero]
    intro h'; apply h; rw [← h', Fin.rev_rev]

lemma rhoMat_smul_one (n : ℕ) (r : R) : rhoMat n (r • (1 : Mat2 R)) = r ^ n • (1 : Matrix _ _ R) := by
  rw [rhoMat, rho_smul_one, map_smul, LinearMap.toMatrix_one]

lemma neg_one_pow_sub' (n t : ℕ) (h : t ≤ n) : (-1 : R) ^ (n - t) = (-1) ^ n * (-1) ^ t := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [Nat.add_sub_cancel_left, pow_add]
  rcases Nat.even_or_odd t with ht | ht <;> simp [ht.neg_one_pow]

lemma gramAdj_swapM (n : ℕ) : GramAdj n (swapM : Mat2 R) := by
  have hadj : (swapM : Mat2 R).adjugate = ((-1 : R) • (1 : Mat2 R)) * swapM := by
    rw [swapM, adjugate_fin_two_of]; ext i j; fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two]
  unfold GramAdj
  rw [hadj, rhoMat_mul, rhoMat_smul_one, Matrix.smul_mul, Matrix.one_mul, Matrix.mul_smul]
  ext t j
  rw [Matrix.smul_apply, transpose_mul_gram, gram_mul_apply, rhoMat_swapM_apply, rhoMat_swapM_apply, smul_eq_mul]
  by_cases h : t = j
  · subst h
    have ht := t.2
    rw [if_pos rfl, gram_apply, gram_apply, val_rev, if_pos (by omega), if_pos (by omega),
      show n - (n - (t : ℕ)) = t by omega, neg_one_pow_sub' n t (by omega)]
    rw [Nat.mul_comm (n - (t : ℕ)).factorial]
    ring
  · rw [if_neg (fun h' => h (Fin.rev_injective h'.symm)), if_neg (fun h' => h (Fin.rev_injective h')),
      zero_mul, mul_zero, mul_zero]

lemma choose_fact_identity (t j k : ℕ) :
    (j + k).choose k * ((t + k).factorial * j.factorial) = t.factorial * (j + k).factorial * (t + k).choose k := by
  apply Nat.eq_of_mul_eq_mul_right (Nat.factorial_pos k)
  have h1 := Nat.add_choose_mul_factorial_mul_factorial k j
  have h2 := Nat.add_choose_mul_factorial_mul_factorial k t
  rw [add_comm j k, add_comm t k, ← h1, ← h2]
  simp only [Nat.choose_symm_add]
  ring

lemma gramAdj_lowerT (n : ℕ) (c : R) : GramAdj n (lowerT c) := by
  have hadj : (lowerT c).adjugate = lowerT (-c) := by
    rw [lowerT, adjugate_fin_two_of]; simp [lowerT]
  unfold GramAdj
  rw [hadj]
  ext t j
  have ht := t.2
  have hj := j.2
  rw [transpose_mul_gram, gram_mul_apply, rhoMat_lowerT_apply, rhoMat_lowerT_apply, gram_apply, gram_apply,
    val_rev, val_rev, if_pos (show n - (j : ℕ) + j = n by omega), if_pos (show (t : ℕ) + (n - t) = n by omega)]
  by_cases h : (t : ℕ) + j ≤ n
  · rw [if_pos (show (t : ℕ) ≤ n - j by omega), if_pos (show (j : ℕ) ≤ n - t by omega)]
    obtain ⟨k, hk⟩ : ∃ k, n = t + j + k := ⟨n - t - j, by omega⟩
    rw [show n - (j : ℕ) - t = k by omega, show n - (t : ℕ) - j = k by omega, show n - (t : ℕ) = j + k by omega,
      show n - (j : ℕ) = t + k by omega, show n - (t + k) = (j : ℕ) by omega, neg_pow, pow_add]
    have key := congrArg (Nat.cast : ℕ → R) (choose_fact_identity t j k)
    push_cast at key ⊢
    linear_combination ((-1 : R) ^ (t : ℕ) * (-1) ^ k * c ^ k) * key
  · rw [if_neg (show ¬ (t : ℕ) ≤ n - j by omega), if_neg (show ¬ (j : ℕ) ≤ n - t by omega), zero_mul, mul_zero]

/-! ### The intertwining over every commutative ring -/

theorem gramAdj_field {K : Type*} [Field K] (n : ℕ) (A : Mat2 K) (ha : A 0 0 ≠ 0) : GramAdj n A := by
  rw [eq_lowerT_diagM_upperT A ha, upperT_eq]
  exact gramAdj_mul n (gramAdj_mul n (gramAdj_lowerT n _) (gramAdj_diagM n _ _))
    (gramAdj_mul n (gramAdj_mul n (gramAdj_swapM n) (gramAdj_lowerT n _)) (gramAdj_swapM n))

theorem gramAdj_generic (n : ℕ) : GramAdj n genMat := by
  set f := algebraMap GenRing (FractionRing GenRing)
  have hinj : Function.Injective f := IsFractionRing.injective GenRing (FractionRing GenRing)
  have hK := gramAdj_field n (genMat.map f) (by
    rw [Matrix.map_apply]; intro h
    exact X_ne_zero (R := ℤ) ((0 : Fin 2), (0 : Fin 2)) (hinj (by simpa [genMat] using h)))
  unfold GramAdj at hK ⊢
  have hadj : (genMat.map f).adjugate = genMat.adjugate.map f := (RingHom.map_adjugate f genMat).symm
  rw [rhoMat_map, hadj, rhoMat_map, ← gram_map f, ← transpose_map, ← Matrix.map_mul, ← Matrix.map_mul] at hK
  exact Matrix.map_injective hinj hK

/-- **`lem:symdual`, the matrix form of the second equivariance**: `ρ_n(g)ᵀ G = G ρ_n(adj g)` for every
`g ∈ Mat₂(R)`, over every commutative ring `R`. -/
theorem rhoMat_transpose_gram (n : ℕ) (g : Mat2 R) :
    (rhoMat n g)ᵀ * gram n = gram n * rhoMat n g.adjugate := by
  rw [← genMat_map g]
  exact gramAdj_map _ n (gramAdj_generic n)

/-! ### The form `B_n` and `Φ_n` -/

/-- **`lem:symdual`**: the bilinear form `B_n` on `Sym^n(R²)`, defined by its Gram matrix `eq:BnGram` in the basis
`u_j`. -/
noncomputable def symForm (n : ℕ) : LinearMap.BilinForm R (SymN R n) :=
  Matrix.toLinearMap₂ (uBasis n) (uBasis n) (gram n)

/-- **`eq:BnGram`**: `B_n(u_t, u_j) = (-1)^t t! (n-t)! δ_{t+j,n}`. -/
theorem symForm_basis (n : ℕ) (t j : Fin (n + 1)) :
    symForm n (uBasis n t) (uBasis n j) =
      if (t : ℕ) + j = n then (-1) ^ (t : ℕ) * (((t : ℕ).factorial * (n - t).factorial : ℕ) : R) else 0 :=
  Matrix.toLinearMap₂_apply_basis _ _ _ _ _

/-- **`lem:symdual`, second equivariance**: `B_n(ρ_n(g) v, w) = B_n(v, ρ_n(adj g) w)` for every `g ∈ Mat₂(R)`. -/
theorem symForm_rho_adj (n : ℕ) (g : Mat2 R) (v w : SymN R n) :
    symForm n (rho n g v) w = symForm n v (rho n g.adjugate w) := by
  have h : (symForm n).compl₁₂ (rho n g) LinearMap.id = (symForm n).compl₁₂ LinearMap.id (rho n g.adjugate) := by
    apply (LinearMap.toMatrix₂ (uBasis n) (uBasis n)).injective
    rw [LinearMap.toMatrix₂_compl₁₂ (uBasis n) (uBasis n) (uBasis n) (uBasis n),
      LinearMap.toMatrix₂_compl₁₂ (uBasis n) (uBasis n) (uBasis n) (uBasis n), symForm,
      LinearMap.toMatrix₂_toLinearMap₂, LinearMap.toMatrix_id, transpose_one, Matrix.one_mul, Matrix.mul_one]
    exact rhoMat_transpose_gram n g
  exact congrArg (fun B => B v w) h

/-- **`lem:symdual`, first equivariance**: `B_n(ρ_n(g) v, ρ_n(g) w) = (det g)^n B_n(v, w)`. -/
theorem symForm_rho_rho (n : ℕ) (g : Mat2 R) (v w : SymN R n) :
    symForm n (rho n g v) (rho n g w) = g.det ^ n * symForm n v w := by
  rw [symForm_rho_adj, ← Module.End.mul_apply, ← rho_mul, adj_mul_self, rho_smul_one, LinearMap.smul_apply,
    Module.End.one_apply, map_smul, smul_eq_mul]

/-- **`lem:symdual`, `Φ_n`**: `Φ_n(v) = B_n(v, ·)` intertwines `ρ_n` with the contragredient `ρ_n^∨`:
`Φ_n ∘ ρ_n(g) = ρ_n^∨(g) ∘ Φ_n` for every `g ∈ Mat₂(R)`. -/
theorem symForm_intertwines (n : ℕ) (g : Mat2 R) :
    (symForm n : SymN R n →ₗ[R] DualN R n) ∘ₗ rho n g = rhoDual n g ∘ₗ symForm n := by
  ext v w
  simp only [LinearMap.comp_apply, rhoDual_apply]
  exact symForm_rho_adj n g v w

/-! ### When `Φ_n` is an isomorphism -/

lemma gram_submatrix_rev (n : ℕ) :
    (gram n : Matrix _ _ R).submatrix Fin.rev id = diagonal fun t => gram n (Fin.rev t) t := by
  ext t j
  rw [submatrix_apply, id, diagonal_apply]
  by_cases h : t = j
  · subst h; rw [if_pos rfl]
  · rw [if_neg h, gram_apply, if_neg]
    rw [val_rev]; intro h'; apply h; apply Fin.ext; have := t.2; omega

lemma isUnit_det_gram_iff (n : ℕ) : IsUnit (gram n : Matrix _ _ R).det ↔ IsUnit ((n.factorial : ℕ) : R) := by
  have hs : IsUnit ((((Equiv.Perm.sign (Fin.revPerm : Equiv.Perm (Fin (n + 1)))) : ℤ) : R)) := by
    rcases Int.units_eq_one_or (Equiv.Perm.sign (Fin.revPerm : Equiv.Perm (Fin (n + 1)))) with h | h <;> simp [h]
  have hperm := det_permute (Fin.revPerm : Equiv.Perm (Fin (n + 1))) (gram n : Matrix _ _ R)
  rw [show ((gram n : Matrix _ _ R).submatrix (⇑Fin.revPerm) id) = (gram n).submatrix Fin.rev id from rfl,
    gram_submatrix_rev, det_diagonal] at hperm
  have h1 : IsUnit (gram n : Matrix _ _ R).det ↔
      IsUnit (∏ t : Fin (n + 1), (gram n : Matrix _ _ R) (Fin.rev t) t) := by
    rw [hperm, IsUnit.mul_iff]; exact ⟨fun h => ⟨hs, h⟩, fun h => h.2⟩
  rw [h1, IsUnit.prod_iff]
  have hfac : ∀ t : Fin (n + 1), (gram n : Matrix _ _ R) (Fin.rev t) t =
      (-1) ^ (n - (t : ℕ)) * (((n - (t : ℕ)).factorial * (t : ℕ).factorial : ℕ) : R) := by
    intro t; have := t.2
    rw [gram_apply, val_rev, if_pos (by omega), show n - (n - (t : ℕ)) = t by omega]
  have hneg : ∀ k : ℕ, IsUnit ((-1 : R) ^ k) := fun k => isUnit_one.neg.pow k
  simp only [hfac, IsUnit.mul_iff, hneg, true_and, Finset.mem_univ, forall_const]
  constructor
  · intro h; simpa using h 0
  · intro hn t
    have hd := Nat.factorial_mul_factorial_dvd_factorial (Nat.sub_le n t)
    rw [Nat.sub_sub_self (Nat.lt_succ_iff.mp t.2)] at hd
    exact isUnit_of_dvd_unit (Nat.cast_dvd_cast hd) hn

lemma repr_toDualEquiv_symm {ι M : Type*} [Fintype ι] [DecidableEq ι] [AddCommGroup M] [Module R M]
    (b : Module.Basis ι R M) (φ : Module.Dual R M) (i : ι) : b.repr (b.toDualEquiv.symm φ) i = φ (b i) := by
  rw [← Module.Basis.toDual_apply_left, ← Module.Basis.toDualEquiv_apply, LinearEquiv.apply_symm_apply]

/-- **`lem:symdual`, last clause**: `Φ_n ⊗ R` is an isomorphism iff `n! ∈ R^×` (for a field of characteristic `ℓ`:
iff `n ≤ ℓ - 1`). -/
theorem symForm_bijective_iff (n : ℕ) :
    Function.Bijective (symForm n : SymN R n →ₗ[R] DualN R n) ↔ IsUnit ((n.factorial : ℕ) : R) := by
  set e := (uBasis (R := R) n).toDualEquiv
  set ψ : Module.End R (SymN R n) := e.symm.toLinearMap ∘ₗ (symForm n : SymN R n →ₗ[R] DualN R n)
  have hb : Function.Bijective (symForm n : SymN R n →ₗ[R] DualN R n) ↔ Function.Bijective ψ := by
    simp only [ψ, LinearMap.coe_comp, LinearEquiv.coe_coe]
    exact (e.symm.bijective.of_comp_iff' _).symm
  have hmat : LinearMap.toMatrix (uBasis n) (uBasis n) ψ = (gram n)ᵀ := by
    ext i j
    rw [LinearMap.toMatrix_apply, transpose_apply]
    simp only [ψ, LinearMap.comp_apply, LinearEquiv.coe_coe]
    rw [repr_toDualEquiv_symm, symForm]
    exact Matrix.toLinearMap₂_apply_basis _ _ _ _ _
  rw [hb, ← Module.End.isUnit_iff, LinearMap.isUnit_iff_isUnit_det, ← LinearMap.det_toMatrix (uBasis n), hmat,
    det_transpose, isUnit_det_gram_iff]

end GL2
