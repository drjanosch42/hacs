import Mathlib
import GL2.Conventions.SymPow
import GL2.Conventions.SymPowMatrix

/-!
# L3-a(1): the executable `ρ_n` (differential oracle, L16-4)

[GT] `orc:rhoexec`, mirroring [C] `def:rhon`, `eq:rhondef`.

`rhoEntry n a b c d t j` is the `(t, j)` entry of `ρ_n(!![a, b; c, d])` in the basis
`u_t = x^{n-t} y^t`, written as the double sum of `eq:rhondef`
(`s` from `(a x + c y)^{n-j}`, `r = t - s` from `(b x + d y)^j`):
`∑_{s ≤ n-j} ∑_{r ≤ j} [s + r = t] C(n-j,s) a^{n-j-s} c^s C(j,r) b^{j-r} d^r`.

* `subst_u_eq_sum_rhoEntry` is the **specification**: over every commutative ring, column `j` of
  the matrix `rhoEntry · j` expands `subst A (u n j)` (= L16-2's `ρ_n(A) u_j`) in the `u_t`, for
  `j ≤ n`.  So the executable matrix *is* the coordinate matrix of `GL2.rho`.
* `map_rhoEntry`: base change, so the integer (here: `ℕ`) evaluation reduced mod `m` is the
  evaluation over `ZMod m`.
* `rhoNat` / `rhoCheck`: the fast `ℕ`-list evaluation the kernel runs on C++ output, with
  `rhoNat_eq` and the soundness theorem `rhoCheck_sound`.
-/

namespace GL2.Exec

open MvPolynomial

variable {R : Type*} [CommSemiring R]

/-- The `(t, j)` entry of `ρ_n(!![a, b; c, d])` ([C] `eq:rhondef`). -/
def rhoEntry (n : ℕ) (a b c d : R) (t j : ℕ) : R :=
  ∑ s ∈ Finset.range (n - j + 1), ∑ r ∈ Finset.range (j + 1),
    if s + r = t then
      ((n - j).choose s : R) * a ^ (n - j - s) * c ^ s * (j.choose r : R) * b ^ (j - r) * d ^ r
    else 0

/-- The executable `ρ_n(A)` as an `(n+1) × (n+1)` matrix. -/
def rhoMat (n : ℕ) (A : Mat2 R) : Matrix (Fin (n + 1)) (Fin (n + 1)) R :=
  Matrix.of fun t j => rhoEntry n (A 0 0) (A 0 1) (A 1 0) (A 1 1) t j

/-- Base change: a ring map commutes with `rhoEntry`. -/
theorem map_rhoEntry {S : Type*} [CommSemiring S] (f : R →+* S) (n : ℕ) (a b c d : R) (t j : ℕ) :
    f (rhoEntry n a b c d t j) = rhoEntry n (f a) (f b) (f c) (f d) t j := by
  simp only [rhoEntry, map_sum]
  refine Finset.sum_congr rfl fun s _ => Finset.sum_congr rfl fun r _ => ?_
  split_ifs <;> simp

/-- **Specification** (`eq:rhondef`): for `j ≤ n`,
`subst A (u n j) = ∑_{t ≤ n} rhoEntry … t j • u n t`. -/
theorem subst_u_eq_sum_rhoEntry {R : Type*} [CommRing R] (A : Mat2 R) {n j : ℕ} (hj : j ≤ n) :
    subst A (u n j) = ∑ t ∈ Finset.range (n + 1),
      MvPolynomial.C (rhoEntry n (A 0 0) (A 0 1) (A 1 0) (A 1 1) t j) * u n t := by
  rw [subst_u]
  -- expand both binomials
  rw [add_comm (C (A 0 0) * X 0), add_pow, add_comm (C (A 0 1) * X 0), add_pow,
    Finset.sum_mul_sum]
  -- the right-hand side as a double sum over (s, r)
  simp only [rhoEntry, map_sum, Finset.sum_mul]
  symm
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun s hs => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun r hr => ?_
  have hs' : s ≤ n - j := Nat.lt_succ_iff.mp (Finset.mem_range.mp hs)
  have hr' : r ≤ j := Nat.lt_succ_iff.mp (Finset.mem_range.mp hr)
  have hmem : s + r ∈ Finset.range (n + 1) := Finset.mem_range.mpr (by omega)
  simp only [apply_ite, map_zero, ite_mul, zero_mul]
  rw [Finset.sum_ite_eq (Finset.range (n + 1)) (s + r), if_pos hmem]
  have hexp : n - (s + r) = (n - j - s) + (j - r) := by omega
  simp only [u, hexp, map_mul, map_pow, map_natCast]
  ring

/-- **The two `rhoMat`s agree** (L16-5 minor 7): the executable matrix is L16-6's coordinate matrix
`GL2.rhoMat n A = LinearMap.toMatrix (uBasis n) (uBasis n) (rho n A)`. -/
theorem rhoMat_eq {R : Type*} [CommRing R] (n : ℕ) (A : Mat2 R) : rhoMat n A = GL2.rhoMat n A := by
  ext i j
  rw [GL2.rhoMat_apply, subst_u_eq_sum_rhoEntry A (Nat.lt_succ_iff.mp j.2), MvPolynomial.coeff_sum]
  simp only [GL2.u_eq_monomial, MvPolynomial.C_mul_monomial, mul_one, MvPolynomial.coeff_monomial]
  rw [Finset.sum_eq_single (i : ℕ)]
  · simp [rhoMat]
  · intro t _ hti
    rw [if_neg]
    intro h
    exact hti (GL2.mono_injective h)
  · intro hi
    exact absurd (Finset.mem_range.mpr i.2) hi

/-! ### The fast evaluation the kernel runs -/

/-- `List`-based evaluation over `ℕ` (for `decide +kernel` on C++ output). -/
def rhoNat (n a b c d t j : ℕ) : ℕ :=
  ((List.range (n - j + 1)).map fun s => ((List.range (j + 1)).map fun r =>
    if s + r = t then (n - j).choose s * a ^ (n - j - s) * c ^ s * j.choose r * b ^ (j - r) * d ^ r
    else 0).sum).sum

lemma list_sum_range_map {M : Type*} [AddCommMonoid M] (f : ℕ → M) (k : ℕ) :
    ((List.range k).map f).sum = ∑ i ∈ Finset.range k, f i := by
  induction k with
  | zero => simp
  | succ k ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

theorem rhoNat_eq (n a b c d t j : ℕ) : rhoNat n a b c d t j = rhoEntry n a b c d t j := by
  simp only [rhoNat, rhoEntry, list_sum_range_map, Nat.cast_id]

/-- The cast of the `ℕ` evaluation is the evaluation over any commutative ring. -/
theorem cast_rhoNat (n a b c d t j : ℕ) :
    ((rhoNat n a b c d t j : ℕ) : R) = rhoEntry n (a : R) (b : R) (c : R) (d : R) t j := by
  rw [rhoNat_eq]
  exact map_rhoEntry (Nat.castRingHom R) n a b c d t j

/-- One dumped C++ matrix: the reduced key `(a, b, c, d)` (entries of `A mod ℓ`) and its rows. -/
structure RhoDump where
  key : ℕ × ℕ × ℕ × ℕ
  rows : List (List ℕ)
  deriving Repr

/-- The matrix `!![a, b; c, d]` of a key, over `R`. -/
def RhoDump.mat {R : Type*} [CommRing R] (D : RhoDump) : Mat2 R :=
  !![(D.key.1 : R), (D.key.2.1 : R); (D.key.2.2.1 : R), (D.key.2.2.2 : R)]

/-- Entry `(t, j)` of a dump (`0` when absent; `rhoCheck` rules absence out). -/
def RhoDump.entry (D : RhoDump) (t j : ℕ) : ℕ := (D.rows.getD t []).getD j 0

/-- The checker: shape `(n+1) × (n+1)`, entries `< m`, and every entry is `rhoNat … mod m`. -/
def rhoCheckOne (n m : ℕ) (D : RhoDump) : Bool :=
  D.rows.length == n + 1 && D.rows.all (fun row => row.length == n + 1 && row.all (· < m)) &&
    (List.range (n + 1)).all fun t => (List.range (n + 1)).all fun j =>
      D.entry t j == rhoNat n D.key.1 D.key.2.1 D.key.2.2.1 D.key.2.2.2 t j % m

def rhoCheck (n m : ℕ) (Ds : List RhoDump) : Bool := Ds.all (rhoCheckOne n m)

/-- **Soundness** (differential oracle): if `rhoCheck` accepts, then every dumped matrix, read in
`ZMod m`, is the executable `rhoMat n` of its key, and each of its columns `j` expands
`subst A (u n j)` -- i.e. the C++ matrix is `ρ_n(A mod m)` in L16-2's convention. -/
theorem rhoCheck_sound (n m : ℕ) (Ds : List RhoDump) (h : rhoCheck n m Ds = true) :
    ∀ D ∈ Ds, (Matrix.of fun (t j : Fin (n + 1)) => ((D.entry t j : ℕ) : ZMod m)) =
        rhoMat n (D.mat (R := ZMod m)) ∧
      ∀ j : Fin (n + 1), subst (D.mat (R := ZMod m)) (u n j) =
        ∑ t ∈ Finset.range (n + 1), MvPolynomial.C ((D.entry t j : ℕ) : ZMod m) * u n t := by
  intro D hD
  have h1 : rhoCheckOne n m D = true := List.all_eq_true.mp h D hD
  simp only [rhoCheckOne, Bool.and_eq_true, List.all_eq_true, beq_iff_eq, List.mem_range] at h1
  obtain ⟨_, hent⟩ := h1
  have key : ∀ t j : ℕ, t < n + 1 → j < n + 1 →
      ((D.entry t j : ℕ) : ZMod m) = rhoEntry n (D.key.1 : ZMod m) (D.key.2.1 : ZMod m)
        (D.key.2.2.1 : ZMod m) (D.key.2.2.2 : ZMod m) t j := by
    intro t j ht hj
    rw [hent t ht j hj, ZMod.natCast_mod, cast_rhoNat]
  refine ⟨?_, fun j => ?_⟩
  · ext t j
    simp only [Matrix.of_apply, rhoMat, RhoDump.mat, Matrix.of_apply, Matrix.cons_val',
      Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.empty_val', Matrix.cons_val_fin_one]
    exact key t j t.2 j.2
  · rw [subst_u_eq_sum_rhoEntry _ (Nat.lt_succ_iff.mp j.2)]
    refine Finset.sum_congr rfl fun t ht => ?_
    rw [key t j (Finset.mem_range.mp ht) j.2]
    simp [RhoDump.mat]

end GL2.Exec
