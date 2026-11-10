/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# LG-4 (slot 23): the `c_p` lemma — `c_p = min(v_ℓ(p^n − 1), min ed(T_p − (1 + p^{n+1})))`

`L23_master_plan.md` v0.4 §4.3 (row "`c_p = min(v_ℓ(p^n − 1), min ed_eis_p)`", package LG-4); the proof is [Rv]
(`census_l23_plan_v02_review.md`) "`c_p` is derived, not independent (repair 2)".
[GT] `gt_part_MP.tex`, rows `thm:lg-cp-core`, `thm:lg-cp-smith`, `thm:lg-cp`; [C] `lem:cprule` (1)–(2);
audit `F-LG34_audit.md` (repairs R5, R6 applied).

Let `R` be **any** commutative ring (the plan's DVR `ℤ_ℓ`, or `ℤ/ℓ^K` — no Noetherian, no PID hypothesis is needed),
`ϖ ∈ R`, `T` a square matrix over `R` with an eigenvector `e`, `T e = λ e`, which is *primitive* in the sense that one
coordinate `e_{i₀}` is a unit (over a DVR: `e ∉ ϖ Λ`).

* **The core** (`zeroMod_sub_iff`): for every `μ ∈ R` and every `a ∈ R`,
  `T − μ ≡ 0 mod a  ⟺  a ∣ λ − μ  ∧  T − λ ≡ 0 mod a` (entrywise).  ([Rv]: `c e₀ = (X + c) e₀`.)
* **Smith forms** (`zeroMod_iff_of_decomp`, `SmithDecomp.edMin_eq_content`): if `A = P · diag(d) · Q` with `P`, `Q`
  invertible (any such decomposition: mathlib's Smith normal form, or the C++ producer's witnesses `(P, P⁻¹, Q, Q⁻¹, D)`
  of `L23_master_plan.md` §4.4), then `A ≡ 0 mod ϖ^m ⟺ ϖ^m ∣ d_i for every i`.  So the **minimal elementary divisor**
  `min ed(A) := min_i v_ϖ(d_i)` (with `v_ϖ(0) = ∞`: a kernel divisor is `inf`, [Rv] repair 1) equals the content
  `min_{i,j} v_ϖ(A_{ij})`, and in particular does not depend on the decomposition.
* **The `c_p` formula** (`cp_eq_min`, `cp_formula`): with `λ = 1 + p^{n+1}` (the Eisenstein eigenvalue of `T_p`),
  `μ = 1 + p`, `p ∈ R^×`:
  `min ed(T_p − (1 + p)) = min(v_ϖ(p^n − 1), min ed(T_p − (1 + p^{n+1})))`, in `ℕ∞`.

Everything is general (no cell, no `lit_hyp`).  **Not here:** the *existence* of a Smith decomposition over a PID (not
needed for the statement: it is quantified over any decomposition, and `edMin_eq_content` reduces it to the entries);
the identification of `T_p` on `Λ¹` with the census operator and of `e` with the primitive Eisenstein vector (per cell,
LH); over `ℤ/ℓ^K` valuations are only meaningful below the precision `K` (`ϖ^K = 0` makes every `v ≥ K` read `∞`), which is
the census's precision rule, not part of this lemma.  `v_ϖ(0) = ∞` holds in every ring; the converse "a divisor that
reads `∞` is `0` (the kernel)" needs `⋂_m ϖ^m R = 0` (true over `ℤ_ℓ`, and over `ℤ/ℓ^K` since `ℓ^K = 0`; false e.g. for
an idempotent `ϖ`) — no statement here relies on it.  **Saturation (not formalised):** [C] `lem:cprule` starts from an
eigenvector in `N ⊗ ℚ_ℓ` and scales it to a primitive `e₀ ∈ N` (one DVR step); here the primitive integral eigenvector
is a hypothesis, and "primitive" means "some coordinate is a unit" (equivalent to `e ∉ ϖΛ` over a local ring).
-/

namespace GL2.L23.Cp

open Matrix

variable {R : Type} [CommRing R] {ι : Type} [Fintype ι] [DecidableEq ι]

/-! ### `A ≡ 0 mod a`, entrywise -/

/-- `A ≡ 0 mod a`: every entry of `A` is divisible by `a`. -/
def ZeroMod (A : Matrix ι ι R) (a : R) : Prop := ∀ i j, a ∣ A i j

omit [DecidableEq ι] in
lemma zeroMod_mul_left {A : Matrix ι ι R} {a : R} (h : ZeroMod A a) (P : Matrix ι ι R) : ZeroMod (P * A) a := by
  intro i j
  rw [Matrix.mul_apply]
  exact Finset.dvd_sum fun k _ => dvd_mul_of_dvd_right (h k j) _

omit [DecidableEq ι] in
lemma zeroMod_mul_right {A : Matrix ι ι R} {a : R} (h : ZeroMod A a) (Q : Matrix ι ι R) : ZeroMod (A * Q) a := by
  intro i j
  rw [Matrix.mul_apply]
  exact Finset.dvd_sum fun k _ => dvd_mul_of_dvd_left (h i k) _

/-- **`A = P D Q` with `P`, `Q` invertible: `A ≡ 0 mod a ⟺ D ≡ 0 mod a`.** -/
theorem zeroMod_iff_of_decomp {A P P' D Q Q' : Matrix ι ι R} (hA : A = P * D * Q) (hP : P' * P = 1)
    (hQ : Q * Q' = 1) (a : R) : ZeroMod A a ↔ ZeroMod D a := by
  constructor
  · intro h
    have hD : D = P' * A * Q' := by
      calc D = (P' * P) * D * (Q * Q') := by rw [hP, hQ, Matrix.one_mul, Matrix.mul_one]
        _ = P' * (P * D * Q) * Q' := by simp only [Matrix.mul_assoc]
        _ = P' * A * Q' := by rw [hA]
    rw [hD]
    exact zeroMod_mul_right (zeroMod_mul_left h _) _
  · intro h
    rw [hA]
    exact zeroMod_mul_right (zeroMod_mul_left h _) _

omit [Fintype ι] in
lemma zeroMod_diagonal_iff (d : ι → R) (a : R) : ZeroMod (diagonal d) a ↔ ∀ i, a ∣ d i := by
  constructor
  · intro h i
    simpa using h i i
  · intro h i j
    by_cases hij : i = j
    · subst hij
      simpa using h i
    · rw [diagonal_apply_ne _ hij]
      exact dvd_zero a

/-! ### The core: an eigenvector with a unit coordinate -/

/-- **The `c_p` core** ([Rv] repair 2): if `T e = λ e` and `e_{i₀}` is a unit, then for every `μ` and every `a`,
`T − μ ≡ 0 mod a ⟺ a ∣ λ − μ ∧ T − λ ≡ 0 mod a`. -/
theorem zeroMod_sub_iff {T : Matrix ι ι R} {e : ι → R} {lam mu : R} (he : T *ᵥ e = lam • e) {i₀ : ι}
    (hu : IsUnit (e i₀)) (a : R) :
    ZeroMod (T - mu • (1 : Matrix ι ι R)) a ↔ a ∣ lam - mu ∧ ZeroMod (T - lam • (1 : Matrix ι ι R)) a := by
  have hent : ∀ (c : R) (i j : ι), (T - c • (1 : Matrix ι ι R)) i j = T i j - c * (1 : Matrix ι ι R) i j :=
    fun c i j => by simp only [Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul]
  constructor
  · intro h
    have hv : ((T - mu • (1 : Matrix ι ι R)) *ᵥ e) i₀ = (lam - mu) * e i₀ := by
      rw [Matrix.sub_mulVec, he, Matrix.smul_mulVec, Matrix.one_mulVec]
      simp only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul]
      ring
    have hd : a ∣ ((T - mu • (1 : Matrix ι ι R)) *ᵥ e) i₀ := by
      simp only [mulVec, dotProduct]
      exact Finset.dvd_sum fun j _ => dvd_mul_of_dvd_left (h i₀ j) _
    rw [hv] at hd
    have hc : a ∣ lam - mu := (hu.dvd_mul_right).mp hd
    refine ⟨hc, fun i j => ?_⟩
    have : (T - lam • (1 : Matrix ι ι R)) i j =
        (T - mu • (1 : Matrix ι ι R)) i j - (lam - mu) * (1 : Matrix ι ι R) i j := by
      rw [hent, hent]; ring
    rw [this]
    exact dvd_sub (h i j) (dvd_mul_of_dvd_left hc _)
  · rintro ⟨hc, h⟩ i j
    have : (T - mu • (1 : Matrix ι ι R)) i j =
        (T - lam • (1 : Matrix ι ι R)) i j + (lam - mu) * (1 : Matrix ι ι R) i j := by
      rw [hent, hent]; ring
    rw [this]
    exact dvd_add (h i j) (dvd_mul_of_dvd_left hc _)

/-! ### Valuations in `ℕ∞` -/

lemma enat_le_of_forall {x y : ℕ∞} (h : ∀ m : ℕ, (m : ℕ∞) ≤ y → (m : ℕ∞) ≤ x) : y ≤ x := by
  induction x using ENat.recTopCoe with
  | top => exact le_top
  | coe k =>
    by_contra hx
    rw [not_le] at hx
    have h1 : ((k + 1 : ℕ) : ℕ∞) ≤ y := by
      push_cast
      exact Order.add_one_le_of_lt hx
    have h2 := h (k + 1) h1
    have h3 : k + 1 ≤ k := by exact_mod_cast h2
    omega

lemma enat_eq_of_forall {x y : ℕ∞} (h : ∀ m : ℕ, (m : ℕ∞) ≤ x ↔ (m : ℕ∞) ≤ y) : x = y :=
  le_antisymm (enat_le_of_forall fun m hm => (h m).mp hm) (enat_le_of_forall fun m hm => (h m).mpr hm)

/-- The content valuation `min_{i,j} v_ϖ(A_{ij})` (`⊤` for `A = 0`). -/
noncomputable def content (ϖ : R) (A : Matrix ι ι R) : ℕ∞ := ⨅ i, ⨅ j, emultiplicity ϖ (A i j)

omit [Fintype ι] [DecidableEq ι] in
lemma le_content_iff (ϖ : R) (A : Matrix ι ι R) (m : ℕ) : (m : ℕ∞) ≤ content ϖ A ↔ ZeroMod A (ϖ ^ m) := by
  simp only [content, le_iInf_iff, ZeroMod, pow_dvd_iff_le_emultiplicity]

/-- A Smith-type decomposition `A = P · diag(d) · Q` with `P`, `Q` invertible (no divisibility chain is required). -/
structure SmithDecomp (A : Matrix ι ι R) where
  P : Matrix ι ι R
  P' : Matrix ι ι R
  Q : Matrix ι ι R
  Q' : Matrix ι ι R
  d : ι → R
  eq : A = P * diagonal d * Q
  hP : P' * P = 1
  hQ : Q * Q' = 1

namespace SmithDecomp

variable {A : Matrix ι ι R}

/-- The minimal elementary divisor `min_i v_ϖ(d_i)`, kernel divisors (`d_i = 0`) counted as `⊤` ([Rv] repair 1). -/
noncomputable def edMin (S : SmithDecomp A) (ϖ : R) : ℕ∞ := ⨅ i, emultiplicity ϖ (S.d i)

lemma le_edMin_iff (S : SmithDecomp A) (ϖ : R) (m : ℕ) : (m : ℕ∞) ≤ S.edMin ϖ ↔ ZeroMod A (ϖ ^ m) := by
  rw [edMin, le_iInf_iff, zeroMod_iff_of_decomp S.eq S.hP S.hQ, zeroMod_diagonal_iff]
  exact forall_congr' fun i => pow_dvd_iff_le_emultiplicity.symm

/-- **`min ed(A)` is the content of `A`**: independent of the decomposition. -/
theorem edMin_eq_content (S : SmithDecomp A) (ϖ : R) : S.edMin ϖ = content ϖ A :=
  enat_eq_of_forall fun m => by rw [le_edMin_iff, le_content_iff]

theorem edMin_eq (S S' : SmithDecomp A) (ϖ : R) : S.edMin ϖ = S'.edMin ϖ := by
  rw [edMin_eq_content, edMin_eq_content]

end SmithDecomp

/-! ### The `c_p` formula -/

/-- **`min ed(T − μ) = min(v_ϖ(λ − μ), min ed(T − λ))`** for an eigenvector `T e = λ e` with a unit coordinate. -/
theorem cp_eq_min {T : Matrix ι ι R} {e : ι → R} {lam mu : R} (he : T *ᵥ e = lam • e) {i₀ : ι} (hu : IsUnit (e i₀))
    (S : SmithDecomp (T - mu • (1 : Matrix ι ι R))) (S' : SmithDecomp (T - lam • (1 : Matrix ι ι R))) (ϖ : R) :
    S.edMin ϖ = min (emultiplicity ϖ (lam - mu)) (S'.edMin ϖ) := by
  refine enat_eq_of_forall fun m => ?_
  rw [S.le_edMin_iff, le_min_iff, ← pow_dvd_iff_le_emultiplicity, S'.le_edMin_iff]
  exact zeroMod_sub_iff he hu _

/-- Multiplying by a unit does not change `v_ϖ`. -/
lemma emultiplicity_unit_mul {p : R} (hp : IsUnit p) (ϖ x : R) : emultiplicity ϖ (p * x) = emultiplicity ϖ x :=
  enat_eq_of_forall fun m => by rw [← pow_dvd_iff_le_emultiplicity, ← pow_dvd_iff_le_emultiplicity, hp.dvd_mul_left]

/-- **The `c_p` formula** ([Rv] §1.2): if `T e = (1 + p^{n+1}) e` (the Eisenstein eigenvalue) with `e` primitive and
`p ∈ R^×` (`p ≠ ℓ`), then `min ed(T − (1 + p)) = min(v_ϖ(p^n − 1), min ed(T − (1 + p^{n+1})))`. -/
theorem cp_formula {T : Matrix ι ι R} {e : ι → R} {p : R} (hp : IsUnit p) {n : ℕ}
    (he : T *ᵥ e = (1 + p ^ (n + 1)) • e) {i₀ : ι} (hu : IsUnit (e i₀))
    (S : SmithDecomp (T - (1 + p) • (1 : Matrix ι ι R)))
    (S' : SmithDecomp (T - (1 + p ^ (n + 1)) • (1 : Matrix ι ι R))) (ϖ : R) :
    S.edMin ϖ = min (emultiplicity ϖ (p ^ n - 1)) (S'.edMin ϖ) := by
  rw [cp_eq_min he hu S S', show (1 + p ^ (n + 1)) - (1 + p) = p * (p ^ n - 1) by ring,
    emultiplicity_unit_mul hp]

/-- The decomposition-free form: `content(T − (1 + p)) = min(v_ϖ(p^n − 1), content(T − (1 + p^{n+1})))`. -/
theorem cp_formula_content {T : Matrix ι ι R} {e : ι → R} {p : R} (hp : IsUnit p) {n : ℕ}
    (he : T *ᵥ e = (1 + p ^ (n + 1)) • e) {i₀ : ι} (hu : IsUnit (e i₀)) (ϖ : R) :
    content ϖ (T - (1 + p) • (1 : Matrix ι ι R)) =
      min (emultiplicity ϖ (p ^ n - 1)) (content ϖ (T - (1 + p ^ (n + 1)) • (1 : Matrix ι ι R))) := by
  refine enat_eq_of_forall fun m => ?_
  rw [le_content_iff, le_min_iff, ← pow_dvd_iff_le_emultiplicity, le_content_iff, zeroMod_sub_iff he hu,
    show (1 + p ^ (n + 1)) - (1 + p) = p * (p ^ n - 1) by ring, hp.dvd_mul_left]

end GL2.L23.Cp
