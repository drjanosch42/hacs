/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# HeckeZ LC-10: the global certificate, 01 Thm. 4.1 over `ℤ` (L-HZ1, M-HZ2)

`docs/implementation/slot24/heckez/01-math-core.md` Thm. 4.1 (`thm:heckeZcert`), Prop. 2.2(1) (the (C3) read), Prop. 2.3(2)
in its `v_ℓ(D) = 0` clause (LC-3a, local form), Lemma 1.6(3) (local–global); `07-lean-plan.md` §2.2 LC-10, LC-3a, LC-2,
§5.1–5.2. Everything is over `ℤ`: `L` is a free `ℤ`-module of finite rank, `𝕋^fr ⊆ End_ℤ L` a subalgebra, `𝕋' ≤ 𝕋^fr`.

**The rank.** `RankLE 𝕋 d`: any `d + 1` elements of `𝕋` satisfy a non-trivial `ℤ`-relation (`rank 𝕋 ≤ d`). At level 1 it
comes from Eichler–Shimura (`GL2.HeckeZ.RankES`), never from the computed data (01 §4: "the rank must come from Prop. 1.3").

**The local property.** `Loc 𝕋^fr 𝕋' ℓ`: every `x ∈ 𝕋^fr` has a multiple `m x ∈ 𝕋'` with `ℓ ∤ m`, i.e. `𝕋'_(ℓ) = 𝕋^fr_(ℓ)`
(equivalently `ℓ ∤ [𝕋^fr : 𝕋']`, and `𝕋' ⊗ ℤ_ℓ = 𝕋^fr ⊗ ℤ_ℓ`). `le_of_loc`: if it holds at every prime, `𝕋^fr = 𝕋'`
(01 Lemma 1.6(3), the local–global step of Thm. 4.1).

**The one local mechanism** (`loc_of_modIndep`): if `d` elements of `𝕋'` stay linearly independent modulo `ℓ` (in
`End(L/ℓ)`) and `rank 𝕋^fr ≤ d`, then `Loc 𝕋^fr 𝕋' ℓ`. Proof: a relation among those `d` elements and `x`, made primitive
at `ℓ`, has an `x`-coefficient prime to `ℓ`. Both routes of Thm. 4.1 feed it:
* **(C3) / K1** (`GL2.HeckeZ.Check`): `d` monomials in the `T_p`, independent mod `ℓ` by a pivot certificate (01 Prop.
  2.2(1): `dim A'_1(L/ℓ) = d`);
* **(C2) / K2g at `ℓ ∤ D`** (`modIndep_pow`, LC-3a in local form): `1, θ, …, θ^{d−1}` are independent mod `ℓ` when `f`
  is monic of degree `d`, `f ∣ χ_θ`, and `u f + v f' = D` with `ℓ ∤ D`. Proof: `g(θ) = ℓ y` makes `y` integral (Cayley–
  Hamilton), so `g(θ)^m ≡ 0` modulo `ℓ` in `ℤ[θ]`, i.e. `f̄ ∣ ḡ^m`; `f̄` is separable, so `f̄ ∣ ḡ`, and `deg ḡ < d` forces
  `ḡ = 0`.

**(C1)** (`dvd_of_aeval_eq_zero`): if `f` is monic, squarefree over `ℚ` and divides `χ_θ`, every `g` with `g(θ) = 0` is
divisible by `f` (from `χ_θ ∣ g^{rank L}`, the adjugate identity `charpoly_dvd_pow`). With `deg f = d ≥ rank 𝕋^fr` this gives
`f(θ) = 0` and `rank ℤ[θ] = d` (`aeval_eq_zero_of_C1`): 01's (C1), `m_θ = f`, by the degree count of v0.5 (P2a-6), with no
semisimplicity hypothesis.

**The theorem** (`heckeZcert`): (C1) for the `θ_j`, the (C3) reads at a set `S` of primes, and (C2) in the form "every
prime `ℓ ∤ c` either misses some `D_j` or lies in `S`" give `Loc` at every prime `ℓ ∤ c`, and `𝕋^fr = 𝕋'` when `c = ±1`.
The `v_ℓ(D) = 1` clause (LC-3b) is not used: such primes are put into `S` (01 Thm. 4.1, "Lean form").
-/

namespace GL2.HeckeZ.Cert

open Polynomial

/-! ### The adjugate identity: `χ_M ∣ g^n` whenever `g(M) = 0` -/

/-- **`χ_M ∣ g^{#n}` for every `g` with `g(M) = 0`** (`g(X)·1 = Q(X)·(X·1 − M)` by the geometric sum of the commuting
`X·1` and `M`, then take determinants). -/
theorem charpoly_dvd_pow {n : Type*} [Fintype n] [DecidableEq n] {R : Type*} [CommRing R] (M : Matrix n n R)
    (g : R[X]) (hg : aeval M g = 0) : M.charpoly ∣ g ^ Fintype.card n := by
  set x : Matrix n n R[X] := algebraMap R[X] (Matrix n n R[X]) X with hx
  set y : Matrix n n R[X] := (C : R →+* R[X]).mapMatrix M with hy
  have hxy : Commute x y := Algebra.commutes _ _
  have hchar : Matrix.charmatrix M = x - y := by
    ext i j
    simp [Matrix.charmatrix, hx, hy, Matrix.algebraMap_eq_diagonal, Matrix.scalar_apply, Matrix.diagonal_apply]
  set Q : Matrix n n R[X] := ∑ k ∈ Finset.range (g.natDegree + 1),
      (C (g.coeff k) : R[X]) • ∑ i ∈ Finset.range k, x ^ i * y ^ (k - 1 - i) with hQdef
  have hg2 : ∀ k, (∑ i ∈ Finset.range k, x ^ i * y ^ (k - 1 - i)) * (x - y) = x ^ k - y ^ k :=
    fun k => hxy.geom_sum₂_mul k
  have e1 : Q * (x - y) = ∑ k ∈ Finset.range (g.natDegree + 1), (C (g.coeff k) : R[X]) • (x ^ k - y ^ k) := by
    rw [hQdef, Finset.sum_mul]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [Matrix.smul_mul, hg2]
  have e2 : ∑ k ∈ Finset.range (g.natDegree + 1), (C (g.coeff k) : R[X]) • y ^ k = 0 := by
    have hk : ∀ k, (C (g.coeff k) : R[X]) • y ^ k = (C : R →+* R[X]).mapMatrix (g.coeff k • M ^ k) := by
      intro k
      rw [hy, ← map_pow]
      ext i j
      simp [Matrix.smul_apply, smul_eq_mul]
    simp_rw [hk, ← map_sum, ← aeval_eq_sum_range, hg, map_zero]
  have e3 : ∑ k ∈ Finset.range (g.natDegree + 1), (C (g.coeff k) : R[X]) • x ^ k =
      algebraMap R[X] (Matrix n n R[X]) g := by
    have hk : ∀ k, (C (g.coeff k) : R[X]) • x ^ k = algebraMap R[X] (Matrix n n R[X]) (C (g.coeff k) * X ^ k) := by
      intro k
      rw [hx, ← map_pow, Algebra.smul_def, ← map_mul]
    simp_rw [hk, ← map_sum]
    rw [← as_sum_range_C_mul_X_pow g]
  have hQ : Q * Matrix.charmatrix M = algebraMap R[X] (Matrix n n R[X]) g := by
    rw [hchar, e1]
    simp only [smul_sub, Finset.sum_sub_distrib, e2, e3, sub_zero]
  have hdet := congrArg Matrix.det hQ
  rw [Matrix.det_mul, Algebra.algebraMap_eq_smul_one, Matrix.det_smul, Matrix.det_one, mul_one] at hdet
  exact ⟨Q.det, by rw [← hdet, Matrix.charpoly, mul_comm]⟩

variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]

/-- The adjugate identity for an endomorphism of a free `ℤ`-module: `χ_θ ∣ g^{rank L}` whenever `g(θ) = 0`. -/
theorem charpoly_dvd_pow_of_aeval (θ : Module.End ℤ L) (g : ℤ[X]) (hg : aeval θ g = 0) :
    θ.charpoly ∣ g ^ Module.finrank ℤ L := by
  classical
  let b := Module.Free.chooseBasis ℤ L
  have h1 : aeval (LinearMap.toMatrix b b θ) g = 0 := by
    have := Polynomial.aeval_algHom_apply (LinearMap.toMatrixAlgEquiv b).toAlgHom θ g
    rw [hg, map_zero] at this
    rw [← this]; rfl
  have h2 := charpoly_dvd_pow (LinearMap.toMatrix b b θ) g h1
  rwa [LinearMap.charpoly_toMatrix, ← Module.finrank_eq_card_chooseBasisIndex] at h2

/-- A Bezout identity `u f + v f' = D` with `φ(D) ≠ 0` makes `f` separable, hence squarefree, over the field `K`. -/
theorem squarefree_map_of_bezout {K : Type*} [Field K] (φ : ℤ →+* K) (f u v : ℤ[X]) (D : ℤ)
    (h : u * f + v * derivative f = C D) (hD : φ D ≠ 0) : Squarefree (f.map φ) := by
  apply Polynomial.Separable.squarefree
  refine ⟨C (φ D)⁻¹ * u.map φ, C (φ D)⁻¹ * v.map φ, ?_⟩
  have h' := congrArg (Polynomial.map φ) h
  simp only [Polynomial.map_add, Polynomial.map_mul, Polynomial.map_C] at h'
  rw [Polynomial.derivative_map]
  calc C (φ D)⁻¹ * u.map φ * f.map φ + C (φ D)⁻¹ * v.map φ * (derivative f).map φ
      = C (φ D)⁻¹ * (u.map φ * f.map φ + v.map φ * (derivative f).map φ) := by ring
    _ = 1 := by rw [h', ← C_mul, inv_mul_cancel₀ hD, C_1]

/-- **(C1), the divisibility form**: if `f` is monic, squarefree over `ℚ`, and divides `χ_θ`, then `f ∣ g` for every `g`
with `g(θ) = 0`. -/
theorem dvd_of_aeval_eq_zero (θ : Module.End ℤ L) (f : ℤ[X]) (hf : f.Monic)
    (hsq : Squarefree (f.map (Int.castRingHom ℚ))) (hfχ : f ∣ θ.charpoly) {g : ℤ[X]} (hg : aeval θ g = 0) :
    f ∣ g := by
  have h1 : f ∣ g ^ Module.finrank ℤ L := hfχ.trans (charpoly_dvd_pow_of_aeval θ g hg)
  rw [← Polynomial.map_dvd_map (Int.castRingHom ℚ) (Int.castRingHom ℚ).injective_int hf]
  have h2 := Polynomial.map_dvd (Int.castRingHom ℚ) h1
  rw [Polynomial.map_pow] at h2
  rcases Nat.eq_zero_or_pos (Module.finrank ℤ L) with h0 | hpos
  · rw [h0, pow_zero] at h2
    exact (isUnit_of_dvd_one h2).dvd
  · exact (hsq.dvd_pow_iff_dvd hpos.ne').mp h2

/-! ### Rank, independence modulo `ℓ`, and the local property -/

/-- `rank 𝕋 ≤ d`: any `d + 1` elements of `𝕋` satisfy a non-trivial `ℤ`-relation. -/
def RankLE (T : Subalgebra ℤ (Module.End ℤ L)) (d : ℕ) : Prop :=
  ∀ v : Fin (d + 1) → Module.End ℤ L, (∀ i, v i ∈ T) → ∃ c : Fin (d + 1) → ℤ, ∑ i, c i • v i = 0 ∧ ∃ i, c i ≠ 0

/-- `t₀, …, t_{d−1}` are linearly independent modulo `ℓ`: a combination lying in `ℓ · End(L)` has all coefficients
divisible by `ℓ` (their images in `End(L/ℓ)` are `𝔽_ℓ`-independent). -/
def ModIndep (ℓ : ℤ) {d : ℕ} (t : Fin d → Module.End ℤ L) : Prop :=
  ∀ a : Fin d → ℤ, (∃ y : Module.End ℤ L, ∑ k, a k • t k = ℓ • y) → ∀ k, ℓ ∣ a k

/-- `𝕋'_(ℓ) = 𝕋^fr_(ℓ)`: every `x ∈ 𝕋^fr` has a multiple `m x ∈ 𝕋'` with `ℓ ∤ m`. -/
def Loc (Tfr T' : Subalgebra ℤ (Module.End ℤ L)) (ℓ : ℤ) : Prop :=
  ∀ x ∈ Tfr, ∃ m : ℤ, ¬ ℓ ∣ m ∧ m • x ∈ T'

/-- `End_ℤ L` is torsion-free. -/
theorem end_smul_eq_zero {m : ℤ} (hm : m ≠ 0) {y : Module.End ℤ L} (h : m • y = 0) : y = 0 := by
  ext v
  have := LinearMap.congr_fun h v
  rw [LinearMap.smul_apply, LinearMap.zero_apply] at this
  rcases smul_eq_zero.mp this with h0 | h0
  · exact absurd h0 hm
  · simpa using h0

/-- A non-trivial relation can be made primitive at `ℓ` (divide by `ℓ` while possible). -/
theorem exists_rel_not_dvd {ι : Type*} [Fintype ι] {ℓ : ℕ} (hℓ : 2 ≤ ℓ) (v : ι → Module.End ℤ L)
    (c : ι → ℤ) (hc : ∑ i, c i • v i = 0) (hne : ∃ i, c i ≠ 0) :
    ∃ c' : ι → ℤ, ∑ i, c' i • v i = 0 ∧ ∃ i, ¬ (ℓ : ℤ) ∣ c' i := by
  suffices H : ∀ N : ℕ, ∀ c : ι → ℤ, ∑ i, (c i).natAbs = N → ∑ i, c i • v i = 0 → (∃ i, c i ≠ 0) →
      ∃ c' : ι → ℤ, ∑ i, c' i • v i = 0 ∧ ∃ i, ¬ (ℓ : ℤ) ∣ c' i from H _ c rfl hc hne
  intro N
  induction N using Nat.strong_induction_on with
  | _ N ih =>
  intro c hN hc hne
  by_cases hall : ∀ i, (ℓ : ℤ) ∣ c i
  · choose c' hc' using hall
    have h0 : (ℓ : ℤ) • ∑ i, c' i • v i = 0 := by
      rw [Finset.smul_sum]
      simp_rw [smul_smul, ← hc']
      exact hc
    have hrel := end_smul_eq_zero (by exact_mod_cast (show ℓ ≠ 0 by omega)) h0
    obtain ⟨i, hi⟩ := hne
    have hi' : c' i ≠ 0 := by
      intro h0'; apply hi; rw [hc' i, h0', mul_zero]
    have habs : ∀ j, (c j).natAbs = ℓ * (c' j).natAbs := fun j => by
      rw [hc' j, Int.natAbs_mul, Int.natAbs_natCast]
    have hpos : 0 < ∑ j, (c' j).natAbs :=
      Finset.sum_pos' (fun j _ => Nat.zero_le _) ⟨i, Finset.mem_univ _, Int.natAbs_pos.mpr hi'⟩
    refine ih (∑ j, (c' j).natAbs) ?_ c' rfl hrel ⟨i, hi'⟩
    rw [← hN]
    simp_rw [habs]
    rw [← Finset.mul_sum]
    nlinarith
  · push Not at hall
    exact ⟨c, hc, hall⟩

/-- **The local mechanism**: `d` elements of `𝕋'` independent modulo `ℓ`, and `rank 𝕋^fr ≤ d`, give `𝕋'_(ℓ) = 𝕋^fr_(ℓ)`
(01 Prop. 2.2(1) through (C3), and Prop. 2.3(2) at `v_ℓ(D) = 0`). -/
theorem loc_of_modIndep {Tfr T' : Subalgebra ℤ (Module.End ℤ L)} (hT : T' ≤ Tfr) {d : ℕ} (hrank : RankLE Tfr d)
    {ℓ : ℕ} (hℓ : ℓ.Prime) (t : Fin d → Module.End ℤ L) (htT : ∀ k, t k ∈ T') (hind : ModIndep (ℓ : ℤ) t) :
    Loc Tfr T' ℓ := by
  intro x hx
  have hv : ∀ i, (Fin.snoc t x : Fin (d + 1) → Module.End ℤ L) i ∈ Tfr := by
    intro i
    induction i using Fin.lastCases with
    | last => simpa using hx
    | cast k => simpa using hT (htT k)
  obtain ⟨c, hc, hne⟩ := hrank _ hv
  obtain ⟨c', hc', i0, hi0⟩ := exists_rel_not_dvd hℓ.two_le _ c hc hne
  rw [Fin.sum_univ_castSucc] at hc'
  simp only [Fin.snoc_castSucc, Fin.snoc_last] at hc'
  refine ⟨c' (Fin.last d), fun hdvd => ?_, ?_⟩
  · obtain ⟨e, he⟩ := hdvd
    have hrel : ∑ k : Fin d, c' k.castSucc • t k = (ℓ : ℤ) • (-(e • x)) := by
      rw [smul_neg, smul_smul, ← he]
      exact eq_neg_of_add_eq_zero_left hc'
    have hall := hind (fun k => c' k.castSucc) ⟨_, hrel⟩
    apply hi0
    induction i0 using Fin.lastCases with
    | last => exact ⟨e, he⟩
    | cast k => exact hall k
  · have h1 : c' (Fin.last d) • x = -∑ k : Fin d, c' k.castSucc • t k := eq_neg_of_add_eq_zero_right hc'
    rw [h1]
    exact T'.neg_mem (T'.sum_mem fun k _ => Subalgebra.smul_mem _ (htT k) _)

/-- **Local–global** (01 Lemma 1.6(3)): `Loc` at every prime gives `𝕋^fr ≤ 𝕋'`. -/
theorem le_of_loc {Tfr T' : Subalgebra ℤ (Module.End ℤ L)} (h : ∀ ℓ : ℕ, ℓ.Prime → Loc Tfr T' ℓ) : Tfr ≤ T' := by
  classical
  intro x hx
  have hex : ∃ m : ℕ, 0 < m ∧ ((m : ℤ) • x ∈ T') := by
    obtain ⟨m, hm, hmx⟩ := h 2 Nat.prime_two x hx
    have hm0 : m ≠ 0 := by rintro rfl; exact hm (dvd_zero _)
    refine ⟨m.natAbs, Int.natAbs_pos.mpr hm0, ?_⟩
    rcases Int.natAbs_eq m with he | he
    · rw [← he]; exact hmx
    · have : ((m.natAbs : ℕ) : ℤ) = -m := by omega
      rw [this, neg_smul]; exact T'.neg_mem hmx
  obtain ⟨hm0pos, hm0x⟩ := Nat.find_spec hex
  set m0 := Nat.find hex with hm0def
  by_cases h1 : m0 = 1
  · have : ((m0 : ℕ) : ℤ) • x = x := by rw [h1]; simp
    rw [← this]; exact hm0x
  · exfalso
    obtain ⟨ℓ, hℓ, hℓm⟩ := Nat.exists_prime_and_dvd h1
    obtain ⟨m, hm, hmx⟩ := h ℓ hℓ x hx
    set g := Int.gcd m (m0 : ℤ) with hg
    have hgx : ((g : ℕ) : ℤ) • x ∈ T' := by
      rw [hg, Int.gcd_eq_gcd_ab, add_smul, mul_smul, mul_smul]
      exact T'.add_mem (by rw [smul_comm]; exact Subalgebra.smul_mem _ hmx _)
        (by rw [smul_comm]; exact Subalgebra.smul_mem _ hm0x _)
    have hgpos : 0 < g := Int.gcd_pos_of_ne_zero_right _ (by exact_mod_cast hm0pos.ne')
    have hgle : g ≤ m0 := Nat.le_of_dvd hm0pos (by
      have := Int.gcd_dvd_right m (m0 : ℤ)
      exact Int.natCast_dvd_natCast.mp (by simpa using this))
    have hgne : g ≠ m0 := by
      intro hgm
      apply hm
      have h2 : (m0 : ℤ) ∣ m := by rw [← hgm]; exact Int.gcd_dvd_left m (m0 : ℤ)
      exact (Int.natCast_dvd_natCast.mpr hℓm).trans h2
    exact (Nat.find_min hex (lt_of_le_of_ne hgle hgne)) ⟨hgpos, hgx⟩

/-! ### LC-3a, local form: `ℓ ∤ D` makes `1, θ, …, θ^{d−1}` independent modulo `ℓ` -/

/-- **LC-3a (local)**: `f` monic of degree `d`, `f ∣ χ_θ`, `u f + v f' = D`, `ℓ ∤ D` ⇒ `θ^0, …, θ^{d−1}` are linearly
independent modulo `ℓ`. -/
theorem modIndep_pow (θ : Module.End ℤ L) (f u v : ℤ[X]) (D : ℤ) {d : ℕ} (hf : f.Monic) (hdeg : f.natDegree = d)
    (hfχ : f ∣ θ.charpoly) (hB : u * f + v * derivative f = C D) (hD : D ≠ 0) {ℓ : ℕ} (hℓ : ℓ.Prime)
    (hℓD : ¬ (ℓ : ℤ) ∣ D) : ModIndep (ℓ : ℤ) (fun k : Fin d => θ ^ (k : ℕ)) := by
  have := Fact.mk hℓ
  rintro a ⟨y, hy⟩ k
  -- `d ≤ rank L`
  have hdle : d ≤ Module.finrank ℤ L := by
    rw [← hdeg, ← LinearMap.charpoly_natDegree θ]
    exact Polynomial.natDegree_le_of_dvd hfχ (LinearMap.charpoly_monic θ).ne_zero
  set g : ℤ[X] := ∑ i : Fin d, C (a i) * X ^ (i : ℕ) with hgdef
  have hgθ : aeval θ g = (ℓ : ℤ) • y := by
    rw [← hy, hgdef, map_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [map_mul, aeval_C, map_pow, aeval_X, Algebra.smul_def]
  -- `y` is integral: `y^m + … = 0` with `m = rank L`
  set P := y.charpoly with hP
  set m := P.natDegree with hm
  have hmL : m = Module.finrank ℤ L := LinearMap.charpoly_natDegree y
  set h : ℤ[X] := ∑ i ∈ Finset.range (m + 1), C (P.coeff i * (ℓ : ℤ) ^ (m - i)) * g ^ i with hhdef
  have hh : aeval θ h = 0 := by
    have e : aeval θ h = (ℓ : ℤ) ^ m • aeval y P := by
      rw [aeval_eq_sum_range (p := P) y, Finset.smul_sum, hhdef, map_sum]
      refine Finset.sum_congr rfl fun i hi => ?_
      have hi' : i ≤ m := Nat.lt_succ_iff.mp (Finset.mem_range.mp hi)
      rw [map_mul, aeval_C, map_pow, hgθ, _root_.smul_pow, ← Algebra.smul_def, smul_smul, smul_smul, mul_assoc,
        ← pow_add, Nat.sub_add_cancel hi', mul_comm]
    rw [e, LinearMap.aeval_self_charpoly, smul_zero]
  have hsqQ : Squarefree (f.map (Int.castRingHom ℚ)) :=
    squarefree_map_of_bezout _ f u v D hB (by simpa using hD)
  have hfh : f ∣ h := dvd_of_aeval_eq_zero θ f hf hsqQ hfχ hh
  -- reduce modulo `ℓ`
  set φ : ℤ →+* ZMod ℓ := Int.castRingHom (ZMod ℓ) with hφ
  have hhmap : h.map φ = (g.map φ) ^ m := by
    rw [hhdef, Polynomial.map_sum, Finset.sum_range_succ, Finset.sum_eq_zero, zero_add]
    · rw [Polynomial.map_mul, Polynomial.map_pow, Polynomial.map_C, Nat.sub_self, pow_zero, mul_one]
      have : P.coeff m = 1 := (LinearMap.charpoly_monic y).coeff_natDegree
      rw [this, map_one, C_1, one_mul]
    · intro i hi
      have hlt : 0 < m - i := Nat.sub_pos_of_lt (Finset.mem_range.mp hi)
      rw [Polynomial.map_mul, Polynomial.map_C, map_mul, map_pow]
      simp [hφ, zero_pow hlt.ne']
  have hfh' : f.map φ ∣ (g.map φ) ^ m := by
    rw [← hhmap]; exact Polynomial.map_dvd φ hfh
  have hsqℓ : Squarefree (f.map φ) :=
    squarefree_map_of_bezout φ f u v D hB (by
      rw [hφ, eq_intCast, Ne, ZMod.intCast_zmod_eq_zero_iff_dvd]; exact hℓD)
  have hdpos : 0 < d := Nat.pos_of_ne_zero (fun h0 => by subst h0; exact k.elim0)
  have hfg : f.map φ ∣ g.map φ := (hsqℓ.dvd_pow_iff_dvd (by omega)).mp hfh'
  have hg0 : g.map φ = 0 := by
    apply Polynomial.eq_zero_of_dvd_of_degree_lt hfg
    have hdf : (f.map φ).degree = d := by
      rw [Polynomial.degree_eq_natDegree (hf.map φ).ne_zero, hf.natDegree_map, hdeg]
    rw [hdf, hgdef]
    exact lt_of_le_of_lt Polynomial.degree_map_le (Polynomial.degree_sum_fin_lt a)
  have hcoef := congrArg (fun p => p.coeff (k : ℕ)) hg0
  simp only [Polynomial.coeff_map, Polynomial.coeff_zero] at hcoef
  have hgk : g.coeff (k : ℕ) = a k := by
    rw [hgdef, Polynomial.finsetSum_coeff]
    simp only [Polynomial.coeff_C_mul_X_pow]
    rw [Finset.sum_eq_single k]
    · simp
    · intro b _ hb; rw [if_neg]; exact fun h' => hb (Fin.ext h'.symm)
    · simp
  rw [hgk, hφ, eq_intCast, ZMod.intCast_zmod_eq_zero_iff_dvd] at hcoef
  exact hcoef

/-! ### (C1) in full, and the theorem -/

/-- Coefficients of `Σ_{i<d} a_i X^i`. -/
lemma coeff_sum_fin {d : ℕ} (a : Fin d → ℤ) (k : Fin d) :
    (∑ i : Fin d, C (a i) * X ^ (i : ℕ)).coeff (k : ℕ) = a k := by
  rw [Polynomial.finsetSum_coeff]
  simp only [Polynomial.coeff_C_mul_X_pow]
  rw [Finset.sum_eq_single k]
  · simp
  · intro b _ hb; rw [if_neg]; exact fun h' => hb (Fin.ext h'.symm)
  · simp

/-- A polynomial of degree `< d` divisible by a monic `f` of degree `d` vanishes. -/
lemma eq_zero_of_dvd_sum {d : ℕ} {f : ℤ[X]} (hf : f.Monic) (hdeg : f.natDegree = d) (a : Fin d → ℤ)
    (h : f ∣ ∑ i : Fin d, C (a i) * X ^ (i : ℕ)) : a = 0 := by
  have hg0 := Polynomial.eq_zero_of_dvd_of_degree_lt h (by
    rw [Polynomial.degree_eq_natDegree hf.ne_zero, hdeg]; exact Polynomial.degree_sum_fin_lt a)
  funext k
  have := congrArg (fun p => p.coeff (k : ℕ)) hg0
  simp only [coeff_sum_fin, Polynomial.coeff_zero] at this
  exact this

/-- **(C1)** (01 Thm. 4.1 (C1), v0.5 P2a-6: the degree count): `f` monic of degree `d`, `f ∣ χ_θ`, `u f + v f' = D ≠ 0`,
`θ ∈ 𝕋` and `rank 𝕋 ≤ d` give `f(θ) = 0` and the independence of `1, θ, …, θ^{d−1}`: `m_θ = f`, `rank ℤ[θ] = d`. -/
theorem aeval_eq_zero_of_C1 {T : Subalgebra ℤ (Module.End ℤ L)} {d : ℕ} (hrank : RankLE T d)
    (θ : Module.End ℤ L) (hθ : θ ∈ T) (f u v : ℤ[X]) (D : ℤ) (hf : f.Monic) (hdeg : f.natDegree = d)
    (hfχ : f ∣ θ.charpoly) (hB : u * f + v * derivative f = C D) (hD : D ≠ 0) :
    aeval θ f = 0 ∧ ∀ a : Fin d → ℤ, ∑ k, a k • θ ^ (k : ℕ) = 0 → a = 0 := by
  have hsq := squarefree_map_of_bezout (Int.castRingHom ℚ) f u v D hB (by simpa using hD)
  have hev : ∀ a : Fin d → ℤ, aeval θ (∑ i : Fin d, C (a i) * X ^ (i : ℕ)) = ∑ k, a k • θ ^ (k : ℕ) := by
    intro a
    rw [map_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [map_mul, aeval_C, map_pow, aeval_X, Algebra.smul_def]
  have hind : ∀ a : Fin d → ℤ, ∑ k, a k • θ ^ (k : ℕ) = 0 → a = 0 := by
    intro a ha
    rw [← hev] at ha
    exact eq_zero_of_dvd_sum hf hdeg a (dvd_of_aeval_eq_zero θ f hf hsq hfχ ha)
  refine ⟨?_, hind⟩
  have hfT : aeval θ f ∈ T := by
    rw [← Polynomial.aeval_subalgebra_coe f T ⟨θ, hθ⟩]; exact SetLike.coe_mem _
  obtain ⟨c, hc, i0, hi0⟩ := hrank (Fin.snoc (fun k : Fin d => θ ^ (k : ℕ)) (aeval θ f)) (by
    intro i
    induction i using Fin.lastCases with
    | last => simpa using hfT
    | cast k => simpa using T.pow_mem hθ _)
  rw [Fin.sum_univ_castSucc] at hc
  simp only [Fin.snoc_castSucc, Fin.snoc_last] at hc
  have hg : aeval θ (∑ i : Fin d, C (c i.castSucc) * X ^ (i : ℕ) + C (c (Fin.last d)) * f) = 0 := by
    rw [map_add, hev, map_mul, aeval_C, ← Algebra.smul_def]; exact hc
  have hfg : f ∣ ∑ i : Fin d, C (c i.castSucc) * X ^ (i : ℕ) :=
    (dvd_add_left (dvd_mul_left f _)).mp (dvd_of_aeval_eq_zero θ f hf hsq hfχ hg)
  have hc' := eq_zero_of_dvd_sum hf hdeg _ hfg
  have hc'' : ∀ k : Fin d, c k.castSucc = 0 := fun k => congrFun hc' k
  have hlast : c (Fin.last d) ≠ 0 := by
    intro h0; apply hi0
    induction i0 using Fin.lastCases with
    | last => exact h0
    | cast k => exact hc'' k
  have h0 : c (Fin.last d) • aeval θ f = 0 := by
    have : ∑ k : Fin d, c k.castSucc • θ ^ (k : ℕ) = 0 := by simp [hc'']
    rw [this, zero_add] at hc; exact hc
  exact end_smul_eq_zero hlast h0

/-- **01 Thm. 4.1 (`thm:heckeZcert`, LC-10), the `ℤ`-form.** `𝕋' ≤ 𝕋^fr`, `rank 𝕋^fr ≤ d` (from Prop. 1.3 / `hES`, never
from the data); (C1) for elements `θ_j ∈ 𝕋'`: `f_j` monic of degree `d` dividing `χ_{θ_j}`, with a Bezout identity
`u_j f_j + v_j f_j' = D_j ≠ 0`; (C3) at every `ℓ ∈ S`: `d` elements of `𝕋'` independent modulo `ℓ`; (C2) with cofactor `c`:
every prime `ℓ ∤ c` misses some `D_j` or lies in `S`. Then `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ c`, and `𝕋^fr = 𝕋'`
when `c = 1`. -/
theorem heckeZcert {Tfr T' : Subalgebra ℤ (Module.End ℤ L)} (hT : T' ≤ Tfr) {d : ℕ} (hrank : RankLE Tfr d)
    {J : Type*} (θ : J → Module.End ℤ L) (hθ : ∀ j, θ j ∈ T') (f u v : J → ℤ[X]) (D : J → ℤ)
    (hf : ∀ j, (f j).Monic) (hdeg : ∀ j, (f j).natDegree = d) (hfχ : ∀ j, f j ∣ (θ j).charpoly)
    (hB : ∀ j, u j * f j + v j * derivative (f j) = C (D j)) (hD : ∀ j, D j ≠ 0)
    (S : Set ℕ) (hC3 : ∀ ℓ ∈ S, ℓ.Prime → ∃ t : Fin d → Module.End ℤ L, (∀ k, t k ∈ T') ∧ ModIndep (ℓ : ℤ) t)
    (c : ℕ) (hC2 : ∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ c → (∃ j, ¬ (ℓ : ℤ) ∣ D j) ∨ ℓ ∈ S) :
    (∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ c → Loc Tfr T' ℓ) ∧ (c = 1 → Tfr = T') := by
  have hloc : ∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ c → Loc Tfr T' ℓ := by
    intro ℓ hℓ hℓc
    rcases hC2 ℓ hℓ hℓc with ⟨j, hj⟩ | hS
    · exact loc_of_modIndep hT hrank hℓ _ (fun k => T'.pow_mem (hθ j) _)
        (modIndep_pow (θ j) (f j) (u j) (v j) (D j) (hf j) (hdeg j) (hfχ j) (hB j) (hD j) hℓ hj)
    · obtain ⟨t, ht, hind⟩ := hC3 ℓ hS hℓ
      exact loc_of_modIndep hT hrank hℓ t ht hind
  exact ⟨hloc, fun h1 => le_antisymm (le_of_loc fun ℓ hℓ => hloc ℓ hℓ (by rw [h1]; exact hℓ.not_dvd_one)) hT⟩

end GL2.HeckeZ.Cert
