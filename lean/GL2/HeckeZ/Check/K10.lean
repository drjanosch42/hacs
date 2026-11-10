/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.K10
import GL2.L23.SmithPK
import GL2.Literature.MazurEis
import Mathlib.LinearAlgebra.Charpoly.Basic
import Mathlib.FieldTheory.Finite.Basic

/-!
# L-K10: K10′ is sound, and the valuation check certifies the Eisenstein block (M-HZ4a, partly)

`docs/implementation/slot24/heckez/07-lean-plan.md` §5.10 (K10′ with A7-M1), [C] `prop:heckeZeiscert`(3), `slot25/H3_referee.md`
note 5. Brief `slot25/LK10_brief.md`.

* **Lists → `ℤ[X]`** (`toPoly`, `toPoly_pmul`, `peval_eq`, `toPoly_plin`, `peqMod_sound`): the executable polynomial arithmetic
  of `GL2/HeckeZ/Exec/K10.lean` is exact integer polynomial arithmetic.
* **`goodB_sound`**: the two tests of K10′ imply Mazur's definition of a good prime, read literally (`MazurGood`).
* **`k10p_sound`**: an accepted certificate gives the guards, goodness, `v = v_ℓ(num((N−1)/12))`, the Hensel factorisation
  `h ≡ F·H (mod ℓ^K)` with `F` monic of degree `e`, `F ≡ (x − 1 − q)^e`, `H(1+q) ≢ 0 (mod ℓ)`, and for **every** lift `F̃ ∈ ℤ_ℓ[x]`
  of `F mod ℓ^K`: `ℓ^v ∣ F̃(1+q)` and `ℓ^{v+1} ∤ F̃(1+q)`.
* **The core (note 5).** `charpoly_eval_mem` (Cayley–Hamilton): for `T` free of finite rank over `R`, `t ∈ T` and `a ∈ R` with
  `a − t ∈ I`, the characteristic polynomial `f` of `t` has `f(a) ↦ 0` in `T/I`. With `hMazurEis` (`MazurEisLocal`, field `quot`):
  `ℓ^{v_ℓ(n)} ∣ f_{q,𝔪}(1+q)` for **every** prime `q ≠ N` (`mazur_dvd_charpoly`; `η_q ∈ 𝔍`, Mazur II.9.7; no goodness).
  `no_contamination`: if `F̃ = f_{q,𝔪} · G` with `G` monic and `G ≡ (x − 1 − q)^d (mod ℓ)` (the factors of the other maximal ideals
  through `(ℓ, η_q)`) and `ℓ^{v+1} ∤ F̃(1+q)`, then `G = 1`.
* **`mazur_val_cert`** (proved mod `hMazurEis`): an accepted K10′ certificate, Mazur's ring `T` for `(N, ℓ)`, and the **block
  identification** `hblock` (a lift `F̃` of the certificate's `F` equals `f_{q,𝔪} · G` as above) give: `G = 1`, so `F̃ = f_{q,𝔪}`
  is the characteristic polynomial of `T_q` on `T`, `rank_ℤ_ℓ T = e`, `ℓ^v ∥ f_{q,𝔪}(1+q)`, and (goodness, (16.2), (16.6))
  `T = ℤ_ℓ[η_q]`, `𝔍T = (η_q)`. `hblock` — the semilocal decomposition of `𝕋⁰ ⊗ ℤ_ℓ` and Hensel uniqueness mod `ℓ^K` — is the
  remaining input of M-HZ4a; it is an explicit hypothesis, not a literature structure.
-/

namespace GL2.HeckeZ.K10

open Polynomial GL2.HeckeZ.Exec.K10 GL2.Literature

/-! ### Lists as integer polynomials -/

/-- The integer polynomial of a coefficient list (lowest degree first). -/
noncomputable def toPoly : List ℤ → ℤ[X]
  | [] => 0
  | x :: a => C x + X * toPoly a

@[simp] lemma toPoly_nil : toPoly [] = 0 := rfl
@[simp] lemma toPoly_cons (x : ℤ) (a : List ℤ) : toPoly (x :: a) = C x + X * toPoly a := rfl

lemma coeff_toPoly : ∀ (p : List ℤ) (i : ℕ), (toPoly p).coeff i = coeffD p i
  | [], i => by simp [coeffD]
  | x :: a, 0 => by simp [coeffD]
  | x :: a, i + 1 => by
      simp only [toPoly_cons, coeff_add, coeff_C, coeff_X_mul, coeff_toPoly a i]
      simp [coeffD]

lemma toPoly_padd : ∀ a b : List ℤ, toPoly (padd a b) = toPoly a + toPoly b
  | [], b => by simp [padd]
  | x :: a, [] => by simp [padd]
  | x :: a, y :: b => by simp only [padd, toPoly_cons, toPoly_padd a b, C_add]; ring

lemma toPoly_map_mul (x : ℤ) : ∀ b : List ℤ, toPoly (b.map (x * ·)) = C x * toPoly b
  | [] => by simp
  | y :: b => by simp only [List.map_cons, toPoly_cons, toPoly_map_mul x b, C_mul]; ring

lemma toPoly_pmul : ∀ a b : List ℤ, toPoly (pmul a b) = toPoly a * toPoly b
  | [], b => by simp [pmul]
  | x :: a, b => by
      simp only [pmul, toPoly_padd, toPoly_map_mul, toPoly_cons, toPoly_pmul a b, C_0]; ring

lemma peval_eq : ∀ (p : List ℤ) (t : ℤ), peval p t = (toPoly p).eval t
  | [], t => by simp [peval]
  | x :: a, t => by
      have := peval_eq a t
      simp only [peval, List.foldr_cons] at this ⊢
      rw [this]; simp

lemma toPoly_plin (a : ℤ) : ∀ e : ℕ, toPoly (plin a e) = (X - C a) ^ e
  | 0 => by simp [plin]
  | e + 1 => by
      rw [plin, toPoly_pmul, toPoly_plin a e, pow_succ]
      simp only [toPoly_cons, toPoly_nil, C_neg, C_1]; ring

lemma toPoly_append_one : ∀ l : List ℤ, toPoly (l ++ [1]) = toPoly l + X ^ l.length
  | [] => by simp
  | x :: a => by
      simp only [List.cons_append, toPoly_cons, toPoly_append_one a, List.length_cons, pow_succ]; ring

lemma degree_toPoly_lt (l : List ℤ) : (toPoly l).degree < l.length := by
  rw [degree_lt_iff_coeff_zero]
  intro m hm
  rw [coeff_toPoly, coeffD, List.getD_eq_default _ _ hm]

/-- A congruence checked by `peqMod` is an equality of polynomials over `ℤ/m`. -/
lemma peqMod_sound {m : ℕ} {a b : List ℤ} (h : peqMod m a b = true) :
    (toPoly a).map (Int.castRingHom (ZMod m)) = (toPoly b).map (Int.castRingHom (ZMod m)) := by
  ext i
  simp only [coeff_map, coeff_toPoly, eq_intCast]
  rw [ZMod.intCast_eq_intCast_iff_dvd_sub]
  by_cases hi : i < max a.length b.length
  · unfold peqMod at h
    have := List.all_eq_true.mp h i (List.mem_range.mpr hi)
    have h0 : (coeffD a i - coeffD b i) % (m : ℤ) = 0 := by simpa using this
    have := Int.dvd_of_emod_eq_zero h0
    rw [← dvd_neg] at this; simpa using this
  · have ha : coeffD a i = 0 := List.getD_eq_default _ _ (by omega)
    have hb : coeffD b i = 0 := List.getD_eq_default _ _ (by omega)
    simp [ha, hb]

/-! ### Primality and goodness -/

lemma primeB_sound {q : ℕ} (h : primeB q = true) : q.Prime := by
  unfold primeB at h
  simp only [Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range, Bool.or_eq_true,
    bne_iff_ne, ne_eq] at h
  refine Nat.prime_def_lt'.mpr ⟨h.1, fun d hd hdq hdvd => ?_⟩
  rcases h.2 d hdq with h1 | h1
  · omega
  · exact h1 (Nat.mod_eq_zero_of_dvd hdvd)

/-- The K10′ goodness tests imply Mazur's definition (case a), read literally. -/
theorem goodB_sound {N ℓ q : ℕ} (hN : N.Prime) (hℓ : ℓ.Prime) (h5 : 5 ≤ ℓ) (hq : q.Prime) (hqN : q ≠ N)
    (hdiv : (N - 1) % ℓ = 0) (hr : (N - 1) / ℓ < 2 ^ rounds) (h : goodB N ℓ q = true) : MazurGood N ℓ q := by
  unfold goodB at h
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at h
  obtain ⟨hii, hi⟩ := h
  have := Fact.mk hN
  have := Fact.mk hℓ
  refine ⟨hq, hqN, by omega, ?_, ?_⟩
  · rintro ⟨x, hx⟩
    apply hi
    rw [GL2.L23.SmithPK.powMod_eq hN.pos _ _ _ hr]
    have hx0 : x ≠ 0 := by
      rintro rfl
      rw [zero_pow hℓ.ne_zero, eq_comm, ZMod.natCast_eq_zero_iff] at hx
      exact hqN ((Nat.prime_dvd_prime_iff_eq hN hq).mp hx).symm
    have hpow : ((q ^ ((N - 1) / ℓ) : ℕ) : ZMod N) = 1 := by
      push_cast
      rw [← hx, ← pow_mul, Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hdiv)]
      exact ZMod.pow_card_sub_one_eq_one hx0
    rw [← ZMod.val_natCast, hpow, ZMod.val_one]
  · have h2 : (2 : ZMod ℓ) ≠ 0 := by
      intro h0
      have : ((2 : ℕ) : ZMod ℓ) = 0 := by exact_mod_cast h0
      rw [ZMod.natCast_eq_zero_iff] at this
      have := Nat.le_of_dvd (by norm_num) this; omega
    refine mul_ne_zero ?_ (inv_ne_zero h2)
    intro h0
    apply hii
    have : (q : ZMod ℓ) = ((1 : ℕ) : ZMod ℓ) := by rw [Nat.cast_one]; exact sub_eq_zero.mp h0
    rw [ZMod.natCast_eq_natCast_iff'] at this
    rw [this, Nat.mod_eq_of_lt hℓ.one_lt]

/-! ### `num((N − 1)/12)` -/

lemma mazurN_eq (N : ℕ) (hN : 1 ≤ N) : mazurN N = nuNat N := by
  unfold mazurN nuNat
  have h1 : ((N : ℚ) - 1) = ((N - 1 : ℕ) : ℚ) := by rw [Nat.cast_sub hN]; simp
  rw [h1]
  generalize N - 1 = a
  obtain ⟨a', ha⟩ := Nat.gcd_dvd_left a 12
  obtain ⟨b', hb⟩ := Nat.gcd_dvd_right a 12
  have hg : 0 < Nat.gcd a 12 := Nat.gcd_pos_of_pos_right _ (by norm_num)
  have hda : a / Nat.gcd a 12 = a' := Nat.div_eq_of_eq_mul_right hg ha
  have hdb : 12 / Nat.gcd a 12 = b' := Nat.div_eq_of_eq_mul_right hg hb
  have hcop : a'.Coprime b' := by
    have := Nat.coprime_div_gcd_div_gcd hg; rwa [hda, hdb] at this
  rw [hda]
  generalize Nat.gcd a 12 = G at ha hb hg
  subst ha
  have hb0 : 0 < b' := by rcases Nat.eq_zero_or_pos b' with h | h <;> [simp [h] at hb; exact h]
  have hq : ((G * a' : ℕ) : ℚ) / 12 = ((a' : ℤ) : ℚ) / ((b' : ℤ) : ℚ) := by
    have h12 : (12 : ℚ) = (G : ℚ) * b' := by exact_mod_cast hb
    rw [div_eq_div_iff (by norm_num) (by exact_mod_cast hb0.ne'), h12]; push_cast; ring
  rw [hq, Rat.num_div_eq_of_coprime (by exact_mod_cast hb0) (by simpa using hcop)]
  simp

lemma factorization_of_mod {n ℓ v : ℕ} (hℓ : ℓ.Prime) (h1 : n % ℓ ^ v = 0) (h2 : n % ℓ ^ (v + 1) ≠ 0) :
    n.factorization ℓ = v := by
  have hn : n ≠ 0 := by rintro rfl; simp at h2
  have a := (hℓ.pow_dvd_iff_le_factorization hn).mp (Nat.dvd_of_mod_eq_zero h1)
  have b : ¬ v + 1 ≤ n.factorization ℓ := fun hb =>
    h2 (Nat.mod_eq_zero_of_dvd ((hℓ.pow_dvd_iff_le_factorization hn).mpr hb))
  omega

/-! ### Divisibility in `ℤ_[ℓ]` read off a residue -/

/-- `ℓ^j ∣ x` in `ℤ_[ℓ]` iff `toZModPow j x = 0`. -/
lemma padic_dvd_iff {ℓ : ℕ} [Fact ℓ.Prime] (j : ℕ) (x : ℤ_[ℓ]) :
    (ℓ : ℤ_[ℓ]) ^ j ∣ x ↔ PadicInt.toZModPow j x = 0 := by
  rw [← RingHom.mem_ker, PadicInt.ker_toZModPow, Ideal.mem_span_singleton]

/-- If `toZModPow K x` is the residue of the integer `z` and `j ≤ K`, then `ℓ^j ∣ x ↔ ℓ^j ∣ z`. -/
lemma padic_dvd_iff_int {ℓ : ℕ} [Fact ℓ.Prime] {K j : ℕ} (hj : j ≤ K) (x : ℤ_[ℓ]) (z : ℤ)
    (hx : PadicInt.toZModPow K x = (z : ZMod (ℓ ^ K))) : (ℓ : ℤ_[ℓ]) ^ j ∣ x ↔ ((ℓ : ℤ) ^ j ∣ z) := by
  rw [padic_dvd_iff, ← PadicInt.zmod_cast_comp_toZModPow j K hj, RingHom.comp_apply, hx, map_intCast,
    ZMod.intCast_zmod_eq_zero_iff_dvd]
  push_cast; rfl

/-! ### K10′ soundness -/

/-- The polynomial `F` of a certificate, over `ℤ`. -/
noncomputable def _root_.GL2.HeckeZ.Exec.K10.K10pCert.Fp (c : K10pCert) : ℤ[X] := toPoly c.F

/-- What an accepted K10′ certificate certifies. -/
structure K10pSpec (c : K10pCert) : Prop where
  guard : MazurGuard c.N c.ell
  ell_prime : c.ell.Prime
  good : MazurGood c.N c.ell c.q
  v_eq : (mazurN c.N).factorization c.ell = c.v
  v_lt : c.v < c.K
  monic : c.Fp.Monic
  natDegree : c.Fp.natDegree = c.e
  hensel : (toPoly c.h).map (Int.castRingHom (ZMod (c.ell ^ c.K))) =
    c.Fp.map (Int.castRingHom (ZMod (c.ell ^ c.K))) * (toPoly c.H).map (Int.castRingHom (ZMod (c.ell ^ c.K)))
  shape : c.Fp.map (Int.castRingHom (ZMod c.ell)) = (X - C ((1 + c.q : ℕ) : ZMod c.ell)) ^ c.e
  unit : ((toPoly c.H).map (Int.castRingHom (ZMod c.ell))).eval ((1 + c.q : ℕ) : ZMod c.ell) ≠ 0
  val : ∀ [Fact c.ell.Prime] (F' : ℤ_[c.ell][X]),
    F'.map (PadicInt.toZModPow c.K) = c.Fp.map (Int.castRingHom (ZMod (c.ell ^ c.K))) →
    (c.ell : ℤ_[c.ell]) ^ c.v ∣ F'.eval ((1 + c.q : ℕ) : ℤ_[c.ell]) ∧
      ¬ (c.ell : ℤ_[c.ell]) ^ (c.v + 1) ∣ F'.eval ((1 + c.q : ℕ) : ℤ_[c.ell])

lemma Fp_eq (c : K10pCert) : c.Fp = toPoly c.flo + X ^ c.e := toPoly_append_one c.flo

lemma Fp_monic (c : K10pCert) : c.Fp.Monic := by
  rw [Fp_eq, add_comm]; exact monic_X_pow_add (degree_toPoly_lt c.flo)

lemma Fp_natDegree (c : K10pCert) : c.Fp.natDegree = c.e := by
  rw [Fp_eq, add_comm, natDegree_add_eq_left_of_degree_lt] <;> simp only [natDegree_X_pow, degree_X_pow]
  exact degree_toPoly_lt c.flo

/-- **K10′ is sound.** -/
theorem k10p_sound (c : K10pCert) (hc : k10pCheck c = true) : K10pSpec c := by
  unfold k10pCheck at hc
  simp only [Bool.and_eq_true] at hc
  obtain ⟨⟨⟨hcell, hgood⟩, hhb⟩, hvb⟩ := hc
  unfold cellB at hcell
  simp only [Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq, beq_iff_eq] at hcell
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hN, hℓ⟩, hq⟩, hN5⟩, hℓ5⟩, hqN⟩, hdiv⟩, hr⟩, hv1⟩, hv2⟩, hv0⟩, hvK⟩ := hcell
  unfold henselB at hhb
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq] at hhb
  obtain ⟨⟨hshape, hhen⟩, hunit⟩ := hhb
  unfold valB at hvb
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq] at hvb
  obtain ⟨hval1, hval2⟩ := hvb
  have pN := primeB_sound hN
  have pℓ := primeB_sound hℓ
  have pq := primeB_sound hq
  have hnu : mazurN c.N = nuNat c.N := mazurN_eq c.N (by omega)
  have hfac : (mazurN c.N).factorization c.ell = c.v := by rw [hnu]; exact factorization_of_mod pℓ hv1 hv2
  have hdvd : c.ell ∣ mazurN c.N := by
    rw [hnu]
    exact Nat.dvd_trans (dvd_pow_self _ (by omega)) (Nat.dvd_of_mod_eq_zero hv1)
  have hpa : ((1 + c.q : ℕ) : ℤ) = c.a := by simp [K10pCert.a]
  have hpa' : ∀ {R : Type} [CommRing R], ((1 + c.q : ℕ) : R) = ((c.a : ℤ) : R) := by
    intro R _; rw [← hpa, Int.cast_natCast]
  refine ⟨⟨pN, hN5, hℓ5, hdvd⟩, pℓ, goodB_sound pN pℓ hℓ5 pq hqN hdiv hr hgood, hfac, by omega,
    Fp_monic c, Fp_natDegree c, ?_, ?_, ?_, ?_⟩
  · have := peqMod_sound hhen
    rw [toPoly_pmul, Polynomial.map_mul] at this; exact this
  · have := peqMod_sound hshape
    rw [toPoly_plin] at this
    rw [K10pCert.Fp, this]; simp [K10pCert.a]
  · rw [hpa', eval_intCast_map, eq_intCast, ← peval_eq, ne_eq, ZMod.intCast_zmod_eq_zero_iff_dvd]
    exact fun h => hunit (Int.emod_eq_zero_of_dvd h)
  · intro _ F' hF'
    have hx : PadicInt.toZModPow c.K (F'.eval ((1 + c.q : ℕ) : ℤ_[c.ell])) =
        ((peval c.F c.a : ℤ) : ZMod (c.ell ^ c.K)) := by
      rw [← eval₂_at_apply, ← eval_map, hF', map_natCast, hpa', eval_intCast_map, eq_intCast, peval_eq]
      rfl
    refine ⟨(padic_dvd_iff_int (by omega) _ _ hx).mpr (Int.dvd_of_emod_eq_zero hval1),
      fun h => hval2 (Int.emod_eq_zero_of_dvd ((padic_dvd_iff_int (by omega) _ _ hx).mp h))⟩

/-! ### The core: Cayley–Hamilton modulo the Eisenstein ideal, and no contamination -/

/-- **Cayley–Hamilton modulo an ideal.** `T` free of finite rank over `R`, `a − t ∈ I`: the characteristic polynomial `f` of
`t` (multiplication on `T`) satisfies `f(a) ↦ 0` in `T/I`. -/
theorem charpoly_eval_mem {R T : Type*} [CommRing R] [CommRing T] [Algebra R T] [Module.Free R T] [Module.Finite R T]
    (I : Ideal T) (t : T) (a : R) (h : algebraMap R T a - t ∈ I) :
    algebraMap R T ((LinearMap.mulLeft R t).charpoly.eval a) ∈ I := by
  set f := (LinearMap.mulLeft R t).charpoly
  have hCH : aeval t f = 0 := by
    have h0 := LinearMap.aeval_self_charpoly (LinearMap.mulLeft R t)
    have : LinearMap.mulLeft R t = Algebra.lmul R T t := rfl
    rw [this, aeval_algHom_apply] at h0
    have := congrArg (fun φ : Module.End R T => φ 1) h0
    simp at this
    exact this
  rw [← Ideal.Quotient.eq_zero_iff_mem]
  have hmk : Ideal.Quotient.mkₐ R I t = algebraMap R (T ⧸ I) a := by
    rw [Ideal.Quotient.mkₐ_eq_mk, IsScalarTower.algebraMap_apply R T (T ⧸ I), Ideal.Quotient.algebraMap_eq,
      eq_comm, Ideal.Quotient.eq]
    exact h
  have := congrArg (Ideal.Quotient.mkₐ R I) hCH
  rw [← aeval_algHom_apply, hmk, aeval_algebraMap_apply, map_zero] at this
  rw [← Ideal.Quotient.algebraMap_eq, ← IsScalarTower.algebraMap_apply]
  simpa [aeval_def, eval₂_eq_eval_map] using this

/-- A ring map `ℤ_[p] → ℤ/p^v` kills `x` only if `p^v ∣ x`. -/
lemma padic_dvd_of_map_zero {p : ℕ} [Fact p.Prime] (v : ℕ) (φ : ℤ_[p] →+* ZMod (p ^ v)) (x : ℤ_[p]) (h : φ x = 0) :
    (p : ℤ_[p]) ^ v ∣ x := by
  obtain ⟨y, hy⟩ := Ideal.mem_span_singleton.mp (PadicInt.appr_spec v x)
  have hx : x = (x.appr v : ℤ_[p]) + (p : ℤ_[p]) ^ v * y := by rw [← hy]; ring
  have hpv : φ ((p : ℤ_[p]) ^ v) = 0 := by
    rw [map_pow, map_natCast, ← Nat.cast_pow, ZMod.natCast_self]
  rw [hx, map_add, map_mul, hpv, zero_mul, add_zero, map_natCast, ZMod.natCast_eq_zero_iff] at h
  rw [hx]
  refine dvd_add ?_ (dvd_mul_right _ _)
  obtain ⟨k, hk⟩ := h
  exact ⟨k, by rw [hk]; push_cast; ring⟩

/-- **(note 5, first half)** With `hMazurEis`: for every prime `q ≠ N` (good or not), `ℓ^{v_ℓ(n)}` divides the value at `1 + q`
of the characteristic polynomial of `T_q` on Mazur's ring `T` (`η_q ∈ 𝔍`; `T/𝔍T = ℤ/ℓ^{v_ℓ(n)}`). -/
theorem mazur_dvd_charpoly {N ℓ : ℕ} [Fact ℓ.Prime] {T : Type*} [CommRing T] [Algebra ℤ_[ℓ] T]
    [Module.Free ℤ_[ℓ] T] [Module.Finite ℤ_[ℓ] T] {Tl : ℕ → T} {w : T}
    (hMazurEis : MazurEisLocal N ℓ T Tl w) (hN : N.Prime) (hN5 : 5 ≤ N) (hℓ5 : 5 ≤ ℓ) (hdvd : ℓ ∣ mazurN N)
    {q : ℕ} (hq : q.Prime) (hqN : q ≠ N) :
    (ℓ : ℤ_[ℓ]) ^ (mazurN N).factorization ℓ ∣ (LinearMap.mulLeft ℤ_[ℓ] (Tl q)).charpoly.eval ((1 + q : ℕ) : ℤ_[ℓ]) := by
  obtain ⟨e⟩ := hMazurEis.quot ⟨hN, hN5, hℓ5, hdvd⟩
  have hmem : algebraMap ℤ_[ℓ] T ((1 + q : ℕ) : ℤ_[ℓ]) - Tl q ∈ eisensteinIdeal N Tl w := by
    apply Ideal.subset_span
    exact Or.inr ⟨q, hq, hqN, by rw [map_natCast]; push_cast; ring⟩
  have h := charpoly_eval_mem _ _ _ hmem
  refine padic_dvd_of_map_zero _ (e.toRingHom.comp ((Ideal.Quotient.mk _).comp (algebraMap ℤ_[ℓ] T))) _ ?_
  show e (Ideal.Quotient.mk _ (algebraMap ℤ_[ℓ] T _)) = 0
  rw [Ideal.Quotient.eq_zero_iff_mem.mpr h, map_zero]

/-- **(note 5, second half) No contamination.** `F = f · G` over `ℤ_[ℓ]`, `G` monic with `G ≡ (x − a)^d (mod ℓ)`, `ℓ^v ∣ f(a)`
and `ℓ^{v+1} ∤ F(a)`: then `G = 1`. -/
theorem no_contamination {ℓ : ℕ} [Fact ℓ.Prime] (v d : ℕ) (a : ℕ) (f G : ℤ_[ℓ][X]) (hG : G.Monic)
    (hred : G.map (PadicInt.toZModPow 1) = (X - C (a : ZMod (ℓ ^ 1))) ^ d)
    (hf : (ℓ : ℤ_[ℓ]) ^ v ∣ f.eval (a : ℤ_[ℓ])) (hF : ¬ (ℓ : ℤ_[ℓ]) ^ (v + 1) ∣ (f * G).eval (a : ℤ_[ℓ])) :
    G = 1 := by
  have : Nontrivial (ZMod (ℓ ^ 1)) := ZMod.nontrivial_iff.mpr (by
    rw [pow_one]; exact (Fact.out : ℓ.Prime).one_lt.ne')
  have hdeg : G.natDegree = d := by
    rw [← hG.natDegree_map (PadicInt.toZModPow 1), hred, (monic_X_sub_C _).natDegree_pow, natDegree_X_sub_C, mul_one]
  rcases Nat.eq_zero_or_pos d with rfl | hd
  · exact Polynomial.eq_one_of_monic_natDegree_zero hG hdeg
  · exfalso
    apply hF
    have hGa : (ℓ : ℤ_[ℓ]) ∣ G.eval (a : ℤ_[ℓ]) := by
      rw [← pow_one (ℓ : ℤ_[ℓ]), padic_dvd_iff, ← eval₂_at_apply, ← eval_map, hred, map_natCast]
      simp [zero_pow hd.ne']
    rw [eval_mul, pow_succ]
    exact mul_dvd_mul hf hGa

/-- **The valuation check certifies the Eisenstein block (M-HZ4a, partly), proved mod `hMazurEis`.**
`c` an accepted K10′ certificate of the cell `(N, ℓ)`; `T` Mazur's ring there (`hMazurEis`); `hblock`: a lift `F'` of the
certificate's `F mod ℓ^K` is `f · G` with `f` the characteristic polynomial of `T_q` on `T` and `G` monic, `G ≡ (x − 1 − q)^d (mod ℓ)`
(the semilocal decomposition; the remaining M-HZ4a input). Then `G = 1`, `f = F'` has degree `e = rank T`, `ℓ^v ∥ f(1+q)`, and
`T = ℤ_ℓ[η_q]`, `𝔍T = (η_q)`. -/
theorem mazur_val_cert (c : K10pCert) (hc : k10pCheck c = true) [Fact c.ell.Prime] {T : Type*} [CommRing T]
    [Algebra ℤ_[c.ell] T] [Module.Free ℤ_[c.ell] T] [Module.Finite ℤ_[c.ell] T] {Tl : ℕ → T} {w : T}
    (hMazurEis : MazurEisLocal c.N c.ell T Tl w) (F' G : ℤ_[c.ell][X]) (d : ℕ)
    (hlift : F'.map (PadicInt.toZModPow c.K) = c.Fp.map (Int.castRingHom (ZMod (c.ell ^ c.K))))
    (hblock : F' = (LinearMap.mulLeft ℤ_[c.ell] (Tl c.q)).charpoly * G) (hGm : G.Monic)
    (hGred : G.map (PadicInt.toZModPow 1) = (X - C ((1 + c.q : ℕ) : ZMod (c.ell ^ 1))) ^ d) :
    G = 1 ∧ F' = (LinearMap.mulLeft ℤ_[c.ell] (Tl c.q)).charpoly ∧ Module.finrank ℤ_[c.ell] T = c.e ∧
      (c.ell : ℤ_[c.ell]) ^ c.v ∣ (LinearMap.mulLeft ℤ_[c.ell] (Tl c.q)).charpoly.eval ((1 + c.q : ℕ) : ℤ_[c.ell]) ∧
      ¬ (c.ell : ℤ_[c.ell]) ^ (c.v + 1) ∣
        (LinearMap.mulLeft ℤ_[c.ell] (Tl c.q)).charpoly.eval ((1 + c.q : ℕ) : ℤ_[c.ell]) ∧
      Algebra.adjoin ℤ_[c.ell] {mazurEta Tl c.q} = ⊤ ∧ eisensteinIdeal c.N Tl w = Ideal.span {mazurEta Tl c.q} := by
  have S := k10p_sound c hc
  have hf := mazur_dvd_charpoly hMazurEis S.guard.1 S.guard.2.1 S.guard.2.2.1 S.guard.2.2.2 S.good.1 S.good.2.1
  rw [S.v_eq] at hf
  have hval := S.val F' hlift
  have hG1 : G = 1 := no_contamination c.v d (1 + c.q) _ G hGm hGred hf (hblock ▸ hval.2)
  have hF : F' = (LinearMap.mulLeft ℤ_[c.ell] (Tl c.q)).charpoly := by rw [hblock, hG1, mul_one]
  have hdeg : Module.finrank ℤ_[c.ell] T = c.e := by
    have : Nontrivial (ZMod (c.ell ^ c.K)) := ZMod.nontrivial_iff.mpr
      (Nat.one_lt_pow (by have := S.v_lt; omega) S.ell_prime.one_lt).ne'
    have hFm : F'.Monic := hF ▸ LinearMap.charpoly_monic _
    rw [← LinearMap.charpoly_natDegree (LinearMap.mulLeft ℤ_[c.ell] (Tl c.q)), ← hF,
      ← hFm.natDegree_map (PadicInt.toZModPow c.K), hlift, S.monic.natDegree_map, S.natDegree]
  exact ⟨hG1, hF, hdeg, hF ▸ hval.1, hF ▸ hval.2, hMazurEis.gen S.guard _ S.good,
    hMazurEis.principal S.guard _ S.good⟩

end GL2.HeckeZ.K10
