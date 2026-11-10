/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Exec.Cert
import GL2.HeckeZ.Cert
import GL2.HeckeZ.RankES
import GL2.HeckeZ.Check.K3

/-!
# HeckeZ K1 / K2g: soundness and the cell theorem of 01 Thm. 4.1 (L-HZ1, M-HZ2)

`07-lean-plan.md` §5 (common shape), §5.1 (K1), §5.2 (K2g), §10 (M-HZ2). The executable checker
`GL2.HeckeZ.Exec.cellCheck` is Mathlib-free; here:
* `toPoly` reads a coefficient list as an element of `ℤ[X]`; `k2OK_sound`: a passing K2g item gives a monic `f` of degree
  `d`, `f · q = χ` and `u f + v f' = D ≠ 0` in `ℤ[X]`;
* `k1_modIndep`: a passing K1 certificate, with **TB-mat** (`TBmatK1`: the dumped matrices are the `X g` on `L/q` in some
  `ℤ`-basis of `L`, up to one invertible `U` modulo `q`), gives `d` monomials of `𝕋'` independent modulo every prime
  `ℓ ∣ q` (01 Prop. 2.2(1), the (C3) read);
* **`cell_sound`** — the cell theorem: with `rank 𝕋^fr ≤ d` (`RankLE`), **TB-χ** (`hχ`: the recorded `χ` are the
  characteristic polynomials of the `X p`) and TB-mat (`hmat`, `TBmatK1` for every K1 certificate, stated inline), `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ cof`, and `𝕋^fr = 𝕋'` when
  `cof = 1`; `cell_sound_ES` takes Eichler–Shimura (`hES`) in place of `RankLE` (`GL2.HeckeZ.RankES.rankLE_of_ES`).

**Trusted inputs (Q10(a), [LP] v1.9).** Exactly the two named hypotheses: `hχ` (tb_chi: ed.1 `char_Z`) and `hmat`
(tb_mat: `HECKE_DUMP_SAT_CERT` blocks); they are stated inline (heads `∀` / `∃`, not a project constant); and the literature hypothesis `hES` (`lit:eichler-shimura-levelone`). Nothing the
generator computes (`f`, `q`, `u`, `v`, `D`, words, pivots, inverses, the cofactor) is trusted.
-/

namespace GL2.HeckeZ.Check

open GL2.HeckeZ.Exec GL2.HeckeZ.Cert Polynomial
open scoped TensorProduct

/-! ### Polynomials -/

/-- A coefficient list read as an integer polynomial. -/
noncomputable def toPoly : List ℤ → ℤ[X]
  | [] => 0
  | a :: p => C a + X * toPoly p

lemma coeff_toPoly (p : List ℤ) (i : ℕ) : (toPoly p).coeff i = p.getD i 0 := by
  induction p generalizing i with
  | nil => simp [toPoly]
  | cons a p ih =>
    cases i with
    | zero => simp [toPoly]
    | succ i =>
      rw [toPoly, Polynomial.coeff_add, Polynomial.coeff_C, Polynomial.coeff_X_mul, ih]
      simp

lemma toPoly_padd (p q : List ℤ) : toPoly (padd p q) = toPoly p + toPoly q := by
  induction p generalizing q with
  | nil => simp [padd, toPoly]
  | cons a p ih =>
    cases q with
    | nil => simp [padd, toPoly]
    | cons b q => simp only [padd, toPoly, ih, map_add]; ring

lemma toPoly_pscale (c : ℤ) (p : List ℤ) : toPoly (pscale c p) = C c * toPoly p := by
  induction p with
  | nil => simp [pscale, toPoly]
  | cons a p ih =>
    have : pscale c (a :: p) = (c * a) :: pscale c p := rfl
    rw [this, toPoly, toPoly, ih, map_mul]; ring

lemma toPoly_pmul (p q : List ℤ) : toPoly (pmul p q) = toPoly p * toPoly q := by
  induction p with
  | nil => simp [pmul, toPoly]
  | cons a p ih => rw [pmul, toPoly_padd, toPoly_pscale, toPoly, ih, toPoly, map_zero]; ring

lemma toPoly_pderiv (p : List ℤ) : toPoly (pderiv p) = derivative (toPoly p) := by
  ext i
  rw [coeff_toPoly, coeff_derivative, coeff_toPoly]
  unfold pderiv
  by_cases hi : i < p.length - 1
  · rw [List.getD_eq_getElem _ _ (by simpa using hi)]
    simp only [List.getElem_map, List.getElem_range]
    push_cast; ring
  · rw [List.getD_eq_default _ _ (by simpa using hi), List.getD_eq_default _ _ (by omega)]
    simp

lemma pzero_toPoly {p : List ℤ} (h : pzero p = true) : toPoly p = 0 := by
  ext i
  rw [coeff_toPoly, coeff_zero]
  rcases lt_or_ge i p.length with hi | hi
  · rw [List.getD_eq_getElem _ _ hi]
    have := List.all_eq_true.mp h _ (List.getElem_mem hi)
    simpa using this
  · exact List.getD_eq_default _ _ hi

lemma toPoly_eq_of_peq {p q : List ℤ} (h : peq p q = true) : toPoly p = toPoly q := by
  have h0 := pzero_toPoly h
  rw [toPoly_padd, toPoly_pscale] at h0
  have : toPoly p - toPoly q = 0 := by rw [← h0]; simp; ring
  exact sub_eq_zero.mp this

lemma toPoly_single (a : ℤ) : toPoly [a] = C a := by simp [toPoly]

lemma monic_toPoly {f : List ℤ} {d : ℕ} (hl : f.length = d + 1) (hd : f.getD d 0 = 1) :
    (toPoly f).Monic ∧ (toPoly f).natDegree = d := by
  have hcd : (toPoly f).coeff d = 1 := by rw [coeff_toPoly, hd]
  have hdeg : (toPoly f).natDegree = d := by
    apply Polynomial.natDegree_eq_of_le_of_coeff_ne_zero
    · rw [Polynomial.natDegree_le_iff_coeff_eq_zero]
      intro N hN
      rw [coeff_toPoly]
      exact List.getD_eq_default _ _ (by rw [hl]; exact_mod_cast hN)
    · rw [hcd]; exact one_ne_zero
  exact ⟨by rw [Polynomial.Monic, Polynomial.leadingCoeff, hdeg, hcd], hdeg⟩

/-- What a passing K2g item gives in `ℤ[X]`. -/
structure K2Spec (d : ℕ) (it : K2Item) : Prop where
  monic : (toPoly it.f).Monic
  deg : (toPoly it.f).natDegree = d
  dvd : toPoly it.f * toPoly it.q = toPoly it.chi
  bez : toPoly it.u * toPoly it.f + toPoly it.v * derivative (toPoly it.f) = C it.D
  hD : it.D ≠ 0

/-- **Soundness of K2g.** -/
theorem k2OK_sound {d : ℕ} {it : K2Item} (h : k2OK d it = true) : K2Spec d it := by
  simp only [k2OK, Bool.and_eq_true, beq_iff_eq, bne_iff_ne, ne_eq] at h
  obtain ⟨⟨⟨⟨hl, hd⟩, hq⟩, hb⟩, hD⟩ := h
  obtain ⟨hm, hdeg⟩ := monic_toPoly hl hd
  refine ⟨hm, hdeg, ?_, ?_, hD⟩
  · rw [← toPoly_pmul]; exact toPoly_eq_of_peq hq
  · have := toPoly_eq_of_peq hb
    rw [toPoly_padd, toPoly_pmul, toPoly_pmul, toPoly_pderiv, toPoly_single] at this
    exact this

/-! ### List matrices modulo `m` -/

/-- A list matrix read in `M_f(ℤ/m)`. -/
def lmat (f m : ℕ) (M : List (List ℕ)) : Matrix (Fin f) (Fin f) (ZMod m) := Matrix.of fun i j => (ment M i j : ZMod m)

lemma list_sum_range_map {M : Type*} [AddCommMonoid M] (g : ℕ → M) (k : ℕ) :
    ((List.range k).map g).sum = ∑ i ∈ Finset.range k, g i := by
  induction k with
  | zero => simp
  | succ k ih => rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]; simp

lemma ment_grid {f : ℕ} (g : ℕ → ℕ → ℕ) {i j : ℕ} (hi : i < f) (hj : j < f) :
    ment ((List.range f).map fun i => (List.range f).map fun j => g i j) i j = g i j := by
  simp [ment, List.getD_eq_getElem?_getD, hi, hj]

lemma lmat_mul_apply (f m : ℕ) (A B : List (List ℕ)) (i j : Fin f) :
    (lmat f m A * lmat f m B) i j = ((mulEnt f A B i j : ℕ) : ZMod m) := by
  simp only [lmat, Matrix.mul_apply, Matrix.of_apply, mulEnt, list_sum_range_map, Nat.cast_sum, Nat.cast_mul]
  exact Fin.sum_univ_eq_sum_range (fun l => (ment A i l : ZMod m) * (ment B l j : ZMod m)) f

lemma lmat_mmul (f m : ℕ) (A B : List (List ℕ)) : lmat f m (mmul f m A B) = lmat f m A * lmat f m B := by
  ext i j
  rw [lmat_mul_apply]
  simp only [lmat, Matrix.of_apply, mmul]
  rw [ment_grid (fun i j => mulEnt f A B i j % m) i.2 j.2, ZMod.natCast_mod]

lemma lmat_mone (f m : ℕ) : lmat f m (mone f) = 1 := by
  ext i j
  simp only [lmat, Matrix.of_apply, mone]
  rw [ment_grid (fun i j => if i = j then 1 else 0) i.2 j.2, Matrix.one_apply]
  by_cases h : i = j
  · subst h; simp
  · have : (i : ℕ) ≠ j := fun h' => h (Fin.ext h')
    simp [h, this]

lemma lmat_wordMat (b1 q : ℕ) (mats : List (List (List ℕ))) (w : List ℕ) :
    lmat b1 q (wordMat b1 q mats w) = (w.map fun j => lmat b1 q (mats.getD j [])).prod := by
  induction w with
  | nil => simp [wordMat, lmat_mone]
  | cons j w ih =>
    have : wordMat b1 q mats (j :: w) = mmul b1 q (mats.getD j []) (wordMat b1 q mats w) := rfl
    rw [this, lmat_mmul, ih, List.map_cons, List.prod_cons]

/-! ### K1: the (C3) read -/

variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]

/-- **TB-mat** for one K1 certificate: there are a `ℤ`-basis `b` of `L` and `U` invertible modulo `q` with
`U · [X g]_b ≡ D_g · U (mod q)` for every dumped generator `g` (the dumped matrices are the operators on `L/q`). -/
def TBmatK1 (b1 : ℕ) (X : ℕ → Module.End ℤ L) (c : K1Cert) : Prop :=
  ∃ (b : Module.Basis (Fin b1) ℤ L) (U : Matrix (Fin b1) (Fin b1) (ZMod c.q)), IsUnit U ∧
    ∀ j < c.gens.length,
      U * (LinearMap.toMatrix b b (X (c.gens.getD j 0))).map (Int.cast) = lmat b1 c.q (c.mats.getD j []) * U

lemma map_cast_mul {n : Type*} [Fintype n] {R : Type*} [Ring R] (A B : Matrix n n ℤ) :
    (A * B).map (Int.cast : ℤ → R) = A.map Int.cast * B.map Int.cast :=
  Matrix.map_mul (f := Int.castRingHom R)

/-- The monomial `X_{g_{w₀}} X_{g_{w₁}} ⋯` of a word. -/
def wordOp (X : ℕ → Module.End ℤ L) (c : K1Cert) (w : List ℕ) : Module.End ℤ L :=
  (w.map fun j => X (c.gens.getD j 0)).prod

lemma conj_word {b1 : ℕ} (X : ℕ → Module.End ℤ L) (c : K1Cert) (b : Module.Basis (Fin b1) ℤ L)
    (U : Matrix (Fin b1) (Fin b1) (ZMod c.q))
    (hU : ∀ j < c.gens.length,
      U * (LinearMap.toMatrix b b (X (c.gens.getD j 0))).map (Int.cast) = lmat b1 c.q (c.mats.getD j []) * U)
    (w : List ℕ) (hw : ∀ j ∈ w, j < c.gens.length) :
    U * (LinearMap.toMatrix b b (wordOp X c w)).map (Int.cast) = lmat b1 c.q (wordMat b1 c.q c.mats w) * U := by
  induction w with
  | nil =>
    simp only [wordOp, List.map_nil, List.prod_nil, LinearMap.toMatrix_one, wordMat, List.foldr_nil, lmat_mone]
    rw [Matrix.map_one _ (Int.cast_zero) (Int.cast_one), mul_one, one_mul]
  | cons j w ih =>
    have hj := hw j (List.mem_cons_self ..)
    have hw' : ∀ j ∈ w, j < c.gens.length := fun j' h' => hw j' (List.mem_cons_of_mem _ h')
    have e1 : wordOp X c (j :: w) = X (c.gens.getD j 0) * wordOp X c w := by
      simp [wordOp]
    have e2 : wordMat b1 c.q c.mats (j :: w) = mmul b1 c.q (c.mats.getD j []) (wordMat b1 c.q c.mats w) := rfl
    rw [e1, e2, lmat_mmul, LinearMap.toMatrix_mul, map_cast_mul, ← mul_assoc, hU j hj, mul_assoc, ih hw',
      mul_assoc]

/-- **K1 ⇒ independence modulo `ℓ`** (01 Prop. 2.2(1) through (C3)): under TB-mat, the `d` certified monomials are
linearly independent modulo every prime `ℓ ∣ q`. -/
theorem k1_modIndep {b1 d : ℕ} (c : K1Cert) (hc : k1OK b1 d c = true) (X : ℕ → Module.End ℤ L)
    (hmat : TBmatK1 b1 X c) {ℓ : ℕ} (hℓ : ℓ.Prime) (hℓq : ℓ ∣ c.q) :
    ModIndep (ℓ : ℤ) (fun k : Fin d => wordOp X c (c.words.getD k [])) := by
  haveI := Fact.mk hℓ
  obtain ⟨b, U, hU, hconj⟩ := hmat
  simp only [k1OK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
  obtain ⟨⟨⟨⟨⟨⟨hq2, _⟩, hwl⟩, hpl⟩, hwords⟩, hpos⟩, hinv⟩ := hc
  have hwk : ∀ k : Fin d, ∀ j ∈ c.words.getD k [], j < c.gens.length := by
    intro k j hj
    have hk : (k : ℕ) < c.words.length := by rw [hwl]; exact k.2
    rw [List.getD_eq_getElem _ _ hk] at hj
    have := List.all_eq_true.mp hwords _ (List.getElem_mem hk)
    simpa using List.all_eq_true.mp this j hj
  rintro a ⟨y, hy⟩ k0
  let φ : ZMod c.q →+* ZMod ℓ := ZMod.castHom hℓq (ZMod ℓ)
  set W : Fin d → Matrix (Fin b1) (Fin b1) (ZMod ℓ) :=
    fun k => (lmat b1 c.q (wordMat b1 c.q c.mats (c.words.getD k []))).map φ with hW
  set M : Fin d → Matrix (Fin b1) (Fin b1) (ZMod ℓ) :=
    fun k => (LinearMap.toMatrix b b (wordOp X c (c.words.getD k []))).map (Int.cast) with hM
  set U' := U.map φ with hU'
  have hU'u : IsUnit U' := hU.map φ.mapMatrix
  have hconjℓ : ∀ k, U' * M k = W k * U' := by
    intro k
    have h := congrArg φ.mapMatrix (conj_word X c b U hconj _ (hwk k))
    rw [map_mul, map_mul, RingHom.mapMatrix_apply, RingHom.mapMatrix_apply, RingHom.mapMatrix_apply,
      Matrix.map_map] at h
    rw [show (⇑φ ∘ Int.cast : ℤ → ZMod ℓ) = Int.cast from funext fun x => map_intCast φ x] at h
    exact h
  -- `Σ ā_k M_k = 0`
  have hsumM : ∑ k, (a k : ZMod ℓ) • M k = 0 := by
    ext i j
    have h1 := congrArg (fun F => LinearMap.toMatrix b b F i j) hy
    simp only [map_sum, map_zsmul, Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul] at h1
    have h2 := congrArg (Int.cast : ℤ → ZMod ℓ) h1
    push_cast at h2
    simp only [hM, Matrix.sum_apply, Matrix.smul_apply, Matrix.map_apply, smul_eq_mul, Matrix.zero_apply]
    rw [h2]; simp
  -- `Σ ā_k W_k = 0`
  have hsumW : ∑ k, (a k : ZMod ℓ) • W k = 0 := by
    have : (∑ k, (a k : ZMod ℓ) • W k) * U' = 0 := by
      rw [Finset.sum_mul]
      simp_rw [Matrix.smul_mul, ← hconjℓ, ← Matrix.mul_smul, ← Finset.mul_sum, hsumM, mul_zero]
    exact (hU'u.mul_left_eq_zero).mp this
  -- the pivot matrix `C` and its inverse
  set C : Matrix (Fin d) (Fin d) (ZMod ℓ) := (lmat d c.q (pivMat b1 c)).map φ with hC
  set Ci : Matrix (Fin d) (Fin d) (ZMod ℓ) := (lmat d c.q c.cinv).map φ with hCi
  have hCCi : C * Ci = 1 := by
    have h1 : lmat d c.q (pivMat b1 c) * lmat d c.q c.cinv = 1 := by
      ext i j
      have := (GL2.HeckeZ.K3.allFin_iff d _).mp hinv i i.2 j j.2
      rw [lmat_mul_apply, Matrix.one_apply]
      have h' : ((mulEnt d (pivMat b1 c) c.cinv i j : ℕ) : ZMod c.q) =
          ((if (i : ℕ) = j then 1 else 0 : ℕ) : ZMod c.q) := by
        rw [ZMod.natCast_eq_natCast_iff']; simpa using this
      rw [h']
      by_cases hij : i = j
      · subst hij; simp
      · have : (i : ℕ) ≠ j := fun h => hij (Fin.ext h)
        simp [hij, this]
    have := congrArg φ.mapMatrix h1
    rw [map_mul, map_one, RingHom.mapMatrix_apply, RingHom.mapMatrix_apply] at this
    exact this
  have hCent : ∀ k m : Fin d, ment (pivMat b1 c) k m =
      ment (wordMat b1 c.q c.mats (c.words.getD k [])) (c.pos.getD m (0, 0)).1 (c.pos.getD m (0, 0)).2 := by
    intro k m
    have hk : (k : ℕ) < c.words.length := by rw [hwl]; exact k.2
    have hm : (m : ℕ) < c.pos.length := by rw [hpl]; exact m.2
    simp [pivMat, ment, List.getD_eq_getElem?_getD, hk, hm]
  have hvec : Matrix.vecMul (fun k => (a k : ZMod ℓ)) C = 0 := by
    funext m
    have hm : (m : ℕ) < c.pos.length := by rw [hpl]; exact m.2
    obtain ⟨hr, hs⟩ : (c.pos.getD m (0, 0)).1 < b1 ∧ (c.pos.getD m (0, 0)).2 < b1 := by
      rw [List.getD_eq_getElem _ _ hm]
      have := List.all_eq_true.mp hpos _ (List.getElem_mem hm)
      simpa using this
    have := congrFun (congrFun hsumW ⟨_, hr⟩) ⟨_, hs⟩
    simp only [Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul, Matrix.zero_apply, hW, Matrix.map_apply, lmat,
      Matrix.of_apply] at this
    simp only [Matrix.vecMul, dotProduct, hC, Matrix.map_apply, lmat, Matrix.of_apply, hCent, Pi.zero_apply]
    exact this
  have ha0 : (fun k => (a k : ZMod ℓ)) = 0 := by
    have := congrArg (fun v => Matrix.vecMul v Ci) hvec
    simp only [Matrix.vecMul_vecMul, hCCi, Matrix.vecMul_one, Matrix.zero_vecMul] at this
    exact this
  have := congrFun ha0 k0
  simpa [ZMod.intCast_zmod_eq_zero_iff_dvd] using this

/-! ### K1m: dumped blocks plus an étale complement -/

lemma lmat_zero_grid (f m : ℕ) :
    lmat f m ((List.range f).map fun _ => (List.range f).map fun _ => 0) = 0 := by
  ext i j
  simp only [lmat, Matrix.of_apply, Matrix.zero_apply]
  rw [ment_grid (fun _ _ => 0) i.2 j.2]; simp

lemma lmat_madd (f m : ℕ) (A B : List (List ℕ)) :
    lmat f m (pevalMat.madd f m A B) = lmat f m A + lmat f m B := by
  ext i j
  simp only [lmat, Matrix.of_apply, Matrix.add_apply, pevalMat.madd]
  rw [ment_grid (fun i j => (ment A i j + ment B i j) % m) i.2 j.2, ZMod.natCast_mod, Nat.cast_add]

lemma lmat_mscal {f m : ℕ} (hm : 0 < m) (a : ℤ) (A : List (List ℕ)) :
    lmat f m (pevalMat.mscal f m a A) = (a : ZMod m) • lmat f m A := by
  ext i j
  simp only [lmat, Matrix.of_apply, Matrix.smul_apply, pevalMat.mscal, smul_eq_mul]
  rw [ment_grid (fun i j => ((a % (m : ℤ)).toNat * ment A i j) % m) i.2 j.2, ZMod.natCast_mod, Nat.cast_mul]
  congr 1
  have h0 : 0 ≤ a % (m : ℤ) := Int.emod_nonneg _ (by exact_mod_cast hm.ne')
  have : ((a % (m : ℤ)).toNat : ZMod m) = ((a % (m : ℤ) : ℤ) : ZMod m) := by
    rw [← Int.cast_natCast, Int.toNat_of_nonneg h0]
  rw [this, ZMod.intCast_mod]

lemma lmat_pevalMat {f m : ℕ} (hm : 0 < m) (M : List (List ℕ)) (p : List ℤ) :
    lmat f m (pevalMat f m M p) = Polynomial.aeval (lmat f m M) (toPoly p) := by
  induction p with
  | nil => rw [pevalMat, lmat_zero_grid, toPoly, map_zero]
  | cons a p ih =>
    rw [pevalMat, lmat_madd, lmat_mscal hm, lmat_mmul, ih, lmat_mone, toPoly, map_add, map_mul, aeval_C, aeval_X,
      Algebra.algebraMap_eq_smul_one, Int.cast_smul_eq_zsmul]

lemma lmat_mpow (f m : ℕ) (A : List (List ℕ)) (e : ℕ) : lmat f m (mpow f m A e) = lmat f m A ^ e := by
  induction e with
  | zero => simp [mpow, lmat_mone]
  | succ e ih => rw [mpow, lmat_mmul, ih, pow_succ']

/-- Coefficientwise congruence modulo `q` gives equality after `ℤ → ℤ/ℓ`, `ℓ ∣ q`. -/
lemma map_eq_of_peqMod {q ℓ : ℕ} (hℓq : ℓ ∣ q) {p r : List ℤ} (h : peqMod q p r = true) :
    (toPoly p).map (Int.castRingHom (ZMod ℓ)) = (toPoly r).map (Int.castRingHom (ZMod ℓ)) := by
  have key : (toPoly (padd p (pscale (-1) r))).map (Int.castRingHom (ZMod ℓ)) = 0 := by
    ext i
    rw [Polynomial.coeff_map, coeff_toPoly, Polynomial.coeff_zero]
    rcases lt_or_ge i (padd p (pscale (-1) r)).length with hi | hi
    · rw [List.getD_eq_getElem _ _ hi]
      have := List.all_eq_true.mp h _ (List.getElem_mem hi)
      simp only [beq_iff_eq] at this
      have hd : (q : ℤ) ∣ (padd p (pscale (-1) r))[i] := Int.dvd_of_emod_eq_zero this
      rw [eq_intCast, ZMod.intCast_zmod_eq_zero_iff_dvd]
      exact (Int.natCast_dvd_natCast.mpr hℓq).trans hd
    · rw [List.getD_eq_default _ _ hi]; simp
  rw [toPoly_padd, toPoly_pscale, Polynomial.map_add, Polynomial.map_mul, Polynomial.map_C] at key
  have : (toPoly p).map (Int.castRingHom (ZMod ℓ)) - (toPoly r).map (Int.castRingHom (ZMod ℓ)) = 0 := by
    rw [← key]; simp; ring
  exact sub_eq_zero.mp this

lemma fromBlocks_pow {m n R : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [CommRing R]
    (A : Matrix m m R) (D : Matrix n n R) (e : ℕ) :
    (Matrix.fromBlocks A 0 0 D) ^ e = Matrix.fromBlocks (A ^ e) 0 0 (D ^ e) := by
  induction e with
  | zero => simp [Matrix.fromBlocks_one]
  | succ e ih => rw [pow_succ, ih, Matrix.fromBlocks_multiply, pow_succ, pow_succ]; simp

lemma fromBlocks_sum {m n R : Type*} [CommRing R] {ι : Type*} (t : Finset ι) (A : ι → Matrix m m R)
    (D : ι → Matrix n n R) :
    ∑ i ∈ t, Matrix.fromBlocks (A i) 0 0 (D i) = Matrix.fromBlocks (∑ i ∈ t, A i) 0 0 (∑ i ∈ t, D i) := by
  classical
  induction t using Finset.induction_on with
  | empty => simp [Matrix.fromBlocks_zero]
  | insert x t hx ih =>
    rw [Finset.sum_insert hx, ih, Matrix.fromBlocks_add, Finset.sum_insert hx, Finset.sum_insert hx]
    simp only [add_zero]

lemma fromBlocks_aeval {m n R : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n] [CommRing R]
    (A : Matrix m m R) (D : Matrix n n R) (h : ℤ[X]) :
    Polynomial.aeval (Matrix.fromBlocks A 0 0 D) h =
      Matrix.fromBlocks (Polynomial.aeval A h) 0 0 (Polynomial.aeval D h) := by
  rw [Polynomial.aeval_eq_sum_range, Polynomial.aeval_eq_sum_range, Polynomial.aeval_eq_sum_range, ← fromBlocks_sum]
  refine Finset.sum_congr rfl fun e _ => ?_
  rw [fromBlocks_pow, Matrix.fromBlocks_smul]
  simp only [smul_zero]

lemma conj_aeval {n R : Type*} [Fintype n] [DecidableEq n] [CommRing R] {U A B : Matrix n n R} (h : U * A = B * U)
    (g : ℤ[X]) : U * Polynomial.aeval A g = Polynomial.aeval B g * U := by
  have hp : ∀ e : ℕ, U * A ^ e = B ^ e * U := by
    intro e
    induction e with
    | zero => simp
    | succ e ih => rw [pow_succ, ← mul_assoc, ih, mul_assoc, h, ← mul_assoc, ← pow_succ]
  rw [Polynomial.aeval_eq_sum_range, Polynomial.aeval_eq_sum_range, Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun e _ => ?_
  rw [Matrix.mul_smul, hp, Matrix.smul_mul]

/-- The monomial `X_{g_{w₀}} X_{g_{w₁}} ⋯` for a generator list. -/
def wordOpG (X : ℕ → Module.End ℤ L) (gens : List ℕ) (w : List ℕ) : Module.End ℤ L :=
  (w.map fun j => X (gens.getD j 0)).prod

/-- **TB-mat** for a K1m certificate: a `ℤ`-basis of `L` indexed by `B ⊔ B'` and `U` invertible modulo `q` such that each
dumped generator is block-diagonal, `U [X g]_b ≡ diag(D_g, Y_g) U (mod q)`, with `D_g` the dumped matrix on the stable
blocks `B` and some (unknown) `Y_g` on the rest. -/
def TBmatK1m (b1 : ℕ) (X : ℕ → Module.End ℤ L) (k : K1mCert) : Prop :=
  ∃ (b : Module.Basis (Fin k.fB ⊕ Fin (b1 - k.fB)) ℤ L)
    (U : Matrix (Fin k.fB ⊕ Fin (b1 - k.fB)) (Fin k.fB ⊕ Fin (b1 - k.fB)) (ZMod k.q)), IsUnit U ∧
    ∀ j < k.gens.length, ∃ Y : Matrix (Fin (b1 - k.fB)) (Fin (b1 - k.fB)) (ZMod k.q),
      U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) =
        Matrix.fromBlocks (lmat k.fB k.q (k.mats.getD j [])) 0 0 Y * U

/-- **K1m ⇒ independence modulo `ℓ`** (01 Prop. 2.2(1) blockwise + Prop. 2.3 on the étale part): under TB-mat for the
dumped blocks and TB-χ for `θ`, the `r_B` monomials and the `s` elements `θ^i f₁(θ)^m` are independent modulo every prime
`ℓ ∣ q`. -/
theorem k1m_modIndep {b1 d : ℕ} (k : K1mCert) (hc : k1mOK d k = true) (X : ℕ → Module.End ℤ L)
    (hχ : (X (k.gens.getD k.th 0)).charpoly = toPoly k.chi) (hmat : TBmatK1m b1 X k) {ℓ : ℕ} (hℓ : ℓ.Prime)
    (hℓq : ℓ ∣ k.q) :
    ∃ t : Fin d → Module.End ℤ L, (∀ i, t i ∈ Algebra.adjoin ℤ (X '' {g | g ∈ k.gens})) ∧ ModIndep (ℓ : ℤ) t := by
  classical
  haveI := Fact.mk hℓ
  obtain ⟨b, U, hU, hconj⟩ := hmat
  simp only [k1mOK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hq2, _⟩, hth⟩, hdeq⟩, hpl⟩, hwords⟩, hpos⟩, hinv⟩, hnil⟩, hf2l⟩, hf2lead⟩, hf2c2⟩, hbez2⟩,
    hcop⟩ := hc
  subst hdeq
  set rB := k.words.length with hrB
  set s := k.f2.length - 1 with hs
  set Tg : Subalgebra ℤ (Module.End ℤ L) := Algebra.adjoin ℤ (X '' {g | g ∈ k.gens}) with hTg
  have hgen : ∀ j < k.gens.length, X (k.gens.getD j 0) ∈ Tg := fun j hj =>
    Algebra.subset_adjoin ⟨_, by rw [List.getD_eq_getElem _ _ hj]; exact List.getElem_mem hj, rfl⟩
  set θ := X (k.gens.getD k.th 0) with hθdef
  have hθT : θ ∈ Tg := hgen _ hth
  have hwk : ∀ i : Fin rB, ∀ j ∈ k.words.getD i [], j < k.gens.length := by
    intro i j hj
    rw [List.getD_eq_getElem _ _ i.2] at hj
    have := List.all_eq_true.mp hwords _ (List.getElem_mem i.2)
    simpa using List.all_eq_true.mp this j hj
  let w : Fin rB → Module.End ℤ L := fun i => wordOpG X k.gens (k.words.getD i [])
  let poly : Fin s → ℤ[X] := fun i => Polynomial.X ^ (i : ℕ) * toPoly k.f1 ^ k.m
  let z : Fin s → Module.End ℤ L := fun i => Polynomial.aeval θ (poly i)
  have hwT : ∀ i, w i ∈ Tg := by
    intro i
    refine Subalgebra.list_prod_mem _ fun x hx => ?_
    obtain ⟨j, hj, rfl⟩ := List.mem_map.mp hx
    exact hgen j (hwk i j hj)
  have hzT : ∀ i, z i ∈ Tg := by
    intro i
    show Polynomial.aeval θ (poly i) ∈ Tg
    rw [← Polynomial.aeval_subalgebra_coe (poly i) Tg ⟨θ, hθT⟩]; exact SetLike.coe_mem _
  refine ⟨Fin.append w z, fun i => ?_, ?_⟩
  · refine Fin.addCases (fun i => ?_) (fun i => ?_) i
    · simpa using hwT i
    · simpa using hzT i
  rintro a ⟨y, hy⟩ k0
  rw [Fin.sum_univ_add] at hy
  simp only [Fin.append_left, Fin.append_right] at hy
  -- matrices modulo `ℓ`
  let φ : ZMod k.q →+* ZMod ℓ := ZMod.castHom hℓq (ZMod ℓ)
  let Mℓ : Module.End ℤ L →ₐ[ℤ] Matrix (Fin k.fB ⊕ Fin (b1 - k.fB)) (Fin k.fB ⊕ Fin (b1 - k.fB)) (ZMod ℓ) :=
    ((Int.castRingHom (ZMod ℓ)).mapMatrix.toIntAlgHom).comp (LinearMap.toMatrixAlgEquiv b).toAlgHom
  have hMℓ : ∀ F, Mℓ F = (LinearMap.toMatrix b b F).map (Int.cast) := fun F => rfl
  set U' := U.map φ with hU'
  have hU'u : IsUnit U' := hU.map φ.mapMatrix
  -- block form of a generator, of a word, of a polynomial in `θ`, modulo `ℓ`
  have hgenℓ : ∀ j < k.gens.length, ∃ Y : Matrix (Fin (b1 - k.fB)) (Fin (b1 - k.fB)) (ZMod ℓ),
      U' * Mℓ (X (k.gens.getD j 0)) = Matrix.fromBlocks ((lmat k.fB k.q (k.mats.getD j [])).map φ) 0 0 Y * U' := by
    intro j hj
    obtain ⟨Y, hY⟩ := hconj j hj
    refine ⟨Y.map φ, ?_⟩
    have h := congrArg φ.mapMatrix hY
    rw [map_mul, map_mul, RingHom.mapMatrix_apply, RingHom.mapMatrix_apply, RingHom.mapMatrix_apply, Matrix.map_map,
      show (⇑φ ∘ Int.cast : ℤ → ZMod ℓ) = Int.cast from funext fun x => map_intCast φ x,
      Matrix.fromBlocks_map, Matrix.map_zero _ (map_zero φ), Matrix.map_zero _ (map_zero φ)] at h
    rw [hMℓ]; exact h
  have hwordℓ : ∀ i : Fin rB, ∃ Y : Matrix (Fin (b1 - k.fB)) (Fin (b1 - k.fB)) (ZMod ℓ),
      U' * Mℓ (w i) = Matrix.fromBlocks ((lmat k.fB k.q (wordMat k.fB k.q k.mats (k.words.getD i []))).map φ) 0 0 Y * U' := by
    intro i
    have : ∀ ws : List ℕ, (∀ j ∈ ws, j < k.gens.length) → ∃ Y : Matrix (Fin (b1 - k.fB)) (Fin (b1 - k.fB)) (ZMod ℓ),
        U' * Mℓ (wordOpG X k.gens ws) =
          Matrix.fromBlocks ((lmat k.fB k.q (wordMat k.fB k.q k.mats ws)).map φ) 0 0 Y * U' := by
      intro ws hws
      induction ws with
      | nil =>
        refine ⟨1, ?_⟩
        simp only [wordOpG, List.map_nil, List.prod_nil, map_one, wordMat, List.foldr_nil, lmat_mone]
        rw [Matrix.map_one _ (map_zero φ) (map_one φ), Matrix.fromBlocks_one, mul_one, one_mul]
      | cons j ws ih =>
        obtain ⟨Y1, h1⟩ := hgenℓ j (hws j (List.mem_cons_self ..))
        obtain ⟨Y2, h2⟩ := ih (fun j' h' => hws j' (List.mem_cons_of_mem _ h'))
        refine ⟨Y1 * Y2, ?_⟩
        have e1 : wordOpG X k.gens (j :: ws) = X (k.gens.getD j 0) * wordOpG X k.gens ws := by simp [wordOpG]
        have e2 : wordMat k.fB k.q k.mats (j :: ws) = mmul k.fB k.q (k.mats.getD j []) (wordMat k.fB k.q k.mats ws) :=
          rfl
        rw [e1, e2, lmat_mmul, Matrix.map_mul, map_mul, ← mul_assoc, h1, mul_assoc, h2, ← mul_assoc,
          Matrix.fromBlocks_multiply]
        simp
    exact this _ (hwk i)
  obtain ⟨Yθ, hYθ⟩ := hgenℓ k.th hth
  have hpolyℓ : ∀ g : ℤ[X], U' * Mℓ (Polynomial.aeval θ g) =
      Matrix.fromBlocks (Polynomial.aeval ((lmat k.fB k.q (k.mats.getD k.th [])).map φ) g) 0 0
        (Polynomial.aeval Yθ g) * U' := by
    intro g
    rw [← Polynomial.aeval_algHom_apply, conj_aeval hYθ, fromBlocks_aeval]
  -- `f₁(D_θ)^m ≡ 0` on `B`
  have hnil' : (Polynomial.aeval ((lmat k.fB k.q (k.mats.getD k.th [])).map φ) (toPoly k.f1)) ^ k.m = 0 := by
    have h0 : lmat k.fB k.q (mpow k.fB k.q (pevalMat k.fB k.q (k.mats.getD k.th []) k.f1) k.m) = 0 := by
      ext i j
      have := (GL2.HeckeZ.K3.allFin_iff _ _).mp hnil i i.2 j j.2
      simp only [beq_iff_eq] at this
      simp only [lmat, Matrix.of_apply, Matrix.zero_apply]
      rw [this]; simp
    rw [lmat_mpow, lmat_pevalMat (by omega)] at h0
    have h1 := congrArg (fun M => M.map φ) h0
    simp only [Matrix.map_pow, Matrix.map_zero _ (map_zero φ)] at h1
    rw [← h1]
    congr 1
    exact Polynomial.aeval_algHom_apply (AlgHom.mapMatrix (m := Fin k.fB) φ.toIntAlgHom) _ _
  have hzB : ∀ i : Fin s, Polynomial.aeval ((lmat k.fB k.q (k.mats.getD k.th [])).map φ) (poly i) = 0 := by
    intro i
    simp only [poly, map_mul, map_pow, Polynomial.aeval_X, hnil', mul_zero]
  -- the relation modulo `ℓ`
  have hrel : ∑ i : Fin rB, (a (Fin.castAdd s i) : ZMod ℓ) • Mℓ (w i) +
      ∑ i : Fin s, (a (Fin.natAdd rB i) : ZMod ℓ) • Mℓ (z i) = 0 := by
    have h1 := congrArg Mℓ hy
    simp only [map_add, map_sum, map_zsmul] at h1
    simp only [← Int.cast_smul_eq_zsmul (ZMod ℓ)] at h1
    rw [h1, Int.cast_natCast, ZMod.natCast_self, zero_smul]
  -- project to the block `B`
  set V := ((hU'u.unit⁻¹ : (Matrix (Fin k.fB ⊕ Fin (b1 - k.fB)) (Fin k.fB ⊕ Fin (b1 - k.fB)) (ZMod ℓ))ˣ) :
    Matrix (Fin k.fB ⊕ Fin (b1 - k.fB)) (Fin k.fB ⊕ Fin (b1 - k.fB)) (ZMod ℓ)) with hV
  have hUV : U' * V = 1 := hU'u.mul_val_inv
  have hπ : ∀ M B', U' * M = B' * U' → U' * M * V = B' := fun M B' h => by rw [h, mul_assoc, hUV, mul_one]
  choose Yw hYw using hwordℓ
  set Wm : Fin rB → Matrix (Fin k.fB) (Fin k.fB) (ZMod ℓ) :=
    fun i => (lmat k.fB k.q (wordMat k.fB k.q k.mats (k.words.getD i []))).map φ with hWm
  have hw' : ∀ r, U' * Mℓ (w r) * V = Matrix.fromBlocks (Wm r) 0 0 (Yw r) := fun r => hπ _ _ (hYw r)
  have hz' : ∀ r, U' * Mℓ (z r) * V = Matrix.fromBlocks 0 0 0 (Polynomial.aeval Yθ (poly r)) := by
    intro r; rw [hπ _ _ (hpolyℓ (poly r)), hzB]
  have hent : ∀ i j : Fin k.fB, ∑ r : Fin rB, (a (Fin.castAdd s r) : ZMod ℓ) * Wm r i j = 0 := by
    intro i j
    have h := congrArg (fun M => (U' * M * V) (Sum.inl i) (Sum.inl j)) hrel
    simp only [mul_add, add_mul, Finset.mul_sum, Finset.sum_mul, Matrix.mul_smul, Matrix.smul_mul, mul_zero,
      zero_mul, hw', hz'] at h
    simpa [Matrix.add_apply, Matrix.sum_apply, Matrix.smul_apply, Matrix.fromBlocks_apply₁₁] using h
  -- the pivot certificate on `B`: `ℓ ∣ a_r` for the monomials
  let c1 : K1Cert := k.toK1
  set C : Matrix (Fin rB) (Fin rB) (ZMod ℓ) := (lmat rB k.q (pivMat k.fB c1)).map φ with hC
  set Ci : Matrix (Fin rB) (Fin rB) (ZMod ℓ) := (lmat rB k.q k.cinv).map φ with hCi
  have hCCi : C * Ci = 1 := by
    have h1 : lmat rB k.q (pivMat k.fB c1) * lmat rB k.q k.cinv = 1 := by
      ext i j
      have := (GL2.HeckeZ.K3.allFin_iff rB _).mp hinv i i.2 j j.2
      rw [lmat_mul_apply, Matrix.one_apply]
      have h' : ((mulEnt rB (pivMat k.fB c1) k.cinv i j : ℕ) : ZMod k.q) =
          ((if (i : ℕ) = j then 1 else 0 : ℕ) : ZMod k.q) := by
        rw [ZMod.natCast_eq_natCast_iff']; simpa using this
      rw [h']
      by_cases hij : i = j
      · subst hij; simp
      · have : (i : ℕ) ≠ j := fun h => hij (Fin.ext h)
        simp [hij, this]
    have := congrArg φ.mapMatrix h1
    rw [map_mul, map_one, RingHom.mapMatrix_apply, RingHom.mapMatrix_apply] at this
    exact this
  have hCent : ∀ r m : Fin rB, ment (pivMat k.fB c1) r m =
      ment (wordMat k.fB k.q k.mats (k.words.getD r [])) (k.pos.getD m (0, 0)).1 (k.pos.getD m (0, 0)).2 := by
    intro r m
    have hr : (r : ℕ) < k.words.length := r.2
    have hm : (m : ℕ) < k.pos.length := by rw [hpl]; exact m.2
    simp [pivMat, ment, c1, K1mCert.toK1, List.getD_eq_getElem?_getD, hr, hm]
  have hvec : Matrix.vecMul (fun r => (a (Fin.castAdd s r) : ZMod ℓ)) C = 0 := by
    funext m
    have hm : (m : ℕ) < k.pos.length := by rw [hpl]; exact m.2
    obtain ⟨hr, hs'⟩ : (k.pos.getD m (0, 0)).1 < k.fB ∧ (k.pos.getD m (0, 0)).2 < k.fB := by
      rw [List.getD_eq_getElem _ _ hm]
      have := List.all_eq_true.mp hpos _ (List.getElem_mem hm)
      simpa using this
    have := hent ⟨_, hr⟩ ⟨_, hs'⟩
    simp only [hWm, Matrix.map_apply, lmat, Matrix.of_apply] at this
    simp only [Matrix.vecMul, dotProduct, hC, Matrix.map_apply, lmat, Matrix.of_apply, hCent, Pi.zero_apply]
    exact this
  have hB0 : ∀ r : Fin rB, (ℓ : ℤ) ∣ a (Fin.castAdd s r) := by
    have ha0 : (fun r => (a (Fin.castAdd s r) : ZMod ℓ)) = 0 := by
      have := congrArg (fun v => Matrix.vecMul v Ci) hvec
      simp only [Matrix.vecMul_vecMul, hCCi, Matrix.vecMul_one, Matrix.zero_vecMul] at this
      exact this
    intro r
    have := congrFun ha0 r
    simpa [ZMod.intCast_zmod_eq_zero_iff_dvd] using this
  -- the étale part: `f₂ ∣ ḡ`, so `ḡ = 0`
  choose bq hbq using hB0
  set g : ℤ[X] := ∑ i : Fin s, Polynomial.C (a (Fin.natAdd rB i)) * Polynomial.X ^ (i : ℕ) with hgdef
  have hg1 : Polynomial.aeval θ (g * toPoly k.f1 ^ k.m) =
      (ℓ : ℤ) • (y - ∑ r : Fin rB, bq r • w r) := by
    have e1 : Polynomial.aeval θ (g * toPoly k.f1 ^ k.m) = ∑ i : Fin s, a (Fin.natAdd rB i) • z i := by
      rw [hgdef, Finset.sum_mul, map_sum]
      refine Finset.sum_congr rfl fun i _ => ?_
      rw [mul_assoc, map_mul, Polynomial.aeval_C, Algebra.smul_def]
    have e2 : ∑ r : Fin rB, a (Fin.castAdd s r) • w r = (ℓ : ℤ) • ∑ r : Fin rB, bq r • w r := by
      rw [Finset.smul_sum]
      refine Finset.sum_congr rfl fun r _ => ?_
      rw [hbq r, mul_smul]
    rw [e1, smul_sub, ← e2, ← hy]; abel
  set ψ : ℤ →+* ZMod ℓ := Int.castRingHom (ZMod ℓ) with hψ
  set A := Mℓ θ with hA
  have hgA : Polynomial.aeval A ((g * toPoly k.f1 ^ k.m).map ψ) = 0 := by
    rw [hψ, ← algebraMap_int_eq, Polynomial.aeval_map_algebraMap, hA, Polynomial.aeval_algHom_apply, hg1,
      map_zsmul, ← Int.cast_smul_eq_zsmul (ZMod ℓ), Int.cast_natCast, ZMod.natCast_self, zero_smul]
  have hchar : A.charpoly = (toPoly k.chi).map ψ := by
    have := Matrix.charpoly_map (LinearMap.toMatrix b b θ) (Int.castRingHom (ZMod ℓ))
    rw [LinearMap.charpoly_toMatrix, hχ] at this
    exact this
  have hdvd := charpoly_dvd_pow A _ hgA
  rw [hchar] at hdvd
  have hf2 : (toPoly k.f2).map ψ ∣ (toPoly k.chi).map ψ := by
    rw [← map_eq_of_peqMod hℓq hf2c2, toPoly_pmul, Polynomial.map_mul]; exact dvd_mul_right _ _
  have hcop : IsCoprime ((toPoly k.f2).map ψ) ((toPoly k.f1).map ψ) := by
    have := map_eq_of_peqMod hℓq hcop
    rw [toPoly_padd, toPoly_pmul, toPoly_pmul, toPoly_single, Polynomial.map_add, Polynomial.map_mul,
      Polynomial.map_mul, Polynomial.map_C, map_one] at this
    exact ⟨_, _, this⟩
  have hsep : Squarefree ((toPoly k.f2).map ψ) := by
    apply Polynomial.Separable.squarefree
    have := map_eq_of_peqMod hℓq hbez2
    rw [toPoly_padd, toPoly_pmul, toPoly_pmul, toPoly_pderiv, toPoly_single, Polynomial.map_add, Polynomial.map_mul,
      Polynomial.map_mul, Polynomial.map_C, map_one, ← Polynomial.derivative_map] at this
    exact ⟨_, _, this⟩
  have hsplit : ((g * toPoly k.f1 ^ k.m).map ψ) ^ Fintype.card (Fin k.fB ⊕ Fin (b1 - k.fB)) =
      (g.map ψ) ^ Fintype.card (Fin k.fB ⊕ Fin (b1 - k.fB)) *
        ((toPoly k.f1).map ψ) ^ (k.m * Fintype.card (Fin k.fB ⊕ Fin (b1 - k.fB))) := by
    rw [Polynomial.map_mul, Polynomial.map_pow, mul_pow, ← pow_mul]
  have h3 : (toPoly k.f2).map ψ ∣ (g.map ψ) ^ Fintype.card (Fin k.fB ⊕ Fin (b1 - k.fB)) := by
    have := hf2.trans hdvd
    rw [hsplit] at this
    exact (hcop.pow_right).dvd_of_dvd_mul_right this
  have h4 : (toPoly k.f2).map ψ ∣ g.map ψ := by
    rcases Nat.eq_zero_or_pos (Fintype.card (Fin k.fB ⊕ Fin (b1 - k.fB))) with h0 | hpos
    · rw [h0, pow_zero] at h3; exact (isUnit_of_dvd_one h3).dvd
    · exact (hsep.dvd_pow_iff_dvd hpos.ne').mp h3
  have hmon := monic_toPoly (f := k.f2) (d := s) (by omega) (by rw [hs]; exact hf2lead)
  have hg0 : g.map ψ = 0 := by
    apply Polynomial.eq_zero_of_dvd_of_degree_lt h4
    rw [Polynomial.degree_eq_natDegree (hmon.1.map ψ).ne_zero, hmon.1.natDegree_map, hmon.2, hgdef]
    exact lt_of_le_of_lt Polynomial.degree_map_le (Polynomial.degree_sum_fin_lt _)
  refine Fin.addCases (fun r => ?_) (fun i => ?_) k0
  · exact ⟨bq r, hbq r⟩
  · have := congrArg (fun p => p.coeff (i : ℕ)) hg0
    simp only [Polynomial.coeff_map, Polynomial.coeff_zero, hgdef, GL2.HeckeZ.Cert.coeff_sum_fin] at this
    rw [hψ, eq_intCast, ZMod.intCast_zmod_eq_zero_iff_dvd] at this
    exact this

/-! ### The cell theorem -/

/-- `𝕋' = ℤ[X p : p ∈ P]`. -/
def Tp (X : ℕ → Module.End ℤ L) (P : List ℕ) : Subalgebra ℤ (Module.End ℤ L) :=
  Algebra.adjoin ℤ (X '' {p | p ∈ P})

lemma dvd_gcdList {ℓ : ℕ} {l : List ℤ} (h : ∀ x ∈ l, (ℓ : ℤ) ∣ x) : ℓ ∣ GL2.HeckeZ.Exec.gcdList l := by
  induction l with
  | nil => simp [GL2.HeckeZ.Exec.gcdList]
  | cons a l ih =>
    simp only [GL2.HeckeZ.Exec.gcdList]
    refine Nat.dvd_gcd ?_ (ih fun x hx => h x (List.mem_cons_of_mem _ hx))
    exact Int.natCast_dvd.mp (h a (List.mem_cons_self ..))

lemma prime_dvd_foldr {ℓ : ℕ} (hℓ : ℓ.Prime) (l : List ℕ) (h : ℓ ∣ l.foldr (· * ·) 1) : ∃ q ∈ l, ℓ ∣ q := by
  induction l with
  | nil => simp at h; exact absurd h hℓ.ne_one
  | cons a l ih =>
    rcases (Nat.Prime.dvd_mul hℓ).mp h with h1 | h1
    · exact ⟨a, List.mem_cons_self .., h1⟩
    · obtain ⟨q, hq, hq'⟩ := ih h1
      exact ⟨q, List.mem_cons_of_mem _ hq, hq'⟩

lemma mem_Tp {X : ℕ → Module.End ℤ L} {P : List ℕ} {p : ℕ} (hp : p ∈ P) : X p ∈ Tp X P :=
  Algebra.subset_adjoin ⟨p, hp, rfl⟩

lemma Tp_le (X : ℕ → Module.End ℤ L) (P : List ℕ) : Tp X P ≤ GL2.HeckeZ.Bridge.Tfr X :=
  Algebra.adjoin_mono (by rintro _ ⟨p, _, rfl⟩; exact ⟨p, rfl⟩)

/-- **The cell theorem** (01 Thm. 4.1 at one level-1 cell, the `ℤ`-form of 07 §2.2 LC-10): a passing cell certificate,
`rank 𝕋^fr ≤ d`, TB-χ and TB-mat give `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ cof`, and `𝕋^fr = 𝕋'` when `cof = 1`. -/
theorem cell_sound (c : CellCert) (hc : cellCheck c = true) (X : ℕ → Module.End ℤ L)
    (hrank : RankLE (GL2.HeckeZ.Bridge.Tfr X) c.d) (hχ : ∀ it ∈ c.items, (X it.p).charpoly = toPoly it.chi)
    (hmat : ∀ k ∈ c.k1, ∃ (b : Module.Basis (Fin c.b1) ℤ L) (U : Matrix (Fin c.b1) (Fin c.b1) (ZMod k.q)), IsUnit U ∧
      ∀ j < k.gens.length,
        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) = lmat c.b1 k.q (k.mats.getD j []) * U)
    (hmatm : ∀ k ∈ c.k1m, ∃ (b : Module.Basis (Fin k.fB ⊕ Fin (c.b1 - k.fB)) ℤ L)
      (U : Matrix (Fin k.fB ⊕ Fin (c.b1 - k.fB)) (Fin k.fB ⊕ Fin (c.b1 - k.fB)) (ZMod k.q)), IsUnit U ∧
      ∀ j < k.gens.length, ∃ Y : Matrix (Fin (c.b1 - k.fB)) (Fin (c.b1 - k.fB)) (ZMod k.q),
        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) =
          Matrix.fromBlocks (lmat k.fB k.q (k.mats.getD j [])) 0 0 Y * U) :
    (∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ c.cof → Loc (GL2.HeckeZ.Bridge.Tfr X) (Tp X c.P) ℓ) ∧
      (c.cof = 1 → GL2.HeckeZ.Bridge.Tfr X = Tp X c.P) := by
  simp only [cellCheck, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.not_eq_true',
    List.isEmpty_eq_false_iff] at hc
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨_, _⟩, hk2⟩, hitP⟩, hk1⟩, hk1P⟩, hk1m⟩, hk1mP⟩, hk1mχ⟩, hne⟩, hG⟩, -⟩, -⟩ := hc
  have hT := Tp_le X c.P
  have hloc : ∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ c.cof → Loc (GL2.HeckeZ.Bridge.Tfr X) (Tp X c.P) ℓ := by
    intro ℓ hℓ hℓc
    by_cases hD : ∃ it ∈ c.items, ¬ (ℓ : ℤ) ∣ it.D
    · -- (C2): `ℓ ∤ D` for some `θ = T_p`
      obtain ⟨it, hit, hℓD⟩ := hD
      have hs := k2OK_sound (List.all_eq_true.mp hk2 it hit)
      have hpP : it.p ∈ c.P := by simpa using List.all_eq_true.mp hitP it hit
      have hθ : X it.p ∈ Tp X c.P := mem_Tp hpP
      have hfχ : toPoly it.f ∣ (X it.p).charpoly := by rw [hχ it hit, ← hs.dvd]; exact dvd_mul_right _ _
      refine loc_of_modIndep hT hrank hℓ _ (fun k : Fin c.d => (Tp X c.P).pow_mem hθ (k : ℕ)) ?_
      exact modIndep_pow (X it.p) (toPoly it.f) (toPoly it.u) (toPoly it.v) it.D hs.monic hs.deg hfχ hs.bez hs.hD
        hℓ hℓD
    · -- (C3): `ℓ` divides every `D`, hence a covered modulus
      push Not at hD
      have hG' : ℓ ∣ c.G := dvd_gcdList fun x hx => by
        obtain ⟨it, hit, rfl⟩ := List.mem_map.mp hx; exact hD it hit
      have h1 : ℓ ∣ c.cof * c.qprod ^ c.ex := hG'.trans (Nat.dvd_of_mod_eq_zero hG)
      have h2 : ℓ ∣ c.qprod := hℓ.dvd_of_dvd_pow (((Nat.Prime.dvd_mul hℓ).mp h1).resolve_left hℓc)
      obtain ⟨q, hq, hℓq⟩ := prime_dvd_foldr hℓ _ h2
      rcases List.mem_append.mp hq with hq | hq
      swap
      · -- a K1m certificate
        obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hq
        have hok := List.all_eq_true.mp hk1m k hk
        have hkk := List.all_eq_true.mp hk1mχ k hk
        simp only [Bool.and_eq_true, decide_eq_true_eq, List.any_eq_true, beq_iff_eq] at hkk
        obtain ⟨_, it, hitm, hitp, hitχ⟩ := hkk
        have hχk : (X (k.gens.getD k.th 0)).charpoly = toPoly k.chi := by
          rw [← hitp, ← hitχ]; exact hχ it hitm
        obtain ⟨t, ht, hind⟩ := k1m_modIndep (b1 := c.b1) k hok X hχk (hmatm k hk) hℓ hℓq
        refine loc_of_modIndep hT hrank hℓ t (fun i => ?_) hind
        have hgP := List.all_eq_true.mp hk1mP k hk
        refine Algebra.adjoin_mono ?_ (ht i)
        rintro _ ⟨g, hg, rfl⟩
        exact ⟨g, by simpa using List.all_eq_true.mp hgP g hg, rfl⟩
      obtain ⟨k, hk, rfl⟩ := List.mem_map.mp hq
      have hok := List.all_eq_true.mp hk1 k hk
      refine loc_of_modIndep hT hrank hℓ _ (fun i => ?_) (k1_modIndep k hok X (hmat k hk) hℓ hℓq)
      have hgP := List.all_eq_true.mp hk1P k hk
      unfold wordOp
      refine Subalgebra.list_prod_mem _ fun x hx => ?_
      obtain ⟨j, hjw, rfl⟩ := List.mem_map.mp hx
      by_cases hj : j < k.gens.length
      · apply mem_Tp
        have := List.all_eq_true.mp hgP _ (List.getElem_mem hj)
        rw [List.getD_eq_getElem _ _ hj]; simpa using this
      · -- out-of-range positions are excluded by `k1OK`; `X 0` is still in `𝕋^fr`, but we need `𝕋'`
        exfalso
        simp only [k1OK, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hok
        obtain ⟨⟨⟨⟨⟨⟨_, _⟩, hwl⟩, _⟩, hwords⟩, _⟩, _⟩ := hok
        have hi : (i : ℕ) < k.words.length := by rw [hwl]; exact i.2
        rw [List.getD_eq_getElem _ _ hi] at hjw
        have := List.all_eq_true.mp (List.all_eq_true.mp hwords _ (List.getElem_mem hi)) j hjw
        exact hj (by simpa using this)
  refine ⟨hloc, fun h1 => le_antisymm (le_of_loc fun ℓ hℓ => hloc ℓ hℓ (by rw [h1]; exact hℓ.not_dvd_one)) hT⟩

/-- **The cell theorem with Eichler–Shimura** (`hES`, `lit:eichler-shimura-levelone`) supplying the rank. -/
theorem cell_sound_ES (c : CellCert) (hc : cellCheck c = true) {p : ℕ} [Fact p.Prime]
    {E : Type*} [Field E] [Algebra ℤ_[p] E] [Algebra ℚ_[p] E] [IsScalarTower ℤ_[p] ℚ_[p] E]
    (X : ℕ → Module.End ℤ L) (hrk : Module.finrank ℤ L = c.b1)
    (hES : GL2.Literature.EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (GL2.HeckeZ.Bridge.Xp p X) E)
    (hχ : ∀ it ∈ c.items, (X it.p).charpoly = toPoly it.chi)
    (hmat : ∀ k ∈ c.k1, ∃ (b : Module.Basis (Fin c.b1) ℤ L) (U : Matrix (Fin c.b1) (Fin c.b1) (ZMod k.q)), IsUnit U ∧
      ∀ j < k.gens.length,
        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) = lmat c.b1 k.q (k.mats.getD j []) * U)
    (hmatm : ∀ k ∈ c.k1m, ∃ (b : Module.Basis (Fin k.fB ⊕ Fin (c.b1 - k.fB)) ℤ L)
      (U : Matrix (Fin k.fB ⊕ Fin (c.b1 - k.fB)) (Fin k.fB ⊕ Fin (c.b1 - k.fB)) (ZMod k.q)), IsUnit U ∧
      ∀ j < k.gens.length, ∃ Y : Matrix (Fin (c.b1 - k.fB)) (Fin (c.b1 - k.fB)) (ZMod k.q),
        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) =
          Matrix.fromBlocks (lmat k.fB k.q (k.mats.getD j [])) 0 0 Y * U) :
    (∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ c.cof → Loc (GL2.HeckeZ.Bridge.Tfr X) (Tp X c.P) ℓ) ∧
      (c.cof = 1 → GL2.HeckeZ.Bridge.Tfr X = Tp X c.P) := by
  have hd : c.d = (c.b1 + 1) / 2 := by
    simp only [cellCheck, Bool.and_eq_true, beq_iff_eq] at hc; exact hc.1.1.1.1.1.1.1.1.1.1.1.1
  have hrank := GL2.HeckeZ.RankES.rankLE_of_ES X hES
  rw [hrk, ← hd] at hrank
  exact cell_sound c hc X hrank hχ hmat hmatm

end GL2.HeckeZ.Check
