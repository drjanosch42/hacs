/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Exec.Sturm
import GL2.HeckeZ.Sturm
import GL2.HeckeZ.Check.K3

/-!
# HeckeZ LC-11: soundness of the Prop. 5.6 certificate, and the cell theorems (slot 25, advances M-HZ2: exit 2 met in the generation form; 17 cells partial)

`07-lean-plan.md` §2.2 LC-11, §5 (common shape), §10 (M-HZ2). `sturmCheck cc sc = true` (Mathlib-free, `GL2.HeckeZ.Exec.Sturm`)
gives the data of `GL2.HeckeZ.Sturm.heckeZeiscert`:
* the items become `EisData`: `c(T_p)` kills `(T_p − λ_p)`-torsion because `f(T_p) = 0` (01 (C1), `aeval_eq_zero_of_C1`, with
  the rank from Eichler–Shimura) and `(x − λ_p) c = f`; `λ_p` is a simple root of TB-χ;
* `g = gcd c(λ_p)` lies in the ideal `(c(λ_p))` (`GL2.HeckeZ.K3.gcdList_mem`);
* a read (TB-mat: the matrix of `T_{p₀}` on `L/ℓ^{s+1}`) gives the read hypothesis of `heckeZeiscert` (`zero_iff_of_conj`: vanishing modulo
  `ℓ^s` and non-vanishing modulo `ℓ^{s+1}` of `c(T_{p₀})` are conjugation-invariant).

**`cell56_sound`** (01 Prop. 5.6 alone): `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ cof56`, and `𝕋^fr = 𝕋'` if `cof56 = 1`.
Hypotheses: `hES` and `hSturm` (literature), TB-χ (`hχ`), TB-mat for the reads (`hread`). No K1 / K1m data are used.
**`cell_sturm_sound`** (with 01 Thm. 4.1, `cell_sound_ES`): the same at every prime `ℓ ∤ cof = gcd(cc.cof, cof56)`; adds
`hmat`, `hmatm`.
-/

namespace GL2.HeckeZ.Check

open GL2.HeckeZ.Exec GL2.HeckeZ.Cert GL2.HeckeZ.Sturm GL2.HeckeZ.K3 GL2.Literature Polynomial
open scoped TensorProduct

set_option linter.unusedSectionVars false

/-! ### The executable pieces -/

lemma primeB_sound {n b : ℕ} (h : primeB n b = true) : n.Prime := by
  simp only [primeB, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range, Bool.or_eq_true,
    bne_iff_ne, ne_eq] at h
  obtain ⟨⟨h2, hb⟩, hall⟩ := h
  by_contra hnp
  have hn1 : n ≠ 1 := by omega
  have hmp : n.minFac.Prime := Nat.minFac_prime hn1
  have hmd : n.minFac ∣ n := Nat.minFac_dvd n
  have hsq : n.minFac * n.minFac ≤ n := by
    have := Nat.minFac_sq_le_self (by omega) hnp; rwa [sq] at this
  have hmb : n.minFac < b + 1 := by
    by_contra h; push Not at h; have := Nat.mul_le_mul h h; omega
  rcases hall _ hmb with (h | h) | h
  · exact absurd hmp.two_le (by omega)
  · omega
  · exact h (Nat.mod_eq_zero_of_dvd hmd)

lemma primeB_complete {n : ℕ} (hn : n.Prime) : primeB n n = true := by
  simp only [primeB, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true, List.mem_range, Bool.or_eq_true,
    bne_iff_ne, ne_eq]
  have h2 := hn.two_le
  refine ⟨⟨h2, by nlinarith⟩, fun m _ => ?_⟩
  by_cases hm : m < 2 ∨ n < m * m
  · exact Or.inl hm
  · right
    push Not at hm
    intro h0
    have hdvd : m ∣ n := Nat.dvd_of_mod_eq_zero h0
    rcases hn.eq_one_or_self_of_dvd m hdvd with h1 | h1
    · omega
    · subst h1; nlinarith

lemma peval_eq (a : ℤ) (p : List ℤ) : (toPoly p).eval a = peval a p := by
  induction p with
  | nil => simp [toPoly, peval]
  | cons c p ih => simp [toPoly, peval, ih]

lemma toPoly_lin (a : ℤ) : toPoly [-a, 1] = Polynomial.X - C a := by
  simp [toPoly, sub_eq_add_neg, add_comm]

lemma eisLamE_eq (k p : ℕ) : eisLamE k p = eisLam k p := rfl

variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]

/-- The matrix of `c(θ)` read modulo `q` is `c` of the matrix of `θ`. -/
lemma toMatrix_aeval_cast {n : ℕ} (b : Module.Basis (Fin n) ℤ L) (θ : Module.End ℤ L) (c : ℤ[X]) (q : ℕ) :
    (LinearMap.toMatrix b b (aeval θ c)).map (Int.cast : ℤ → ZMod q) =
      aeval ((LinearMap.toMatrix b b θ).map (Int.cast : ℤ → ZMod q)) c := by
  have h1 : LinearMap.toMatrix b b (aeval θ c) = aeval (LinearMap.toMatrix b b θ) c :=
    (Polynomial.aeval_algHom_apply (LinearMap.toMatrixAlgEquiv b).toAlgHom θ c).symm
  rw [h1]
  exact (Polynomial.aeval_algHom_apply ((Int.castRingHom (ZMod q)).mapMatrix (m := Fin n)).toIntAlgHom
    (LinearMap.toMatrix b b θ) c).symm

/-- **The read** from TB-mat: `c(M) ≡ 0 (ℓ^s)`, `c(M) ≢ 0 (ℓ^{s+1})` for the matrix `M` of `θ` on `L/ℓ^{s+1}` give an entry of
`c(θ)` in a `ℤ`-basis of valuation exactly `s`. -/
lemma read_entry {n ℓ s : ℕ} (b : Module.Basis (Fin n) ℤ L) (θ : Module.End ℤ L) (c : List ℤ) (M : List (List ℕ))
    (U : Matrix (Fin n) (Fin n) (ZMod (ℓ ^ (s + 1)))) (hU : IsUnit U)
    (hconj : U * (LinearMap.toMatrix b b θ).map (Int.cast) = lmat n (ℓ ^ (s + 1)) M * U) (hℓ : 0 < ℓ)
    (h0 : allFin n (fun i j => ment (pevalMat n (ℓ ^ (s + 1)) M c) i j % ℓ ^ s == 0) = true)
    (h1 : allFin n (fun i j => ment (pevalMat n (ℓ ^ (s + 1)) M c) i j % ℓ ^ (s + 1) == 0) = false) :
    ∃ i j : Fin n, (ℓ : ℤ) ^ s ∣ b.repr (aeval θ (toPoly c) (b j)) i ∧
      ¬ (ℓ : ℤ) ^ (s + 1) ∣ b.repr (aeval θ (toPoly c) (b j)) i := by
  have hm0 : 0 < ℓ ^ (s + 1) := pow_pos hℓ _
  set A : Fin n → Fin n → ℤ := fun i j => b.repr (aeval θ (toPoly c) (b j)) i with hA
  set C : Fin n → Fin n → ℤ := fun i j => (ment (pevalMat n (ℓ ^ (s + 1)) M c) i j : ℤ) with hC
  have hzA : zmat (ℓ ^ (s + 1)) A =
      aeval ((LinearMap.toMatrix b b θ).map (Int.cast : ℤ → ZMod (ℓ ^ (s + 1)))) (toPoly c) := by
    rw [← toMatrix_aeval_cast]; ext i j; simp [zmat, hA, LinearMap.toMatrix_apply]
  have hzC : zmat (ℓ ^ (s + 1)) C = lmat n (ℓ ^ (s + 1)) (pevalMat n (ℓ ^ (s + 1)) M c) := by
    ext i j; simp [zmat, hC, lmat]
  have hconjc : U * zmat (ℓ ^ (s + 1)) A = zmat (ℓ ^ (s + 1)) C * U := by
    rw [hzA, hzC, lmat_pevalMat hm0]; exact conj_aeval hconj _
  have hs : ℓ ^ s ∣ ℓ ^ (s + 1) := pow_dvd_pow _ (Nat.le_succ s)
  have hAs : ∀ i j, ((ℓ ^ s : ℕ) : ℤ) ∣ A i j := by
    have := (zero_iff_of_conj hs hU hconjc).mpr (by
      rw [mapMatrix_zmat, zmat_eq_zero]; intro i j
      have := (allFin_iff n _).mp h0 i i.2 j j.2
      simp only [beq_iff_eq] at this
      simp only [hC]; exact_mod_cast Nat.dvd_of_mod_eq_zero this)
    rw [mapMatrix_zmat, zmat_eq_zero] at this; exact this
  have hAs1 : ∃ i j, ¬ ((ℓ ^ (s + 1) : ℕ) : ℤ) ∣ A i j := by
    by_contra hcon
    push Not at hcon
    have := (zero_iff_of_conj (dvd_refl (ℓ ^ (s + 1))) hU hconjc).mp
      (by rw [mapMatrix_zmat, zmat_eq_zero]; exact hcon)
    rw [mapMatrix_zmat, zmat_eq_zero] at this
    have hall : allFin n (fun i j => ment (pevalMat n (ℓ ^ (s + 1)) M c) i j % ℓ ^ (s + 1) == 0) = true := by
      rw [allFin_iff]; intro i hi j hj
      have := this ⟨i, hi⟩ ⟨j, hj⟩
      simp only [hC] at this
      simp only [beq_iff_eq]; exact Nat.mod_eq_zero_of_dvd (by exact_mod_cast this)
    rw [h1] at hall; exact absurd hall (by simp)
  obtain ⟨i, j, hij⟩ := hAs1
  exact ⟨i, j, by have := hAs i j; push_cast at this; exact this, by push_cast at hij; exact hij⟩

/-! ### The cell theorems -/

section Cells

variable {p : ℕ} [Fact p.Prime] {E : Type*} [Field E] [Algebra ℤ_[p] E] [Algebra ℚ_[p] E] [IsScalarTower ℤ_[p] ℚ_[p] E]

/-- **01 Prop. 5.6 at one level-1 cell** (LC-11): under `hES`, `hSturm`, TB-χ and TB-mat for the reads, a passing
`sturmCheck` gives `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ cof56`, and `𝕋^fr = 𝕋'` when `cof56 = 1`. -/
theorem cell56_sound (cc : CellCert) (hcc : cellCheck cc = true) (sc : SturmCert) (hsc : sturmCheck cc sc = true)
    (X : ℕ → Module.End ℤ L) (hrk : Module.finrank ℤ L = cc.b1)
    (hES : EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (GL2.HeckeZ.Bridge.Xp p X) E)
    (hS : SturmCuspLevelOne X (cc.n + 2))
    (hχ : ∀ it ∈ cc.items, (X it.p).charpoly = toPoly it.chi)
    (hread : ∀ rd ∈ sc.reads, ∃ (b : Module.Basis (Fin cc.b1) ℤ L)
      (U : Matrix (Fin cc.b1) (Fin cc.b1) (ZMod (rd.ell ^ (rd.s + 1)))), IsUnit U ∧
        U * (LinearMap.toMatrix b b (X rd.p)).map (Int.cast) = lmat cc.b1 (rd.ell ^ (rd.s + 1)) rd.mat * U) :
    (∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ sc.cof56 → Loc (GL2.HeckeZ.Bridge.Tfr X) (Tp X cc.P) ℓ) ∧
      (sc.cof56 = 1 → GL2.HeckeZ.Bridge.Tfr X = Tp X cc.P) := by
  have hsc' := hsc
  simp only [sturmCheck, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq, Bool.not_eq_true',
    List.isEmpty_eq_false_iff, bne_iff_ne, ne_eq] at hsc'
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hk, hne⟩, hcov⟩, hit⟩, hrd⟩, hg0⟩, hcof1⟩, hdiv⟩, -⟩ := hsc'
  -- the rank and the K2g items of the cell
  have hcc' := hcc
  simp only [cellCheck, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hcc'
  have hd : cc.d = (cc.b1 + 1) / 2 := hcc'.1.1.1.1.1.1.1.1.1.1.1.1
  have hk2 : cc.items.all (k2OK cc.d) = true := hcc'.1.1.1.1.1.1.1.1.1.1.2
  have hrank : RankLE (GL2.HeckeZ.Bridge.Tfr X) cc.d := by
    have := GL2.HeckeZ.RankES.rankLE_of_ES X hES; rwa [hrk, ← hd] at this
  -- each Prop. 5.6 item
  have hitem : ∀ it ∈ sc.items, ∃ ci ∈ cc.items, ci.p = it.p ∧
      toPoly ci.f = (Polynomial.X - C (eisLam sc.k it.p)) * toPoly it.c ∧
      toPoly ci.chi = (Polynomial.X - C (eisLam sc.k it.p)) * toPoly it.r ∧ it.p.Prime ∧ it.p ∈ cc.P ∧
      (toPoly it.r).eval (eisLam sc.k it.p) ≠ 0 := by
    intro it hm
    have h := List.all_eq_true.mp hit it hm
    simp only [sitemOK, Bool.and_eq_true, List.any_eq_true, bne_iff_ne, ne_eq, List.contains_iff_mem,
      beq_iff_eq] at h
    obtain ⟨⟨⟨hpr, hP⟩, ci, hci, ⟨hcp, hf⟩, hχr⟩, hr⟩ := h
    refine ⟨ci, hci, hcp, ?_, ?_, primeB_sound hpr, hP, ?_⟩
    · rw [← toPoly_eq_of_peq hf, toPoly_pmul, toPoly_lin, eisLamE_eq]
    · rw [← toPoly_eq_of_peq hχr, toPoly_pmul, toPoly_lin, eisLamE_eq]
    · rw [peval_eq, ← eisLamE_eq]; exact hr
  -- the `EisData`
  let D : EisData X sc.k sc.items SItem.p :=
    { c := fun it => toPoly it.c
      r := fun it => toPoly it.r
      prime := fun it hm => (hitem it hm).choose_spec.2.2.2.2.1
      hc := by
        intro it hm
        obtain ⟨ci, hci, hcp, hf, -, -, -, -⟩ := hitem it hm
        have hs := k2OK_sound (List.all_eq_true.mp hk2 ci hci)
        have hθ : X ci.p ∈ GL2.HeckeZ.Bridge.Tfr X := Algebra.subset_adjoin ⟨ci.p, rfl⟩
        have hfχ : toPoly ci.f ∣ (X ci.p).charpoly := by rw [hχ ci hci, ← hs.dvd]; exact dvd_mul_right _ _
        have h0 := (aeval_eq_zero_of_C1 hrank (X ci.p) hθ (toPoly ci.f) (toPoly ci.u) (toPoly ci.v) ci.D hs.monic
          hs.deg hfχ hs.bez hs.hD).1
        rw [hcp, hf, map_mul, map_sub, aeval_X, aeval_C, Algebra.algebraMap_eq_smul_one] at h0
        simpa [Module.End.one_eq_id] using h0
      hχ := by
        intro it hm
        obtain ⟨ci, hci, hcp, -, hcr, -, -, -⟩ := hitem it hm
        rw [← hcp, hχ ci hci, hcp, hcr]
      hr := fun it hm => (hitem it hm).choose_spec.2.2.2.2.2.2 }
  have hQP : ∀ q ∈ sc.items, SItem.p q ∈ cc.P := fun it hm => (hitem it hm).choose_spec.2.2.2.2.2.1
  -- `g ∈ (c(λ_p))`
  have hgI : (sc.g : ℤ) ∈ D.gIdeal := by
    refine Ideal.span_mono ?_ (gcdList_mem (sc.items.map (SItem.val sc.k)))
    rintro x hx
    obtain ⟨it, hm, rfl⟩ := List.mem_map.mp hx
    exact ⟨it, hm, by simp only [D, SItem.val, peval_eq, eisLamE_eq]⟩
  -- Sturm's bound
  have hk' : sc.k = cc.n + 2 := hk
  have hP : ∀ q : ℕ, q.Prime → q ≤ sc.k / 12 → q ∈ cc.P := by
    intro q hq hqk
    have := List.all_eq_true.mp hcov q (List.mem_range.mpr (by omega))
    simp only [Bool.or_eq_true, Bool.not_eq_true', List.contains_iff_mem] at this
    rcases this with h | h
    · rw [primeB_complete hq] at h; exact absurd h (by simp)
    · exact h
  have hS' : SturmCuspLevelOne X sc.k := by rw [hk']; exact hS
  have hloc : ∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ sc.cof56 → Loc (GL2.HeckeZ.Bridge.Tfr X) (Tp X cc.P) ℓ := by
    intro ℓ hℓ hℓc
    by_cases hgℓ : (ℓ : ℤ) ∣ (sc.g : ℤ)
    · -- `ℓ ∣ g`: a read at `ℓ`
      have h1 : ℓ ∣ sc.cof56 * sc.rprod ^ sc.ex :=
        (Int.natCast_dvd_natCast.mp hgℓ).trans (Nat.dvd_of_mod_eq_zero hdiv)
      have h2 : ℓ ∣ sc.rprod := hℓ.dvd_of_dvd_pow (((Nat.Prime.dvd_mul hℓ).mp h1).resolve_left hℓc)
      obtain ⟨e, he, hℓe⟩ := prime_dvd_foldr hℓ _ h2
      obtain ⟨rd, hrdm, rfl⟩ := List.mem_map.mp he
      have hok := List.all_eq_true.mp hrd rd hrdm
      simp only [sreadOK, Bool.and_eq_true, List.any_eq_true, beq_iff_eq, bne_iff_ne, ne_eq, Bool.not_eq_true'] at hok
      obtain ⟨⟨⟨hpr, -⟩, it, hitm, hitp, ⟨h0, h1'⟩, hval⟩, hgv⟩ := hok
      have hprime := primeB_sound hpr
      have hℓeq : ℓ = rd.ell := (Nat.prime_dvd_prime_iff_eq hℓ hprime).mp hℓe
      subst hℓeq
      obtain ⟨b, U, hU, hconj⟩ := hread rd hrdm
      rw [← hitp] at hconj
      obtain ⟨i, j, hij, hij1⟩ := read_entry b (X it.p) it.c rd.mat U hU hconj hℓ.pos h0 h1'
      refine heckeZeiscert_ES (p := p) (E := E) hES hS' hP D hQP hgI hℓ (v := rd.v) ?_ ?_
      · intro hdvd
        apply hgv
        exact Nat.mod_eq_zero_of_dvd (by exact_mod_cast hdvd)
      · refine Or.inr ⟨it, hitm, cc.b1, b, i, j, rd.s, hij, hij1, ?_⟩
        simp only [SItem.val] at hval
        have := Int.dvd_of_emod_eq_zero hval
        simpa [D, peval_eq, eisLamE_eq] using this
    · exact heckeZeiscert_ES (p := p) (E := E) hES hS' hP D hQP hgI hℓ (v := 0) (by simpa using hgℓ) (Or.inl rfl)
  refine ⟨hloc, fun h1 => le_antisymm (le_of_loc fun ℓ hℓ => hloc ℓ hℓ (by rw [h1]; exact hℓ.not_dvd_one))
    (Tp_le X cc.P)⟩

/-- **01 Prop. 5.6 with 01 Thm. 4.1 at one level-1 cell**: `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ cof = gcd(cc.cof, cof56)`,
and `𝕋^fr = 𝕋'` when `cof = 1`. -/
theorem cell_sturm_sound (cc : CellCert) (hcc : cellCheck cc = true) (sc : SturmCert) (hsc : sturmCheck cc sc = true)
    (X : ℕ → Module.End ℤ L) (hrk : Module.finrank ℤ L = cc.b1)
    (hES : EichlerShimuraLevelOne (ℤ_[p] ⊗[ℤ] L) (GL2.HeckeZ.Bridge.Xp p X) E)
    (hS : SturmCuspLevelOne X (cc.n + 2))
    (hχ : ∀ it ∈ cc.items, (X it.p).charpoly = toPoly it.chi)
    (hmat : ∀ k ∈ cc.k1, ∃ (b : Module.Basis (Fin cc.b1) ℤ L) (U : Matrix (Fin cc.b1) (Fin cc.b1) (ZMod k.q)), IsUnit U ∧
      ∀ j < k.gens.length,
        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) = lmat cc.b1 k.q (k.mats.getD j []) * U)
    (hmatm : ∀ k ∈ cc.k1m, ∃ (b : Module.Basis (Fin k.fB ⊕ Fin (cc.b1 - k.fB)) ℤ L)
      (U : Matrix (Fin k.fB ⊕ Fin (cc.b1 - k.fB)) (Fin k.fB ⊕ Fin (cc.b1 - k.fB)) (ZMod k.q)), IsUnit U ∧
      ∀ j < k.gens.length, ∃ Y : Matrix (Fin (cc.b1 - k.fB)) (Fin (cc.b1 - k.fB)) (ZMod k.q),
        U * (LinearMap.toMatrix b b (X (k.gens.getD j 0))).map (Int.cast) =
          Matrix.fromBlocks (lmat k.fB k.q (k.mats.getD j [])) 0 0 Y * U)
    (hread : ∀ rd ∈ sc.reads, ∃ (b : Module.Basis (Fin cc.b1) ℤ L)
      (U : Matrix (Fin cc.b1) (Fin cc.b1) (ZMod (rd.ell ^ (rd.s + 1)))), IsUnit U ∧
        U * (LinearMap.toMatrix b b (X rd.p)).map (Int.cast) = lmat cc.b1 (rd.ell ^ (rd.s + 1)) rd.mat * U) :
    (∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ sc.cof → Loc (GL2.HeckeZ.Bridge.Tfr X) (Tp X cc.P) ℓ) ∧
      (sc.cof = 1 → GL2.HeckeZ.Bridge.Tfr X = Tp X cc.P) := by
  have hcof : sc.cof = Nat.gcd cc.cof sc.cof56 := by
    simp only [sturmCheck, Bool.and_eq_true, beq_iff_eq] at hsc; exact hsc.2
  have h41 := (cell_sound_ES cc hcc X hrk hES hχ hmat hmatm).1
  have h56 := (cell56_sound cc hcc sc hsc X hrk hES hS hχ hread).1
  have hloc : ∀ ℓ : ℕ, ℓ.Prime → ¬ ℓ ∣ sc.cof → Loc (GL2.HeckeZ.Bridge.Tfr X) (Tp X cc.P) ℓ := by
    intro ℓ hℓ hℓc
    by_cases h1 : ℓ ∣ cc.cof
    · refine h56 ℓ hℓ fun h2 => hℓc ?_
      rw [hcof]; exact Nat.dvd_gcd h1 h2
    · exact h41 ℓ hℓ h1
  refine ⟨hloc, fun h1 => le_antisymm (le_of_loc fun ℓ hℓ => hloc ℓ hℓ (by rw [h1]; exact hℓ.not_dvd_one))
    (Tp_le X cc.P)⟩

end Cells

end GL2.HeckeZ.Check
