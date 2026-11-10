/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Cert

/-!
# HeckeZ 03 B8(1), B9 (LB-5): level one at `ℓ ∈ {2, 3}` (L-HZ2, M-HZ3)

`docs/implementation/slot24/heckez/03-bockstein-tor.md` §3.2 (B8, B9, **in A3-M1's corrected form**), §8.1 (LB-5);
`07-lean-plan.md` §2.3 (LB-5), §3 (`hScalar23`, `hTellRank`); [C] `prop:heckeZlevelone23`(1), (6) (1.25.0-pre.3).

**The two carried [C] statements** (the `hrank` precedent of `chk:stablecert-sound`; they are hypotheses of every theorem that
uses them, never axioms):
* `HScalar23 ℓ X` — [C, prop:ltwothreescalar]: `T_p = 1 + p` on `H^q(Γ; V/ℓ)`, `ℓ ∈ {2, 3}`, level one, read on the
  subquotient `L/ℓ` of `H¹(Γ; V/ℓ)` (B8(1)'s proof; `L/ℓ ↞ H¹/ℓ ↪ H¹(Γ; V/ℓ)` are `𝕋`-maps, [C, prop:heckeZuct](4)): for
  every prime `p ≠ ℓ` and `v ∈ L`, `T_p v − (1+p) v ∈ ℓ L`;
* `HTellRank ℓ b X` — [C, lem:tellrank]: `rank(T_ℓ | H¹(Γ; M_n)) ≤ 2`, read on `L/ℓ` (rank does not grow on subquotients):
  the reduction modulo `ℓ` of the matrix of `T_ℓ` in a `ℤ`-basis `b` of `L` has rank `≤ 2` (basis-independent).

**Results.**
* B8(1), the lattice clause: `exists_eq_smul_of_scalar` — `T_p − (1 + p) ∈ ℓ · End(L)` (**not** nilpotent on `L`, A3-M1);
  `pow_eq_zero_of_mem_smul` — a generic lemma: `φ M ⊆ ℓ M` and `ℓ^K M = 0` give `φ^K = 0`. Its premise is available from
  the hypotheses for `M = L/ℓ^K` only; B8(1)'s nilpotency on `tors_ℓ`, `H^{q+1}[ℓ^K]` and `H^q(Γ; V/ℓ^K)` is **not** derived here.
  The hypotheses are the reads on `L/ℓ`: the passage from `H¹(Γ; V/ℓ)` to `L/ℓ` is assumed, not proved (M-HZ3 audit B2).
* B9, the degree bound: `exists_monic_aeval_eq_zero` — over a field, a matrix `Y` of rank `r` satisfies a monic polynomial of
  degree `≤ r + 1` (`g(Y) Y = 0` for `g` the characteristic polynomial of `Y` on its image); `adjoin_le_span` — so
  `𝔽_ℓ[Ȳ] ⊆ span(1, Ȳ, Ȳ²)` when `r ≤ 2`, `finrank_adjoin_le_three` — `dim_{𝔽_ℓ} A₁(L/ℓ) ≤ 3`.
* B9's consequence, **`not_modIndep_levelOne23`**: if `d ≥ 4`, **no `d` elements of `𝕋^fr = ℤ[T_p : p prime]` are linearly
  independent modulo `ℓ`** — Thm. 4.1's (C3) (`GL2.HeckeZ.Cert.ModIndep`) cannot pass at 2 or 3 for any operator set, and
  `𝕋^fr_ℓ` is not saturated on `L_ℓ` ([C, prop:saturation](2)). This is why M-HZ2's level-one cells with `d ≥ 3` stay partial
  at 2 and 3 (`LHZ1_report.md` §4) for every `d ≥ 4`: their completion must come from Prop. 5.6 (LC-11), not from (C3).
-/

namespace GL2.HeckeZ.Bock

open Polynomial

section Nil

variable {M : Type*} [AddCommGroup M]

/-- `φ M ⊆ ℓ M` gives `φ^K M ⊆ ℓ^K M`. -/
theorem pow_apply_mem_smul (φ : Module.End ℤ M) (ℓ : ℤ) (h : ∀ v, ∃ w, φ v = ℓ • w) (K : ℕ) :
    ∀ v, ∃ w, (φ ^ K) v = ℓ ^ K • w := by
  induction K with
  | zero => intro v; exact ⟨v, by simp⟩
  | succ K ih =>
    intro v
    obtain ⟨w, hw⟩ := h v
    obtain ⟨w', hw'⟩ := ih w
    refine ⟨w', ?_⟩
    rw [pow_succ, Module.End.mul_apply, hw, map_zsmul, hw', smul_smul, pow_succ, mul_comm]

/-- **B8(1), nilpotency on the finite terms**: if `φ M ⊆ ℓ M` and `ℓ^K M = 0` then `φ^K = 0`. -/
theorem pow_eq_zero_of_mem_smul (φ : Module.End ℤ M) (ℓ : ℤ) (h : ∀ v, ∃ w, φ v = ℓ • w) {K : ℕ}
    (hK : ∀ v : M, ℓ ^ K • v = 0) : φ ^ K = 0 := by
  ext v
  obtain ⟨w, hw⟩ := pow_apply_mem_smul φ ℓ h K v
  rw [hw, hK, LinearMap.zero_apply]

end Nil

section Lattice

variable {L : Type*} [AddCommGroup L] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Reduction modulo `ℓ` of the matrix in the basis `b`: `End_ℤ(L) → M_ι(𝔽_ℓ)`. -/
noncomputable def redMat (ℓ : ℕ) (b : Module.Basis ι ℤ L) : Module.End ℤ L →+* Matrix ι ι (ZMod ℓ) :=
  (Int.castRingHom (ZMod ℓ)).mapMatrix.comp (LinearMap.toMatrixAlgEquiv b).toRingEquiv.toRingHom

theorem redMat_apply (ℓ : ℕ) (b : Module.Basis ι ℤ L) (t : Module.End ℤ L) (i j : ι) :
    redMat ℓ b t i j = ((LinearMap.toMatrix b b t i j : ℤ) : ZMod ℓ) := rfl

/-- `red t = 0` iff `t ∈ ℓ · End(L)`. -/
theorem redMat_eq_zero_iff (ℓ : ℕ) (b : Module.Basis ι ℤ L) (t : Module.End ℤ L) :
    redMat ℓ b t = 0 ↔ ∃ y : Module.End ℤ L, t = (ℓ : ℤ) • y := by
  constructor
  · intro h
    have hd : ∀ i j, (ℓ : ℤ) ∣ LinearMap.toMatrix b b t i j := fun i j => by
      have := congrFun (congrFun h i) j
      rw [redMat_apply] at this
      exact (ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mp this
    refine ⟨Matrix.toLin b b (Matrix.of fun i j => LinearMap.toMatrix b b t i j / ℓ), ?_⟩
    apply (LinearMap.toMatrix b b).injective
    rw [map_zsmul, LinearMap.toMatrix_toLin]
    ext i j
    simp [Int.mul_ediv_cancel' (hd i j)]
  · rintro ⟨y, rfl⟩
    ext i j
    rw [redMat_apply, map_zsmul]
    simp

/-- `t v ∈ ℓ L` for all `v` iff `red t = 0`. -/
theorem redMat_eq_zero_of_forall (ℓ : ℕ) (b : Module.Basis ι ℤ L) (t : Module.End ℤ L) (h : ∀ v, ∃ w, t v = (ℓ : ℤ) • w) :
    redMat ℓ b t = 0 := by
  ext i j
  obtain ⟨w, hw⟩ := h (b j)
  rw [redMat_apply, LinearMap.toMatrix_apply, hw, map_zsmul, Finsupp.smul_apply, smul_eq_mul]
  simp

/-- **`hScalar23`** ([C, prop:ltwothreescalar], carried; read on `L/ℓ`). -/
structure HScalar23 (ℓ : ℕ) (X : ℕ → Module.End ℤ L) : Prop where
  /-- `ℓ ∈ {2, 3}` -/
  ell : ℓ = 2 ∨ ℓ = 3
  /-- `T_p ≡ 1 + p` on `L/ℓ` for every prime `p ≠ ℓ` -/
  scalar : ∀ p : ℕ, p.Prime → p ≠ ℓ → ∀ v : L, ∃ w : L, X p v - ((1 + p : ℕ) : ℤ) • v = (ℓ : ℤ) • w

/-- **`hTellRank`** ([C, lem:tellrank], carried; read on `L/ℓ`). -/
structure HTellRank (ℓ : ℕ) (b : Module.Basis ι ℤ L) (X : ℕ → Module.End ℤ L) : Prop where
  /-- `rank(T_ℓ | L/ℓ) ≤ 2` -/
  rank_le : (redMat ℓ b (X ℓ)).rank ≤ 2

/-- **B8(1), the lattice clause**: `T_p − (1 + p) ∈ ℓ · End(L)` for every prime `p ≠ ℓ` (A3-M1: on `L` itself it is only
`ℓ`-adically topologically nilpotent). -/
theorem exists_eq_smul_of_scalar {ℓ : ℕ} (b : Module.Basis ι ℤ L) {X : ℕ → Module.End ℤ L} (hScalar23 : HScalar23 ℓ X) {p : ℕ}
    (hp : p.Prime) (hpl : p ≠ ℓ) : ∃ y : Module.End ℤ L, X p - ((1 + p : ℕ) : ℤ) • 1 = (ℓ : ℤ) • y :=
  (redMat_eq_zero_iff ℓ b _).mp (redMat_eq_zero_of_forall ℓ b _ fun v => by
    obtain ⟨w, hw⟩ := hScalar23.scalar p hp hpl v
    exact ⟨w, by rw [LinearMap.sub_apply, LinearMap.smul_apply, Module.End.one_apply]; exact hw⟩)

end Lattice

section Field

variable {F : Type*} [Field F] {ι : Type*} [Fintype ι] [DecidableEq ι]

theorem aeval_restrict_apply_field {V : Type*} [AddCommGroup V] [Module F V] {f : Module.End F V} {W : Submodule F V}
    (hW : ∀ x ∈ W, f x ∈ W) (P : F[X]) (y : W) : ((aeval (f.restrict hW) P y : W) : V) = aeval f P (y : V) := by
  induction P using Polynomial.induction_on generalizing y with
  | C a => simp
  | add p q hp hq => simp only [map_add, LinearMap.add_apply, Submodule.coe_add, hp, hq]
  | monomial n a ih =>
    have e : (C a * X ^ (n + 1) : F[X]) = (C a * X ^ n) * X := by ring
    rw [e, map_mul, aeval_X, Module.End.mul_apply, ih, LinearMap.coe_restrict_apply, map_mul (aeval f) (C a * X ^ n) X,
      aeval_X, Module.End.mul_apply]

/-- **B9, the degree bound** ([C, prop:heckeZlevelone23](6)'s proof): a matrix of rank `r` over a field satisfies a monic
polynomial of degree `≤ r + 1`, namely `X · g` for `g` the characteristic polynomial of `Y` on its image. -/
theorem exists_monic_aeval_eq_zero (Y : Matrix ι ι F) :
    ∃ q : F[X], q.Monic ∧ q.natDegree ≤ Y.rank + 1 ∧ aeval Y q = 0 := by
  set f := Matrix.toLinAlgEquiv' Y
  set W := LinearMap.range f
  have hW : ∀ x ∈ W, f x ∈ W := fun x _ => LinearMap.mem_range_self f x
  set g := (f.restrict hW).charpoly
  have hg : g.Monic := LinearMap.charpoly_monic _
  have hgd : g.natDegree = Module.finrank F W := LinearMap.charpoly_natDegree _
  refine ⟨X * g, monic_X.mul hg, ?_, ?_⟩
  · rw [natDegree_mul X_ne_zero hg.ne_zero, natDegree_X, hgd, add_comm]
    have : Module.finrank F W = Y.rank := by
      rw [Matrix.rank]; rfl
    omega
  · have hf : aeval f (X * g) = 0 := by
      refine LinearMap.ext fun v => ?_
      rw [mul_comm, map_mul, aeval_X, Module.End.mul_apply]
      have := aeval_restrict_apply_field hW g ⟨f v, LinearMap.mem_range_self f v⟩
      rw [LinearMap.aeval_self_charpoly] at this
      simp only [LinearMap.zero_apply, ZeroMemClass.coe_zero] at this
      rw [← this]; rfl
    have key := Polynomial.aeval_algHom_apply (Matrix.toLinAlgEquiv' (R := F) (n := ι)).symm f (X * g)
    rw [show (Matrix.toLinAlgEquiv' (R := F) (n := ι)).symm f = Y from AlgEquiv.symm_apply_apply _ Y, hf, map_zero] at key
    exact key

/-- `𝔽[Y] ⊆ span(1, Y, …, Y^{k−1})` for a monic `q` of degree `k` with `q(Y) = 0`. -/
theorem adjoin_le_span (Y : Matrix ι ι F) {q : F[X]} (hq : q.Monic) (hq0 : aeval Y q = 0) (hq1 : 0 < q.natDegree) :
    Subalgebra.toSubmodule (Algebra.adjoin F {Y}) ≤
      Submodule.span F (Set.range fun i : Fin q.natDegree => Y ^ (i : ℕ)) := by
  intro m hm
  rw [Subalgebra.mem_toSubmodule, Algebra.adjoin_singleton_eq_range_aeval] at hm
  obtain ⟨P, rfl⟩ := hm
  have hdiv := Polynomial.modByMonic_add_div P q
  have : (aeval Y) P = aeval Y (P %ₘ q) := by
    conv_lhs => rw [← hdiv]
    simp [map_add, map_mul, hq0]
  change (aeval Y) P ∈ _
  rw [this]
  have hlt : (P %ₘ q).natDegree < q.natDegree ∨ P %ₘ q = 0 := by
    by_cases h0 : P %ₘ q = 0
    · exact Or.inr h0
    · left
      exact natDegree_modByMonic_lt P hq (by intro h1; rw [h1] at hq1; simp at hq1)
  rcases hlt with hlt | h0
  · rw [aeval_eq_sum_range' hlt]
    refine Submodule.sum_mem _ fun i hi => Submodule.smul_mem _ _ (Submodule.subset_span ⟨⟨i, Finset.mem_range.mp hi⟩, rfl⟩)
  · rw [h0, map_zero]; exact Submodule.zero_mem _

/-- **B9: `dim_𝔽 𝔽[Y] ≤ 3`** for a matrix of rank `≤ 2`. -/
theorem finrank_adjoin_le_three (Y : Matrix ι ι F) (hY : Y.rank ≤ 2) :
    Module.finrank F (Subalgebra.toSubmodule (Algebra.adjoin F {Y})) ≤ 3 := by
  obtain ⟨q, hq, hqd, hq0⟩ := exists_monic_aeval_eq_zero Y
  by_cases hq1 : 0 < q.natDegree
  · calc Module.finrank F (Subalgebra.toSubmodule (Algebra.adjoin F {Y}))
        ≤ Module.finrank F (Submodule.span F (Set.range fun i : Fin q.natDegree => Y ^ (i : ℕ))) :=
          Submodule.finrank_mono (adjoin_le_span Y hq hq0 hq1)
      _ ≤ Fintype.card (Fin q.natDegree) := finrank_range_le_card _
      _ ≤ 3 := by simp; omega
  · have h0 : q = 1 := hq.natDegree_eq_zero.mp (by omega)
    rw [h0, map_one] at hq0
    have : Subsingleton (Matrix ι ι F) := subsingleton_of_zero_eq_one hq0.symm
    have : Subsingleton (Subalgebra.toSubmodule (Algebra.adjoin F {Y})) := inferInstance
    rw [Module.finrank_zero_of_subsingleton]; omega

end Field

section C3

variable {L : Type*} [AddCommGroup L] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The reductions of `𝕋^fr = ℤ[T_p : p prime]` lie in `𝔽_ℓ[T̄_ℓ]` under `hScalar23`. -/
theorem redMat_mem_adjoin {ℓ : ℕ} (b : Module.Basis ι ℤ L) {X : ℕ → Module.End ℤ L} (hScalar23 : HScalar23 ℓ X)
    {t : Module.End ℤ L} (ht : t ∈ Algebra.adjoin ℤ (X '' {p | p.Prime})) :
    redMat ℓ b t ∈ Algebra.adjoin (ZMod ℓ) {redMat ℓ b (X ℓ)} := by
  induction ht using Algebra.adjoin_induction with
  | mem y hy =>
    obtain ⟨p, hp, rfl⟩ := hy
    by_cases hpl : p = ℓ
    swap
    · obtain ⟨z, hz⟩ := exists_eq_smul_of_scalar b hScalar23 hp hpl
      have e : X p = ((1 + p : ℕ) : ℤ) • 1 + (ℓ : ℤ) • z := by rw [← hz]; abel
      rw [e, map_add, map_zsmul, map_one, (redMat_eq_zero_iff ℓ b _).mpr ⟨z, rfl⟩, add_zero]
      exact Subalgebra.zsmul_mem _ (Subalgebra.one_mem _) _
    · subst hpl; exact Algebra.subset_adjoin rfl
  | algebraMap r =>
    have : redMat ℓ b (algebraMap ℤ (Module.End ℤ L) r) = r • 1 := by
      rw [Algebra.algebraMap_eq_smul_one, map_zsmul, map_one]
    rw [this]
    exact Subalgebra.zsmul_mem _ (Subalgebra.one_mem _) _
  | add y z _ _ hy hz => simpa only [map_add] using Subalgebra.add_mem _ hy hz
  | mul y z _ _ hy hz => simpa only [map_mul] using Subalgebra.mul_mem _ hy hz

/-- **B9's consequence: Thm. 4.1's (C3) cannot pass at `ℓ ∈ {2, 3}` when `d ≥ 4`.** Under `hScalar23` and `hTellRank`, no
`d ≥ 4` elements of `𝕋^fr = ℤ[T_p : p prime]` are linearly independent modulo `ℓ`. -/
theorem not_modIndep_levelOne23 {ℓ : ℕ} [Fact ℓ.Prime] (b : Module.Basis ι ℤ L) {X : ℕ → Module.End ℤ L}
    (hScalar23 : HScalar23 ℓ X) (hTellRank : HTellRank ℓ b X) {d : ℕ} (hd : 4 ≤ d)
    (t : Fin d → Module.End ℤ L) (ht : ∀ k, t k ∈ Algebra.adjoin ℤ (X '' {p | p.Prime})) :
    ¬ GL2.HeckeZ.Cert.ModIndep (ℓ : ℤ) t := by
  intro hind
  set A := Subalgebra.toSubmodule (Algebra.adjoin (ZMod ℓ) {redMat ℓ b (X ℓ)})
  have hmem : ∀ k, redMat ℓ b (t k) ∈ A := fun k => redMat_mem_adjoin b hScalar23 (ht k)
  set v : Fin d → A := fun k => ⟨redMat ℓ b (t k), hmem k⟩
  have hnot : ¬ LinearIndependent (ZMod ℓ) v := by
    intro hli
    have h1 := hli.fintype_card_le_finrank
    have h2 : Module.finrank (ZMod ℓ) A ≤ 3 := finrank_adjoin_le_three _ hTellRank.rank_le
    simp only [Fintype.card_fin] at h1
    omega
  obtain ⟨g, hg, i, hi⟩ := Fintype.not_linearIndependent_iff.mp hnot
  have hsum : redMat ℓ b (∑ k, ((g k).val : ℤ) • t k) = 0 := by
    have := congrArg Subtype.val hg
    simp only [Submodule.coe_sum, Submodule.coe_smul, ZeroMemClass.coe_zero, v] at this
    rw [map_sum, ← this]
    refine Finset.sum_congr rfl fun k _ => ?_
    rw [map_zsmul, ← Int.cast_smul_eq_zsmul (ZMod ℓ), Int.cast_natCast, ZMod.natCast_zmod_val]
  obtain ⟨y, hy⟩ := (redMat_eq_zero_iff ℓ b _).mp hsum
  have hdv := hind (fun k => ((g k).val : ℤ)) ⟨y, hy⟩ i
  apply hi
  have : (((g i).val : ℤ) : ZMod ℓ) = 0 := (ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mpr hdv
  simpa using this

end C3

end GL2.HeckeZ.Bock
