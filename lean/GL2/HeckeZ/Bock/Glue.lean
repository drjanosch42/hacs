/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.HeckeZ.Bock.Split

/-!
# HeckeZ 03 B11–B14 (LB-6): the glue `J` (L-HZ2, M-HZ3)

`docs/implementation/slot24/heckez/03-bockstein-tor.md` §4 (B11–B14), §8.1 (LB-6); `07-lean-plan.md` §2.3 (LB-6), §5.7 (K7);
[C] `prop:heckeZglue` (1.25.0-pre.3). Elementary throughout; **CB-3 is not a hypothesis** (A3-M2): nothing here says "`J` is
computable"; the statements about computed data are the sandwich and the vanishing criteria.

Setting: an `R`-module `H` (`H¹`, or `H¹ ⊗ ℤ_ℓ`, or `H¹/ℓ^K`), a submodule `S` (the torsion, or its `ℓ`-part), and sets of
endomorphisms (`𝕋`, `𝕋' = ℤ[P]`). With `I_fr = {t : t H ⊆ S}` (`Ifr`) and `I_tors = {t : t S = 0}` (`Itors`):
* `Jset T S = T ∩ I_fr ∩ I_tors` — the glue `J` of [C] `prop:heckeZglue`(1);
* `HomP P S` — the endomorphisms with values in `S`, killing `S`, commuting with `P`: `Hom_{ℤ[P]}(H/S, S)`.

Results:
* B11(1) `mem_Ifr_iff_torsion`: for `S` the torsion of a `ℤ`-module with finite exponent, `I_fr` is the additive torsion of
  `End H`, so `J = 𝕋_tors ∩ I_tors`;
* B11(2) `Jset_subset_HomP`: `t ↦ φ_t` (here `φ_t = t`, factored through `H/S`) embeds `J` into `Hom_{𝕋'}(H/S, S)` for every
  `𝕋' ⊆ 𝕋` when `𝕋` is commutative — **the upper bound of the sandwich** (B13); `Jset_eq_zero_of_HomP` (status `zero`);
* B12(3) `Jset_eq_zero_of_compl`: a `𝕋'`-stable complement of `S` forces `J ∩ 𝕋' = 0` — **the lower bound** (split ⇒ `J = 0`);
  the converse fails: `Counter.*`, `H = ℤ² ⊕ 𝔽₂`, `x(a, b, c) = (0, 2b, ā)` (over `ℤ` at `ℓ = 2`, A3-N7);
* B12(1) `isCompl_comap`: a stable complement of `S̄` in `H/N` (`N = ℓ^K H`, `S ∩ N = 0`) pulls back to a stable complement
  of `S` in `H` (split at level `K` ⇒ split);
* B13 `eq_zero_of_mem_Ifr_of_mod`: an element of `I_fr` acting as `0` on `H/N` with `S ∩ N = 0` is `0` (`J_ℓ → A_K` injective);
* B14 `not_bezoutSep_of_nonsplit`: if `S` has no `𝕋'`-stable complement, no `θ ∈ 𝕋'` separates by a Bezout identity (a
  non-split verdict certifies a common point; the support form is not formalised, LB-4);
* `Jset_subset_of_ellPart`: the `ℓ`-primary part of `J` (relative to the full torsion) lies in `J` relative to `tors_ℓ`.
-/

namespace GL2.HeckeZ.Bock

open Polynomial

section Defs

variable {R : Type*} [CommRing R] {H : Type*} [AddCommGroup H] [Module R H]

/-- `I_fr`: the endomorphisms with values in `S` (they kill `H/S`). -/
def Ifr (S : Submodule R H) : Set (Module.End R H) := {t | ∀ x, t x ∈ S}

/-- `I_tors`: the endomorphisms killing `S`. -/
def Itors (S : Submodule R H) : Set (Module.End R H) := {t | ∀ x ∈ S, t x = 0}

/-- The glue `J = T ∩ I_fr ∩ I_tors` ([C] `prop:heckeZglue`(1)). -/
def Jset (T : Set (Module.End R H)) (S : Submodule R H) : Set (Module.End R H) := T ∩ Ifr S ∩ Itors S

/-- `Hom_{ℤ[P]}(H/S, S)`, as endomorphisms of `H` with values in `S`, killing `S`, commuting with every element of `P`. -/
def HomP (P : Set (Module.End R H)) (S : Submodule R H) : Set (Module.End R H) :=
  {f | f ∈ Ifr S ∧ f ∈ Itors S ∧ ∀ x ∈ P, Commute f x}

/-- **B11(2), the upper bound.** If every element of `T` commutes with `P` (e.g. `T` commutative and `P ⊆ T`), then
`J ⊆ Hom_{ℤ[P]}(H/S, S)`. -/
theorem Jset_subset_HomP {T P : Set (Module.End R H)} {S : Submodule R H} (hc : ∀ t ∈ T, ∀ x ∈ P, Commute t x) :
    Jset T S ⊆ HomP P S := fun _ ht => ⟨ht.1.2, ht.2, hc _ ht.1.1⟩

/-- **Status `zero`**: `Hom_{ℤ[P]}(H/S, S) = 0` forces `J = 0`. -/
theorem Jset_eq_zero_of_HomP {T P : Set (Module.End R H)} {S : Submodule R H}
    (hc : ∀ t ∈ T, ∀ x ∈ P, Commute t x) (h0 : ∀ f ∈ HomP P S, f = 0) : ∀ t ∈ Jset T S, t = 0 :=
  fun t ht => h0 t (Jset_subset_HomP hc ht)

/-- Every element of `adjoin R (range X)` commutes with every `X i`, when the `X i` commute pairwise. -/
theorem commute_of_mem_adjoin {ι : Type*} {X : ι → Module.End R H} (hX : ∀ i j, Commute (X i) (X j))
    {t : Module.End R H} (ht : t ∈ Algebra.adjoin R (Set.range X)) (i : ι) : Commute t (X i) := by
  refine (Algebra.commute_of_mem_adjoin_of_forall_mem_commute ht ?_).symm
  rintro _ ⟨j, rfl⟩
  exact hX i j

/-- **B12(3), the lower bound**: a `T`-stable complement of `S` forces `J ∩ T = 0` (split ⇒ `J = 0`). -/
theorem Jset_eq_zero_of_compl {T : Set (Module.End R H)} {S C : Submodule R H} (hC : IsCompl S C)
    (hst : ∀ t ∈ T, ∀ x ∈ C, t x ∈ C) : ∀ t ∈ Jset T S, t = 0 := by
  intro t ht
  ext x
  have hx : x ∈ S ⊔ C := by rw [hC.codisjoint.eq_top]; exact Submodule.mem_top
  obtain ⟨s, hs, c, hc, rfl⟩ := Submodule.mem_sup.mp hx
  have h1 : t c ∈ S ⊓ C := ⟨ht.1.2 c, hst t ht.1.1 c hc⟩
  rw [hC.disjoint.eq_bot, Submodule.mem_bot] at h1
  simp [map_add, ht.2 s hs, h1]

/-- **B13, injectivity of the reduction**: an element of `I_fr` acting as `0` on `H/N` (values in `N`) is `0` when
`S ∩ N = 0` (for `N = ℓ^K H`, `K ≥ e_ℓ`: `J_ℓ → End(H/ℓ^K)` is injective). -/
theorem eq_zero_of_mem_Ifr_of_mod {S N : Submodule R H} (hSN : S ⊓ N = ⊥) {t : Module.End R H} (ht : t ∈ Ifr S)
    (hN : ∀ x, t x ∈ N) : t = 0 := by
  ext x
  have : t x ∈ S ⊓ N := ⟨ht x, hN x⟩
  rw [hSN, Submodule.mem_bot] at this
  simpa using this

/-- **B12(1)**: a complement of `S̄` in `H/N` pulls back to a complement of `S` in `H`, if `S ∩ N = 0`; it is stable under
every `t` whose reduction preserves the complement (split at level `K` ⇒ split). -/
theorem isCompl_comap {S N : Submodule R H} (hSN : S ⊓ N = ⊥) {Cb : Submodule R (H ⧸ N)}
    (hC : IsCompl (S.map N.mkQ) Cb) : IsCompl S (Cb.comap N.mkQ) := by
  refine ⟨?_, ?_⟩
  · rw [Submodule.disjoint_def]
    intro x hxS hxC
    have h1 : N.mkQ x ∈ S.map N.mkQ ⊓ Cb := ⟨Submodule.mem_map_of_mem hxS, hxC⟩
    rw [hC.disjoint.eq_bot, Submodule.mem_bot, Submodule.mkQ_apply, Submodule.Quotient.mk_eq_zero] at h1
    have : x ∈ S ⊓ N := ⟨hxS, h1⟩
    rwa [hSN, Submodule.mem_bot] at this
  · rw [codisjoint_iff, eq_top_iff]
    intro x _
    have hx : N.mkQ x ∈ S.map N.mkQ ⊔ Cb := by rw [hC.codisjoint.eq_top]; exact Submodule.mem_top
    obtain ⟨sb, hsb, cb, hcb, he⟩ := Submodule.mem_sup.mp hx
    obtain ⟨s, hs, rfl⟩ := Submodule.mem_map.mp hsb
    have hc : x - s ∈ Cb.comap N.mkQ := by
      rw [Submodule.mem_comap, map_sub, ← he]; simpa using hcb
    have : x = s + (x - s) := by abel
    rw [this]
    exact Submodule.add_mem_sup hs hc

/-- Stability of the pulled-back complement. -/
theorem comap_stable {N : Submodule R H} {Cb : Submodule R (H ⧸ N)} {t : Module.End R H}
    (ht : ∀ x, N.mkQ x ∈ Cb → N.mkQ (t x) ∈ Cb) : ∀ x ∈ Cb.comap N.mkQ, t x ∈ Cb.comap N.mkQ :=
  fun x hx => ht x hx

end Defs

section OverZ

variable {H : Type*} [AddCommGroup H]

/-- **B11(1)**: if `S` is the torsion submodule and has a finite exponent `N`, then `t ∈ I_fr` iff `t` is torsion in
`End_ℤ H`; so `I_fr ∩ 𝕋 = 𝕋_tors` and `J = {t ∈ 𝕋_tors : t S = 0}`. -/
theorem mem_Ifr_iff_torsion {S : Submodule ℤ H} (hS : ∀ x, x ∈ S ↔ ∃ n : ℤ, n ≠ 0 ∧ n • x = 0)
    (hexp : ∃ N : ℤ, N ≠ 0 ∧ ∀ x ∈ S, N • x = 0) (t : Module.End ℤ H) :
    t ∈ Ifr S ↔ ∃ n : ℤ, n ≠ 0 ∧ n • t = 0 := by
  constructor
  · intro ht
    obtain ⟨N, hN, hNS⟩ := hexp
    exact ⟨N, hN, LinearMap.ext fun x => by simpa using hNS _ (ht x)⟩
  · rintro ⟨n, hn, hnt⟩ x
    exact (hS _).mpr ⟨n, hn, by simpa using congrArg (· x) hnt⟩

/-- **B14 (Bezout form)**: if `S` has no complement stable under the commutative `T'`, then no `θ ∈ T'` separates `M/S`
from `S` by a Bezout identity (a non-split verdict certifies that the free and torsion systems meet). -/
theorem not_bezoutSep_of_nonsplit {S : Submodule ℤ H} {T' : Set (Module.End ℤ H)}
    (hcomm : ∀ a ∈ T', ∀ b ∈ T', Commute a b)
    (hns : ¬ ∃ C : Submodule ℤ H, IsCompl S C ∧ ∀ t ∈ T', ∀ x ∈ C, t x ∈ C) :
    ∀ θ ∈ T', ∀ (ℓ : ℤ) (F G U V W : ℤ[X]), ¬ BezoutSep ℓ θ S F G U V W := by
  intro θ hθ ℓ F G U V W h
  exact hns ⟨_, h.isCompl_ker, fun t ht x hx => BezoutSep.ker_stable t (hcomm t ht θ hθ) hx⟩

/-- The `ℓ`-primary part of `J` (relative to a submodule `tors`) lies in `J` relative to `Sℓ ⊆ tors`, if `Sℓ` contains every
element killed by a power of `ℓ`. -/
theorem Jset_subset_of_ellPart {T : Set (Module.End ℤ H)} {tors Sℓ : Submodule ℤ H} (hle : Sℓ ≤ tors) {ℓ : ℤ}
    (hSℓ : ∀ x (j : ℕ), ℓ ^ j • x = 0 → x ∈ Sℓ) {t : Module.End ℤ H} (ht : t ∈ Jset T tors) {k : ℕ}
    (hk : ℓ ^ k • t = 0) : t ∈ Jset T Sℓ := by
  refine ⟨⟨ht.1.1, fun x => hSℓ (t x) k ?_⟩, fun x hx => ht.2 x (hle hx)⟩
  have := LinearMap.congr_fun hk x
  rwa [LinearMap.smul_apply, LinearMap.zero_apply] at this

end OverZ

/-! ### The counterexample of B12(3): split ⇏ `J = 0`'s converse, over `ℤ` at `ℓ = 2` -/

namespace Counter

/-- `H = ℤ² ⊕ 𝔽₂`. -/
abbrev H := (ℤ × ℤ) × ZMod 2

/-- `x(a, b, c) = (0, 2b, ā)`. -/
def x : Module.End ℤ H where
  toFun v := ((0, 2 * v.1.2), (v.1.1 : ZMod 2))
  map_add' v w := by ext <;> simp; ring
  map_smul' n v := by ext <;> simp; ring

/-- The torsion `0 ⊕ 𝔽₂`. -/
def tors : Submodule ℤ H where
  carrier := {v | v.1 = 0}
  add_mem' ha hb := by simp_all
  zero_mem' := rfl
  smul_mem' n v hv := by simp_all

theorem x_mul_sub : x * (x - 2 • 1) = 0 := by
  have h2 : (2 : ZMod 2) = 0 := rfl
  ext v <;> simp [x, h2]

/-- Every element of `ℤ[x]` is `a + b x`. -/
theorem mem_adjoin {t : Module.End ℤ H} (ht : t ∈ Algebra.adjoin ℤ {x}) : ∃ a b : ℤ, t = a • 1 + b • x := by
  have hxx : x * x = (2 : ℤ) • x := by
    have := x_mul_sub
    rw [mul_sub, mul_smul_comm, mul_one, sub_eq_zero] at this
    exact this
  induction ht using Algebra.adjoin_induction with
  | mem y hy => rw [Set.mem_singleton_iff.mp hy]; exact ⟨0, 1, by simp⟩
  | algebraMap r => exact ⟨r, 0, by simp⟩
  | add y z _ _ hy hz =>
    obtain ⟨a, b, rfl⟩ := hy; obtain ⟨c, d, rfl⟩ := hz
    exact ⟨a + c, b + d, by rw [add_smul, add_smul]; abel⟩
  | mul y z _ _ hy hz =>
    obtain ⟨a, b, rfl⟩ := hy; obtain ⟨c, d, rfl⟩ := hz
    refine ⟨a * c, a * d + b * c + 2 * b * d, ?_⟩
    simp only [add_mul, mul_add, smul_mul_smul_comm, one_mul, mul_one, hxx, smul_smul, add_smul]
    module

/-- **`J = 0`** for `ℤ[x]` (it acts faithfully on `L = ℤ²` by `diag(0, 2)`). -/
theorem J_eq_zero : ∀ t ∈ Jset (Algebra.adjoin ℤ {x} : Set (Module.End ℤ H)) tors, t = 0 := by
  rintro t ⟨⟨ht, hfr⟩, -⟩
  obtain ⟨a, b, rfl⟩ := mem_adjoin ht
  have h1 := hfr ((1, 0), 0)
  have h2 := hfr ((0, 1), 0)
  simp [tors, x, Prod.ext_iff] at h1 h2
  have ha : a = 0 := h1
  subst ha
  have hb : b = 0 := by omega
  subst hb
  simp

/-- **Non-split**: no complement of `tors` is `x`-stable. -/
theorem nonsplit : ¬ ∃ C : Submodule ℤ H, IsCompl tors C ∧ ∀ v ∈ C, x v ∈ C := by
  rintro ⟨C, hC, hx⟩
  have hv : ((1, 0), (0 : ZMod 2)) ∈ tors ⊔ C := by rw [hC.codisjoint.eq_top]; exact Submodule.mem_top
  obtain ⟨s, hs, c, hc, he⟩ := Submodule.mem_sup.mp hv
  have hs1 : s.1 = 0 := hs
  have hc1 : c.1 = (1, 0) := by
    have := congrArg Prod.fst he; simp [hs1] at this; exact this
  have hxc : x c ∈ tors ⊓ C := by
    refine ⟨?_, hx c hc⟩
    show (x c).1 = 0
    simp [x, hc1]
  rw [hC.disjoint.eq_bot, Submodule.mem_bot] at hxc
  have := congrArg Prod.snd hxc
  simp [x, hc1] at this

end Counter

end GL2.HeckeZ.Bock
