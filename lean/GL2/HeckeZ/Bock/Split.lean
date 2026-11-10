/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# HeckeZ 03 B7 (LB-4): splitting by a Bezout separation (L-HZ2, M-HZ3)

`docs/implementation/slot24/heckez/03-bockstein-tor.md` §3.1 (B7), §8.1 (LB-4); `07-lean-plan.md` §2.3 (LB-4), §5.6 (K6);
[C] `prop:heckeZsplit` (1.25.0-pre.3). Stated directly, **without `Ext¹`** (A3-N7).

B7 says: a short exact sequence `0 → S → M → M/S → 0` of modules over a commutative ring whose outer terms have disjoint
supports splits uniquely and equivariantly, and `Hom(M/S, S) = 0`. The separation certificate K6 (03 §3.1) proves the
disjointness by **one operator** `θ` and a Bezout identity `u χ̄ + v P̄ = 1` in `𝔽_ℓ[x]`, `χ̄` killing the quotient and `P̄`
nilpotent on `S`. This module proves B7 in exactly that form, over `ℤ`:

**`BezoutSep ℓ θ S F G U V W`**: `S` is `θ`-stable, `U F + V G = 1 + ℓ W` in `ℤ[X]`, `F(θ)` maps `M` into `S`, `G(θ)` is nilpotent
on `S`, and `S` is killed by a power of `ℓ`. Then
* `isUnit_res`: `F(θ)` is a unit on `S` (`U F = 1 − N` with `N` nilpotent on `S`);
* `hom_eq_zero`: every `f : M → M` with values in `S`, killing `S` and commuting with `θ` is `0` (`Hom_{ℤ[θ]}(M/S, S) = 0`);
* `isCompl_ker`: `M = S ⊕ ker F(θ)`; `ker F(θ)` is stable under everything that commutes with `θ` (`ker_stable`); and it is
  the **unique** `θ`-stable complement of `S` (`eq_ker_of_isCompl`).

The general support statement (an arbitrary disjoint pair of supports, through the idempotents of the Artinian image,
`IsArtinianRing.isNilpotent_nilradical` and `CompleteOrthogonalIdempotents.lift_of_isNilpotent_ker`, 07 F07-4) is **not**
formalised here; the Bezout form is what every K6 certificate supplies (a separation computed from one operator `T_p`, which
is a certificate for all of `𝕋` because the complement `ker F(T_p)` is stable under the whole commutative algebra).
-/

namespace GL2.HeckeZ.Bock

open Polynomial

variable {M : Type*} [AddCommGroup M]

/-- Commuting with `θ` passes to every polynomial in `θ`. -/
theorem commute_aeval {f θ : Module.End ℤ M} (h : Commute f θ) (P : ℤ[X]) : Commute f (aeval θ P) := by
  induction P using Polynomial.induction_on with
  | C a => simp only [aeval_C]; exact Algebra.commutes a f |>.symm
  | add p q hp hq => rw [map_add]; exact hp.add_right hq
  | monomial n a ih =>
    rw [pow_succ, ← mul_assoc, map_mul, aeval_X]
    exact ih.mul_right h

/-- Polynomials in one endomorphism commute. -/
theorem commute_aeval_aeval {N : Type*} [Ring N] [Algebra ℤ N] (θ : N) (P Q : ℤ[X]) :
    Commute (aeval θ P) (aeval θ Q) := by
  unfold Commute SemiconjBy; rw [← map_mul, ← map_mul, mul_comm]

/-- A `θ`-stable submodule is stable under every polynomial in `θ`. -/
theorem aeval_mem {θ : Module.End ℤ M} {S : Submodule ℤ M} (hS : ∀ x ∈ S, θ x ∈ S) (P : ℤ[X]) {x : M} (hx : x ∈ S) :
    aeval θ P x ∈ S := by
  induction P using Polynomial.induction_on generalizing x with
  | C a => simp only [aeval_C, Module.algebraMap_end_apply]; exact S.smul_mem a hx
  | add p q hp hq => rw [map_add, LinearMap.add_apply]; exact S.add_mem (hp hx) (hq hx)
  | monomial n a ih =>
    rw [pow_succ, ← mul_assoc, map_mul, aeval_X, Module.End.mul_apply]
    exact ih (hS x hx)

/-- `aeval` of the restriction to a stable submodule, computed in `M`. -/
theorem aeval_restrict_apply {θ : Module.End ℤ M} {S : Submodule ℤ M} (hS : ∀ x ∈ S, θ x ∈ S) (P : ℤ[X]) (y : S) :
    ((aeval (θ.restrict hS) P y : S) : M) = aeval θ P (y : M) := by
  induction P using Polynomial.induction_on generalizing y with
  | C a => simp
  | add p q hp hq => simp only [map_add, LinearMap.add_apply, Submodule.coe_add, hp, hq]
  | monomial n a ih =>
    have e : (C a * X ^ (n + 1) : ℤ[X]) = (C a * X ^ n) * X := by ring
    rw [e, map_mul, aeval_X, Module.End.mul_apply, ih, LinearMap.restrict_coe_apply, map_mul (aeval θ) (C a * X ^ n) X, aeval_X,
      Module.End.mul_apply]

/-- The torsion half of the Bezout separation of K6 (03 §3.1, A3-m4(a)), over `ℤ`: `G` (a lift of the product of the torsion
layer polynomials) is nilpotent on the `θ`-stable, `ℓ`-power-torsion `S`, and `U F + V G ≡ 1 (mod ℓ)`. -/
structure BezoutNil (ℓ : ℤ) (θ : Module.End ℤ M) (S : Submodule ℤ M) (F G U V W : ℤ[X]) : Prop where
  /-- `S` is `θ`-stable -/
  stable : ∀ x ∈ S, θ x ∈ S
  /-- the Bezout identity modulo `ℓ`, lifted: `U F + V G = 1 + ℓ W` -/
  bez : U * F + V * G = 1 + C ℓ * W
  /-- `G(θ)` is nilpotent on `S` -/
  hG : ∃ m : ℕ, ∀ x ∈ S, ((aeval θ G) ^ m) x = 0
  /-- `S` is killed by a power of `ℓ` -/
  hl : ∃ e : ℕ, ∀ x ∈ S, ℓ ^ e • x = 0

/-- The Bezout separation of K6: `BezoutNil` and `F(θ)` kills `M/S` (`F` a lift of the free characteristic polynomial). -/
structure BezoutSep (ℓ : ℤ) (θ : Module.End ℤ M) (S : Submodule ℤ M) (F G U V W : ℤ[X]) : Prop
    extends BezoutNil ℓ θ S F G U V W where
  /-- `F(θ)` kills `M/S` -/
  hF : ∀ x, aeval θ F x ∈ S

namespace BezoutNil

variable {ℓ : ℤ} {θ : Module.End ℤ M} {S : Submodule ℤ M} {F G U V W : ℤ[X]}

/-- **`F(θ)` is a unit on `S`.** -/
theorem isUnit_res (h : BezoutNil ℓ θ S F G U V W) : IsUnit (aeval (θ.restrict h.stable) F) := by
  set θS := θ.restrict h.stable
  have key : aeval θS U * aeval θS F = 1 - (aeval θS (V * G) - aeval θS (C ℓ * W)) := by
    have hb : aeval θS (U * F) + aeval θS (V * G) = 1 + aeval θS (C ℓ * W) := by
      rw [← map_add, h.bez, map_add, map_one]
    rw [← eq_sub_iff_add_eq] at hb
    rw [← map_mul, hb]; abel
  obtain ⟨m, hm⟩ := h.hG
  obtain ⟨e, he⟩ := h.hl
  have nG : IsNilpotent (aeval θS G) := by
    refine ⟨m, LinearMap.ext fun y => Subtype.ext ?_⟩
    rw [← map_pow, aeval_restrict_apply, map_pow]
    exact hm y y.2
  have nL : IsNilpotent (aeval θS (C ℓ)) := by
    refine ⟨e, LinearMap.ext fun y => Subtype.ext ?_⟩
    rw [← map_pow, aeval_restrict_apply, ← C_pow, aeval_C, Module.algebraMap_end_apply]
    simpa using he y y.2
  have n1 : IsNilpotent (aeval θS (V * G)) := by
    rw [map_mul]; exact (commute_aeval_aeval θS V G).isNilpotent_mul_left nG
  have n2 : IsNilpotent (aeval θS (C ℓ * W)) := by
    rw [map_mul]; exact (commute_aeval_aeval θS (C ℓ) W).isNilpotent_mul_right nL
  have nN : IsNilpotent (aeval θS (V * G) - aeval θS (C ℓ * W)) :=
    (commute_aeval_aeval θS (V * G) (C ℓ * W)).isNilpotent_sub n1 n2
  have hu : IsUnit (aeval θS U * aeval θS F) := by rw [key]; exact nN.isUnit_one_sub
  exact ((commute_aeval_aeval θS U F).isUnit_mul_iff.mp hu).2

/-- `F(θ)` is injective on `S`. -/
theorem eq_zero_of_mem (h : BezoutNil ℓ θ S F G U V W) {y : M} (hy : y ∈ S) (h0 : aeval θ F y = 0) : y = 0 := by
  obtain ⟨u, hu⟩ := h.isUnit_res
  have hinj : Function.Injective (aeval (θ.restrict h.stable) F) := by
    rw [← hu]; exact (Module.End.isUnit_iff _).mp u.isUnit |>.1
  have : (⟨y, hy⟩ : S) = 0 := hinj (Subtype.ext (by rw [aeval_restrict_apply, map_zero]; simpa using h0))
  simpa using congrArg Subtype.val this

end BezoutNil

namespace BezoutSep

variable {ℓ : ℤ} {θ : Module.End ℤ M} {S : Submodule ℤ M} {F G U V W : ℤ[X]}

/-- **`Hom_{ℤ[θ]}(M/S, S) = 0`** (B7, the `Hom` half; B11(3) for `J`): an endomorphism with values in `S`, killing `S` and
commuting with `θ` is zero. -/
theorem hom_eq_zero (h : BezoutSep ℓ θ S F G U V W) (f : Module.End ℤ M) (hf1 : ∀ x, f x ∈ S) (hf2 : ∀ x ∈ S, f x = 0)
    (hfθ : Commute f θ) : f = 0 := by
  ext x
  refine h.toBezoutNil.eq_zero_of_mem (hf1 x) ?_
  have := congrArg (· x) (commute_aeval hfθ F).eq
  simp only [Module.End.mul_apply] at this
  rw [← this, hf2 _ (h.hF x)]

/-- **`M = S ⊕ ker F(θ)`** (B7, the splitting). -/
theorem isCompl_ker (h : BezoutSep ℓ θ S F G U V W) : IsCompl S (LinearMap.ker (aeval θ F)) := by
  refine ⟨?_, ?_⟩
  · rw [Submodule.disjoint_def]
    intro y hy hk
    exact h.toBezoutNil.eq_zero_of_mem hy hk
  · rw [codisjoint_iff, eq_top_iff]
    intro x _
    obtain ⟨u, hu⟩ := h.toBezoutNil.isUnit_res
    set s : S := (↑u⁻¹ : Module.End ℤ S) ⟨aeval θ F x, h.hF x⟩
    have hs : aeval θ F (s : M) = aeval θ F x := by
      rw [← aeval_restrict_apply h.stable, ← hu]
      change (((↑u : Module.End ℤ S) ((↑u⁻¹ : Module.End ℤ S) ⟨aeval θ F x, h.hF x⟩)) : M) = _
      rw [← Module.End.mul_apply, Units.mul_inv, Module.End.one_apply]
    have hk : x - s ∈ LinearMap.ker (aeval θ F) := by
      rw [LinearMap.mem_ker, map_sub, hs, sub_self]
    have : x = (s : M) + (x - s) := by abel
    rw [this]
    exact Submodule.add_mem_sup s.2 hk

/-- `ker F(θ)` is stable under every endomorphism commuting with `θ` (so the splitting is `𝕋`-equivariant for any
commutative `𝕋 ∋ θ`: a separation computed from one operator is a certificate for all of `𝕋`). -/
theorem ker_stable (t : Module.End ℤ M) (ht : Commute t θ) {x : M} (hx : x ∈ LinearMap.ker (aeval θ F)) :
    t x ∈ LinearMap.ker (aeval θ F) := by
  rw [LinearMap.mem_ker] at hx ⊢
  have := congrArg (· x) (commute_aeval ht F).eq
  simp only [Module.End.mul_apply] at this
  rw [← this, hx, map_zero]

/-- **Uniqueness**: every `θ`-stable complement of `S` is `ker F(θ)`. -/
theorem eq_ker_of_isCompl (h : BezoutSep ℓ θ S F G U V W) {C' : Submodule ℤ M} (hC : IsCompl S C')
    (hθ : ∀ x ∈ C', θ x ∈ C') : C' = LinearMap.ker (aeval θ F) := by
  have hle : C' ≤ LinearMap.ker (aeval θ F) := by
    intro x hx
    have h1 : aeval θ F x ∈ C' := aeval_mem hθ F hx
    have h2 : aeval θ F x ∈ S := h.hF x
    have := hC.disjoint.le_bot (show aeval θ F x ∈ S ⊓ C' from ⟨h2, h1⟩)
    simpa using this
  have hmod := sup_inf_assoc_of_le S hle
  rw [hC.symm.codisjoint.eq_top, top_inf_eq, (h.isCompl_ker).disjoint.eq_bot, sup_bot_eq] at hmod
  exact hmod.symm

end BezoutSep

end GL2.HeckeZ.Bock
