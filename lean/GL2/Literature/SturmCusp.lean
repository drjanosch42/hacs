/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# L1: Sturm for the cuspidal Hecke algebra over `ℤ`, level one (`hSturm`; LC-11, slot 25)

`docs/implementation/slot24/heckez/07-lean-plan.md` §3 (the row `hSturm`), F07-7; 01 Prop. 1.5(2) (the `S_k` statement only).
The quotations are those of [GT] `lit:hz-sturm-integral` (read on the page images by the LIT agent, 2026-11-04) and
`lit:sturm` (`GL2.Literature.SturmLevelOne`, LIT-13):

* **Stein**, *Modular forms: a computational approach* (AMS GSM 79), §9.4.2, printed p. 175: "Theorem 9.23. Suppose `Γ` is a
  congruence subgroup that contains `Γ₁(N)` and let (9.4.5) `r = km/12 − (m−1)/N`, where `m = [SL₂(ℤ):Γ]`. Then the Hecke
  algebra `𝕋 = ℤ[…, T_n, …] ⊂ End(S_k(Γ))` is generated as a `ℤ`-module by the Hecke operators `T_n` for `n ≤ r`." Its proof:
  "there is a perfect pairing `S_k(N,R) ⊗_R 𝕋_R → R` given by `⟨f, T⟩ ↦ a₁(T(f))` (this is true for `R = ℤ`, hence for any `R`)."
* **Diamond–Im**, §12.4, printed p. 118, Prop. 12.4.13: "For every `A`, `S_k(Γ; A)` is isomorphic to `Hom_A(𝕋, A)`. This is
  proved in the case `A = ℤ` by showing that `(f, T) ↦ a₁(f|T)` is a perfect pairing" (the integral perfect pairing).
* **Sturm**, LNM 1240, Thm. 1, p. 276 (`lit:sturm`): the mod-`λ` vanishing above `k [Γ(1):Γ]/12`; with the pairing it is the
  `ℤ`-span statement (01 Prop. 1.5(2), P2a-8: the mod-`ℓ` form is needed for the `ℤ`-span).

At level one (`Γ = SL₂(ℤ)`, `m = N = 1`) the bound is `r = k/12`.

**The derived consequence stated here** (no single quotation states it; like `EichlerShimuraLevelOne`, a derived-consequence
structure over interface data). `L` stands for `H¹(SL₂(ℤ); Sym^n ℤ)/tors` and `X m` for `T_m` on it (`k = n + 2`); the
Eisenstein eigenvalue of `T_p` is `1 + p^{k−1}` (Serre GTM 7 §5.5 Prop. 13, `lit:serre-course-prop13`). Three steps:
1. Stein Thm. 9.23 at level one: `𝕋(S_k) = Σ_{n ≤ k/12} ℤ T_n`.
2. Every `T_n` is a polynomial over `ℤ` in the `T_p`, `p ≤ n` prime (Serre GTM 7 VII §5.3, Prop. 11, (72)–(73),
   `lit:serre-course-prop11`); so `𝕋(S_k) = ℤ[T_p : p ≤ k/12]` as an algebra. This is **weaker** than the module form of
   step 1 (which implies it), and it is the form used.
3. Eichler–Shimura (the sources of `lit:eichler-shimura-levelone`): `L_ℚ = Eis ⊕ cusp` with `cusp ≅ S_k ⊕ \bar S_k` carrying the
   systems of `S_k`, each twice; an element of the Hecke ring acting as `0` on `S_k` acts as `0` on `cusp`. The lattice
   `cuspLat X k = Σ_p (T_p − 1 − p^{k−1}) L` lies in `L ∩ cusp` (the `T_p − 1 − p^{k−1}` kill `Eis`) with finite index (every
   cuspidal system differs from the Eisenstein one at some prime: the systems of `M_k` are distinct, Serre §5.4 Thm. 7).
So every `t ∈ 𝕋^fr = ℤ[X m]` agrees on `cuspLat X k` with some element of `ℤ[X p : p prime, p ≤ k/12]`.

**What is not stated:** nothing about `M_k` ([E, cl:sturm](a)'s `M_k` sentence is wrong, erratum pending; 01 Prop. 1.5(1) is
a CONJ). The Eisenstein line itself is **not** asserted: if `1 + p^{k−1}` were no system of `L`, `cuspLat` would be all of
`L_ℚ` up to finite index and the structure would assert more (`𝕋^fr = ℤ[X p : p ≤ k/12]` up to finite index), which is false
for the real objects only if the Eisenstein system were absent — it is present (Eichler–Shimura). Sanity: the hypothesis is
satisfiable (`sturmCuspLevelOne_zero`).
-/

namespace GL2.Literature

/-- The Eisenstein eigenvalue of `T_p` at level one, weight `k`: `1 + p^{k−1}`. -/
def eisLam (k p : ℕ) : ℤ := 1 + (p : ℤ) ^ (k - 1)

/-- The **cuspidal lattice** (up to finite index): the `ℤ`-span of the images of `X p − (1 + p^{k−1})`, `p` prime. -/
def cuspLat {L : Type*} [AddCommGroup L] (X : ℕ → Module.End ℤ L) (k : ℕ) : Submodule ℤ L :=
  ⨆ p ∈ {p : ℕ | p.Prime}, LinearMap.range (X p - eisLam k p • (LinearMap.id : Module.End ℤ L))

open Classical in
/-- **Sturm for the cuspidal Hecke algebra over `ℤ`, level one** (`hSturm` of 07 §3; sources Stein Thm. 9.23, Diamond–Im
Prop. 12.4.13, Sturm LNM 1240 Thm. 1, with Serre VII §5.3 Prop. 11 and the Eichler–Shimura sources of
`lit:eichler-shimura-levelone`; a derived consequence, see the module docstring). Every element of `𝕋^fr = ℤ[X m]` agrees on
the cuspidal lattice with an element of `ℤ[X p : p prime, p ≤ k/12]`. Interface data: `L`, `X`, the weight `k`. -/
structure SturmCuspLevelOne {L : Type*} [AddCommGroup L] (X : ℕ → Module.End ℤ L) (k : ℕ) : Prop where
  span : ∀ t ∈ Algebra.adjoin ℤ (Set.range X), ∃ s ∈ Algebra.adjoin ℤ (X '' {p : ℕ | p.Prime ∧ p ≤ k / 12}),
    ∀ v ∈ cuspLat X k, t v = s v

/-- Sanity: the hypothesis is satisfiable (by the zero family on any `L`; `𝕋^fr = ℤ` then). -/
theorem sturmCuspLevelOne_zero {L : Type*} [AddCommGroup L] (k : ℕ) :
    SturmCuspLevelOne (fun _ : ℕ => (0 : Module.End ℤ L)) k := by
  refine ⟨fun t ht => ⟨t, ?_, fun _ _ => rfl⟩⟩
  have h0 : Algebra.adjoin ℤ (Set.range fun _ : ℕ => (0 : Module.End ℤ L)) = ⊥ := by
    refine le_bot_iff.mp (Algebra.adjoin_le ?_)
    rintro _ ⟨_, rfl⟩; exact Subalgebra.zero_mem _
  rw [h0] at ht
  exact (bot_le : (⊥ : Subalgebra ℤ (Module.End ℤ L)) ≤ _) ht

end GL2.Literature
