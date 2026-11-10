/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# HeckeZ 03 B4(2) (LB-3): the filtration and its Hecke algebra (L-HZ2, M-HZ3)

`docs/implementation/slot24/heckez/03-bockstein-tor.md` §2.4 (B4(2)), §8.1 (LB-3); `07-lean-plan.md` §2.3 (LB-3);
[C] `prop:heckeZuct`(4).

For a filtration `0 = F₀ ≤ F₁ ≤ … ≤ F_m = M` of an `R`-module (B4: `0 ⊆ tors/ℓ^K ⊆ H^q/ℓ^K ⊆ H^q(Γ; V/ℓ^K)`, `m = 3`):
* `pow_eq_zero_of_filtration`: an endomorphism acting by `0` on every graded piece (`a F_{i+1} ⊆ F_i`) has `a^m = 0`; more
  generally (`prod_eq_zero_of_filtration`) any product of `m` such endomorphisms is `0`, so the kernel of
  `A → ∏ End(gr_i)` is nilpotent of index `≤ m`;
* `kerGr`, `kerGr_le_prime`: for a commutative ring `A` acting faithfully by `ρ` and preserving the filtration, the kernel of
  `A → ∏ End(gr_i)` is an ideal contained in every prime ideal: **`Spec A` is the union of the spectra of the graded pieces**
  (nilpotent ideals do not change the spectrum). The census algebra at `ℓ ∈ {2, 3}` can therefore carry maximal ideals of
  `𝕋^{(2)}` that are not maximal ideals of `𝕋^{(1)}` (B4(2)'s caution).
-/

namespace GL2.HeckeZ.Bock

variable {R : Type*} [CommRing R] {M : Type*} [AddCommGroup M] [Module R M]

/-- **B4(2)**: a product of `m` endomorphisms, each acting by `0` on every graded piece of a length-`m` filtration, is `0`. -/
theorem prod_eq_zero_of_filtration (F : ℕ → Submodule R M) {m : ℕ} (h0 : F 0 = ⊥) (htop : F m = ⊤)
    (a : Fin m → Module.End R M) (ha : ∀ k, ∀ i, ∀ x ∈ F (i + 1), a k x ∈ F i) :
    (List.ofFn a).prod = 0 := by
  -- the product of the last `j` factors maps `F m` into `F (m - j)`
  have key : ∀ j ≤ m, ∀ x, ((List.ofFn a).drop (m - j)).prod x ∈ F (m - j) := by
    intro j hj
    induction j with
    | zero =>
      intro x
      simp only [Nat.sub_zero]
      rw [List.drop_eq_nil_of_le (by simp)]
      simp [htop]
    | succ j ih =>
      intro x
      have hlt : m - (j + 1) < (List.ofFn a).length := by simp; omega
      rw [← List.cons_getElem_drop_succ (h := hlt)]
      have e : m - (j + 1) + 1 = m - j := by omega
      rw [List.prod_cons, Module.End.mul_apply, e]
      have := ih (by omega) x
      rw [List.getElem_ofFn]
      have hi := ha ⟨m - (j + 1), by omega⟩ (m - (j + 1)) _ (by rw [e]; exact this)
      exact hi
  ext x
  have := key m le_rfl x
  simp only [Nat.sub_self, List.drop_zero, h0, Submodule.mem_bot] at this
  simpa using this

/-- `a^m = 0` for one endomorphism acting by `0` on the graded pieces. -/
theorem pow_eq_zero_of_filtration (F : ℕ → Submodule R M) {m : ℕ} (h0 : F 0 = ⊥) (htop : F m = ⊤)
    (a : Module.End R M) (ha : ∀ i, ∀ x ∈ F (i + 1), a x ∈ F i) : a ^ m = 0 := by
  have := prod_eq_zero_of_filtration F h0 htop (fun _ => a) (fun _ => ha)
  simpa [List.ofFn_const, List.prod_replicate] using this

/-- The kernel of `A → ∏_i End(F_{i+1}/F_i)` for a commutative ring `A` acting by `ρ` and preserving the filtration. -/
def kerGr {A : Type*} [CommRing A] (ρ : A →+* Module.End R M) (F : ℕ → Submodule R M)
    (hρ : ∀ b i, ∀ x ∈ F i, ρ b x ∈ F i) : Ideal A where
  carrier := {a | ∀ i, ∀ x ∈ F (i + 1), ρ a x ∈ F i}
  add_mem' ha hb i x hx := by rw [map_add, LinearMap.add_apply]; exact (F i).add_mem (ha i x hx) (hb i x hx)
  zero_mem' i x _ := by simp
  smul_mem' b a ha i x hx := by
    rw [smul_eq_mul, map_mul, Module.End.mul_apply]; exact hρ b i _ (ha i x hx)

/-- **Spectrum equality (B4(2))**: if `ρ` is faithful, `kerGr` lies in every prime ideal of `A`. -/
theorem kerGr_le_prime {A : Type*} [CommRing A] (ρ : A →+* Module.End R M) (hinj : Function.Injective ρ)
    (F : ℕ → Submodule R M) {m : ℕ} (h0 : F 0 = ⊥) (htop : F m = ⊤) (hρ : ∀ b i, ∀ x ∈ F i, ρ b x ∈ F i)
    (P : Ideal A) [P.IsPrime] : kerGr ρ F hρ ≤ P := by
  intro a ha
  have hpow : a ^ m = 0 := hinj (by rw [map_pow, map_zero]; exact pow_eq_zero_of_filtration F h0 htop (ρ a) ha)
  exact ‹P.IsPrime›.mem_of_pow_mem m (by rw [hpow]; exact P.zero_mem)

end GL2.HeckeZ.Bock
