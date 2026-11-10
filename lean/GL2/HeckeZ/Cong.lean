/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# HeckeZ LC-9 at `r = 1`: the congruence module of one rational Eisenstein line (L-HZ0, M-HZ1)

`docs/implementation/slot24/heckez/01-math-core.md` Def. 5.1, Props. 5.2(1), 5.3, 5.4 (`r = 1`); `07-lean-plan.md` §2.2
LC-9, §7. Everything here is over `ℤ`: `L` is a torsion-free `ℤ`-module, `T ⊆ End_ℤ L` a subalgebra (in the application
`T = 𝕋^fr = ℤ[T_n]`), and the data of an **Eisenstein split** (`EisSplit`) in the *two-system* case of the T1 cell:

* two ring maps `λ, μ : T → ℤ` (the rational Eisenstein system and the one other, rational, system);
* an element `θ ∈ T` separating them (`λ θ ≠ μ θ`), with `P := θ − μ(θ)` and `Q := θ − λ(θ)` satisfying `Q ∘ P = 0`
  (`P L ⊆ L_S := ker Q`, the Eisenstein line, and `Q L ⊆ L_{S'} := ker P`);
* every `t ∈ T` acts on `ker Q` by `λ t` and on `ker P` by `μ t` (01 Prop. 5.2(1): `t` acts on `(L_ℚ)_S` by its
  eigenvalue).

**The key identity** (`key`): `(λθ − μθ)·(t − μ t) = (λ t − μ t)·P` on `L` — 01 Prop. 5.4's `π_S = c(λ)⁻¹ c(T_p)`
cleared of denominators, with `c = x − μθ`. **The lower bound** (`dvd_of_read`, Prop. 5.4 read at a root, `r = 1`): if
`ℓ^a ∣ λθ − μθ` and some coordinate of some `P v` has `ℓ`-valuation exactly `s`, then `ℓ^{a−s} ∣ λ t − μ t` for **every**
`t ∈ T` — no generation statement is used (Prop. 5.2(1)). **The upper bound** (`mem_congIdeal_sub`): `t − μ(t) ∈ T` has
`μ = 0`, `λ = λ t − μ t` (Cayley–Hamilton on the `S'`-part, Prop. 5.3). **The fibre product** (`fibre_iff`,
`pair_injective`): with both bounds at `g`, the image of `T` in `ℤ × ℤ` is `{(a, b) : a ≡ b mod g}` and `T` embeds,
i.e. `T ≅ ℤ ×_{ℤ/g} ℤ` and the algebra congruence module `C_T = T/(I_S + I_{S'})` is `ℤ/g` ([U, lem:u-fibre] with `A = ℤ`).
-/

namespace GL2.HeckeZ

variable {L : Type*} [AddCommGroup L] {T : Subalgebra ℤ (Module.End ℤ L)}

/-- An **Eisenstein split** of `T` (two rational systems, `r = 1`; 01 Def. 5.1, Prop. 5.3). -/
structure EisSplit (T : Subalgebra ℤ (Module.End ℤ L)) where
  /-- the Eisenstein system `λ` -/
  lam : T →+* ℤ
  /-- the other (cuspidal) system `μ` -/
  mu : T →+* ℤ
  /-- the separating element `θ` (in T1: `T₂`) -/
  θ : T
  hD : lam θ ≠ mu θ
  /-- `Q ∘ P = 0`, `P = θ − μ θ`, `Q = θ − λ θ` -/
  hQP : ∀ v : L, (θ : Module.End ℤ L) ((θ : Module.End ℤ L) v - mu θ • v) - lam θ • ((θ : Module.End ℤ L) v - mu θ • v) = 0
  /-- every `t` acts on `ker Q` (the Eisenstein line) by `λ t` -/
  hA : ∀ (t : T) (a : L), (θ : Module.End ℤ L) a - lam θ • a = 0 → (t : Module.End ℤ L) a = lam t • a
  /-- every `t` acts on `ker P` (the cuspidal part) by `μ t` -/
  hB : ∀ (t : T) (b : L), (θ : Module.End ℤ L) b - mu θ • b = 0 → (t : Module.End ℤ L) b = mu t • b

namespace EisSplit

variable (E : EisSplit T)

/-- `P v = θ v − μ(θ) v` (01's `c(T_p)` with `c = x − μ_p`). -/
def P (v : L) : L := (E.θ : Module.End ℤ L) v - E.mu E.θ • v

/-- `Q v = θ v − λ(θ) v`. -/
def Q (v : L) : L := (E.θ : Module.End ℤ L) v - E.lam E.θ • v

lemma P_sub_Q (v : L) : E.P v - E.Q v = (E.lam E.θ - E.mu E.θ) • v := by
  simp only [P, Q, sub_smul]; abel

lemma Q_P (v : L) : E.Q (E.P v) = 0 := E.hQP v

lemma P_Q (v : L) : E.P (E.Q v) = 0 := by
  have : E.P (E.Q v) = E.Q (E.P v) := by
    simp only [P, Q, map_sub, map_zsmul, smul_sub, smul_smul, mul_comm]; abel
  rw [this, E.Q_P]

lemma act_P (t : T) (v : L) : (t : Module.End ℤ L) (E.P v) = E.lam t • E.P v := E.hA t _ (E.Q_P v)

lemma act_Q (t : T) (v : L) : (t : Module.End ℤ L) (E.Q v) = E.mu t • E.Q v := E.hB t _ (E.P_Q v)

/-- **The key identity**: `(λθ − μθ)·(t v − μ(t) v) = (λ t − μ t)·P v`. -/
theorem key (t : T) (v : L) :
    (E.lam E.θ - E.mu E.θ) • ((t : Module.End ℤ L) v - E.mu t • v) = (E.lam t - E.mu t) • E.P v := by
  have e : (E.lam E.θ - E.mu E.θ) • ((t : Module.End ℤ L) v - E.mu t • v) =
      (t : Module.End ℤ L) ((E.lam E.θ - E.mu E.θ) • v) - E.mu t • ((E.lam E.θ - E.mu E.θ) • v) := by
    rw [map_zsmul, smul_sub, smul_comm]
  rw [e, ← E.P_sub_Q, map_sub, E.act_P, E.act_Q, smul_sub, sub_smul]; abel

/-- **The lower bound** (01 Prop. 5.4 at `r = 1`, read at a root; Prop. 5.2(1)): if `ℓ^a ∣ λθ − μθ` and a coordinate of
some `P v` (in some basis) has `ℓ`-valuation exactly `s`, then `ℓ^{a−s} ∣ λ t − μ t` for every `t ∈ T`. -/
theorem dvd_of_read {ι : Type*} (b : Module.Basis ι ℤ L) (ℓ : ℕ) (hℓ : ℓ.Prime) (a s : ℕ)
    (hD : (ℓ : ℤ) ^ a ∣ E.lam E.θ - E.mu E.θ) (v : L) (i : ι)
    (hs : (ℓ : ℤ) ^ s ∣ b.repr (E.P v) i) (hs1 : ¬ (ℓ : ℤ) ^ (s + 1) ∣ b.repr (E.P v) i) (t : T) :
    (ℓ : ℤ) ^ (a - s) ∣ E.lam t - E.mu t := by
  have hk : (E.lam E.θ - E.mu E.θ) * b.repr ((t : Module.End ℤ L) v - E.mu t • v) i =
      (E.lam t - E.mu t) * b.repr (E.P v) i := by
    have := congrArg (fun w => b.repr w i) (E.key t v)
    simpa only [LinearEquiv.map_smul, Finsupp.smul_apply, smul_eq_mul] using this
  have h0 : (ℓ : ℤ) ^ a ∣ (E.lam t - E.mu t) * b.repr (E.P v) i := by
    rw [← hk]; exact hD.mul_right _
  obtain ⟨y, hy⟩ := hs
  have hy' : ¬ (ℓ : ℤ) ∣ y := by
    rintro ⟨z, rfl⟩; exact hs1 ⟨z, by rw [hy, pow_succ]; ring⟩
  rcases le_or_gt a s with has | has
  · rw [Nat.sub_eq_zero_of_le has, pow_zero]; exact one_dvd _
  have h1 : (ℓ : ℤ) ^ (a - s) * (ℓ : ℤ) ^ s ∣ ((E.lam t - E.mu t) * y) * (ℓ : ℤ) ^ s := by
    rw [← pow_add, Nat.sub_add_cancel has.le]
    rw [hy] at h0
    exact h0.trans (dvd_of_eq (by ring))
  have hℓ0 : (ℓ : ℤ) ^ s ≠ 0 := pow_ne_zero _ (by exact_mod_cast hℓ.ne_zero)
  have h2 := (mul_dvd_mul_iff_right hℓ0).1 h1
  have hirr : Irreducible (ℓ : ℤ) := (Nat.prime_iff_prime_int.mp hℓ).irreducible
  have hcop : IsCoprime ((ℓ : ℤ) ^ (a - s)) y := ((hirr.coprime_iff_not_dvd).2 hy').pow_left
  exact hcop.dvd_of_dvd_mul_right h2

/-! ### The congruence ideal and the fibre product -/

/-- The **congruence ideal** `{a : (a, 0) ∈ image of T}` (01 Prop. 5.3: `C_T ≅ ℤ/g`, `gℤ = {a : (a,0) ∈ 𝕋^fr}`). -/
def congIdeal : Ideal ℤ where
  carrier := {a | ∃ t : T, E.lam t = a ∧ E.mu t = 0}
  add_mem' := by
    rintro _ _ ⟨t, rfl, ht⟩ ⟨s, rfl, hs⟩; exact ⟨t + s, by simp, by simp [ht, hs]⟩
  zero_mem' := ⟨0, by simp, by simp⟩
  smul_mem' := by
    rintro c _ ⟨t, rfl, ht⟩; exact ⟨(c : T) * t, by simp, by simp [ht]⟩

lemma mem_congIdeal {a : ℤ} : a ∈ E.congIdeal ↔ ∃ t : T, E.lam t = a ∧ E.mu t = 0 := Iff.rfl

/-- **The upper bound** (01 Prop. 5.3): `t − μ(t) ∈ T` lies over `(λ t − μ t, 0)`. -/
lemma mem_congIdeal_sub (t : T) : E.lam t - E.mu t ∈ E.congIdeal :=
  ⟨t - ((E.mu t : ℤ) : T), by simp, by simp⟩

/-- **The fibre product** ([U, lem:u-fibre], `A = ℤ`): with the lower bound `g ∣ λ t − μ t` for every `t` and `g` in the
congruence ideal, a pair `(a, c)` comes from `T` iff `a ≡ c mod g`. -/
theorem fibre_iff (g : ℤ) (hlow : ∀ t : T, g ∣ E.lam t - E.mu t) (hup : g ∈ E.congIdeal) (a c : ℤ) :
    (∃ t : T, E.lam t = a ∧ E.mu t = c) ↔ g ∣ a - c := by
  constructor
  · rintro ⟨t, rfl, rfl⟩; exact hlow t
  · rintro ⟨k, hk⟩
    obtain ⟨t0, h0, h0'⟩ := hup
    refine ⟨(c : T) + (k : T) * t0, ?_, ?_⟩
    · simp only [map_add, map_mul, map_intCast, h0, Int.cast_id]; linarith
    · simp [h0']

/-- `T` is faithful on its two systems: `t ↦ (λ t, μ t)` is injective (`L` torsion-free). -/
theorem pair_injective [Module.IsTorsionFree ℤ L] {t s : T} (h1 : E.lam t = E.lam s) (h2 : E.mu t = E.mu s) :
    t = s := by
  rw [← sub_eq_zero]
  apply Subtype.ext; apply LinearMap.ext; intro v
  have hk := E.key (t - s) v
  simp only [map_sub, h1, h2, sub_self, zero_smul, sub_zero] at hk
  rcases smul_eq_zero.mp hk with hk | hk
  · exact absurd (sub_eq_zero.mp hk) E.hD
  · simpa using hk

end EisSplit

end GL2.HeckeZ
