/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib
import GL2.Literature.AnemicRank
import GL2.Literature.FullRank
import GL2.HeckeZ.Bridge
import GL2.HeckeZ.BridgeN

/-!
# HeckeZ M-HZ5: three rational systems from the rank alone; pair congruence modules (L-HZ4, slot 26)

`07-lean-plan.md` §10 M-HZ5; 01 §4, §9 (T4); [C] `prop:heckeZcong`, `prop:heckeZrat`(2). Setting: `L` free of finite rank, `T ⊆ End_ℤ L`
a subalgebra of rank `≤ 3`, `U ∈ T` with three distinct integer eigenvalues `a, b, c` and integral eigenvectors. Then, as in
`GL2.HeckeZ.BridgeN.eisSplit_of_rank2` (the rank-2 case; 07 P2a-6):

* every `t ∈ T` satisfies `k t = α + β U + γ U²` for some `k ≠ 0` — 01's (C1) with `θ = U`, `T_ℚ = ℚ[U]` (`dep3`);
* every `t` acts on each eigenspace of `U` by an integer: three ring maps `T → ℤ` (`ThreeSys`, `exists_threeSys`), jointly
  injective — `T` embeds into `ℤ³`.

**Pair congruence modules.** For two ring maps `l₁, l₂ : T → ℤ`: `pairIdeal l₁ l₂ = {n : (n, 0) ∈ image of T}`, and `fibre_iff`:
with `g ∣ l₁ t − l₂ t` for all `t` and `g ∈ pairIdeal`, the image of `(l₁, l₂)` is `{(x, y) : g ∣ x − y}` — the
congruence module of the pair is `ℤ/g`.

**A read at a root for one Eisenstein system against the others** (`dvd_of_read_pair`, [C] `prop:heckeZcong`(5) in its
elementwise form): from `m • A = P − Q`, `z P = κ P`, `z Q = 0`: `m • (z A) = κ • P`, so a coordinate of `P v` of valuation `s`
and `ℓ^a ∣ m` give `ℓ^{a−s} ∣ κ`. At T4 (`U_2`, `z = t − μ t`, `P = (U−1)(U+8)`, `Q = (U−1)(U−128)`, `A = U − 1`, `m = 136`): `17` divides `λ₁₂₈ t − μ t`.

**The full-level theorem** `full_three` ties these to `hES_N`, `hES_N_an` and the exact matrices (TB-int): the pair `(E¹, E^{p^{k−1}})` of `(a, b)`.
-/

namespace GL2.HeckeZ.Three

open GL2.Literature GL2.HeckeZ GL2.HeckeZ.Bridge GL2.HeckeZ.BridgeN Polynomial

set_option linter.unusedSectionVars false

noncomputable section

variable {L : Type*} [AddCommGroup L] [Module.Free ℤ L] [Module.Finite ℤ L]

/-- **Dependence** (step 2 of `BridgeN.dep`, three eigenvalues): every `t ∈ T` is a polynomial in `U` over `ℚ`, cleared of denominators. -/
lemma dep3 (T : Subalgebra ℤ (Module.End ℤ L)) (hR : Module.finrank ℤ T ≤ 3) {U : Module.End ℤ L} (hU : U ∈ T)
    {a b c : ℤ} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) {va vb vc : L} (hva : va ≠ 0) (hvb : vb ≠ 0) (hvc : vc ≠ 0)
    (ea : U va = a • va) (eb : U vb = b • vb) (ec : U vc = c • vc) (t : T) :
    ∃ k α β γ : ℤ, k ≠ 0 ∧ k • (t : Module.End ℤ L) = α • (1 : Module.End ℤ L) + β • U + γ • (U * U) := by
  haveI hfin : Module.Finite ℤ T := Module.IsNoetherian.finite ℤ T.toSubmodule
  let uT : T := ⟨U, hU⟩
  have hni : ¬ LinearIndependent ℤ ![(1 : T), uT, uT * uT, t] := fun hli => by
    have := hli.fintype_card_le_finrank
    simp at this
    omega
  obtain ⟨g, hg, i, hi⟩ := Fintype.not_linearIndependent_iff.mp hni
  rw [Fin.sum_univ_four] at hg
  simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three, Matrix.tail_cons,
    Matrix.head_cons, Matrix.cons_val] at hg
  have hgE : g 0 • (1 : Module.End ℤ L) + g 1 • U + g 2 • (U * U) + g 3 • (t : Module.End ℤ L) = 0 := by
    have := congrArg (fun s : T => (s : Module.End ℤ L)) hg
    simpa [uT] using this
  by_cases h3 : g 3 = 0
  · exfalso
    rw [h3, zero_smul, add_zero] at hgE
    have ev : ∀ {v : L} {x : ℤ}, v ≠ 0 → U v = x • v → g 0 + g 1 * x + g 2 * x ^ 2 = 0 := by
      intro v x hv ex
      have h1 := congrArg (fun F : Module.End ℤ L => F v) hgE
      simp only [LinearMap.add_apply, LinearMap.smul_apply, Module.End.one_apply, Module.End.mul_apply, LinearMap.zero_apply] at h1
      rw [ex, map_zsmul, ex] at h1
      have h2 : (g 0 + g 1 * x + g 2 * x ^ 2) • v = 0 := by
        have e : (g 0 + g 1 * x + g 2 * x ^ 2) • v = g 0 • v + g 1 • x • v + g 2 • x • x • v := by module
        rw [e, h1]
      exact (smul_eq_zero.mp h2).resolve_right hv
    have h0a := ev hva ea
    have h0b := ev hvb eb
    have h0c := ev hvc ec
    have h2 : g 2 = 0 := by
      have : g 2 * ((a - b) * (a - c) * (b - c)) = 0 := by linear_combination (b - c) * h0a - (a - c) * h0b + (a - b) * h0c
      have hne : (a - b) * (a - c) * (b - c) ≠ 0 :=
        mul_ne_zero (mul_ne_zero (sub_ne_zero.mpr hab) (sub_ne_zero.mpr hac)) (sub_ne_zero.mpr hbc)
      exact (mul_eq_zero.mp this).resolve_right hne
    have h1 : g 1 = 0 := by
      rw [h2] at h0a h0b
      have : g 1 * (a - b) = 0 := by linear_combination h0a - h0b
      exact (mul_eq_zero.mp this).resolve_right (sub_ne_zero.mpr hab)
    have h0 : g 0 = 0 := by rw [h1, h2] at h0a; simpa using h0a
    rcases i with ⟨_ | _ | _ | _ | n, hn⟩
    · exact hi h0
    · exact hi h1
    · exact hi h2
    · exact hi h3
    · omega
  · refine ⟨g 3, -g 0, -g 1, -g 2, h3, ?_⟩
    rw [eq_neg_of_add_eq_zero_right hgE]; module

/-- The three systems of `T` on the eigenspaces of `U` (`a, b, c` distinct). -/
structure ThreeSys (T : Subalgebra ℤ (Module.End ℤ L)) (U : Module.End ℤ L) (a b c : ℤ) where
  /-- the system on `ker(U − a)` -/
  la : T →+* ℤ
  /-- the system on `ker(U − b)` -/
  lb : T →+* ℤ
  /-- the system on `ker(U − c)` -/
  lc : T →+* ℤ
  hUa : ∀ h : U ∈ T, la ⟨U, h⟩ = a
  hUb : ∀ h : U ∈ T, lb ⟨U, h⟩ = b
  hUc : ∀ h : U ∈ T, lc ⟨U, h⟩ = c
  acta : ∀ (t : T) (z : L), U z = a • z → (t : Module.End ℤ L) z = la t • z
  actb : ∀ (t : T) (z : L), U z = b • z → (t : Module.End ℤ L) z = lb t • z
  actc : ∀ (t : T) (z : L), U z = c • z → (t : Module.End ℤ L) z = lc t • z
  inj : ∀ t : T, la t = 0 → lb t = 0 → lc t = 0 → (t : Module.End ℤ L) = 0

/-- An element `t = (α + βU + γU²)/k` acts on the eigenspace `ker(U − x)` by an integer whenever it is non-zero. -/
lemma scalar_on_eig (U : Module.End ℤ L) {k α β γ x : ℤ} (hk : k ≠ 0) {t : Module.End ℤ L}
    (hform : k • t = α • (1 : Module.End ℤ L) + β • U + γ • (U * U)) {y : L} (hy : y ≠ 0) (ey : U y = x • y) :
    ∃ n : ℤ, t y = n • y ∧ ∀ z : L, U z = x • z → t z = n • z := by
  have act : ∀ z : L, U z = x • z → k • t z = (α + β * x + γ * x ^ 2) • z := by
    intro z ez
    have h1 := congrArg (fun F : Module.End ℤ L => F z) hform
    simp only [LinearMap.smul_apply, LinearMap.add_apply, Module.End.one_apply, Module.End.mul_apply] at h1
    rw [h1, ez, map_zsmul, ez]
    module
  obtain ⟨n, hn⟩ := int_eigen_div t hy hk (act y ey)
  have hcn : k * n = α + β * x + γ * x ^ 2 := by
    have h1 := act y ey
    rw [hn, smul_smul] at h1
    have : (k * n - (α + β * x + γ * x ^ 2)) • y = 0 := by rw [sub_smul, h1, sub_self]
    exact sub_eq_zero.mp ((smul_eq_zero.mp this).resolve_right hy)
  refine ⟨n, hn, fun z ez => ?_⟩
  have h1 := act z ez
  rw [← hcn, ← smul_smul] at h1
  exact sub_eq_zero.mp (zsmul_cancel hk (by rw [smul_sub, h1, sub_self]))

/-- **Three rational systems from the rank alone** (`T` of rank `≤ 3`, `U` with three distinct integral eigenvalues). -/
theorem exists_threeSys (T : Subalgebra ℤ (Module.End ℤ L)) (hR : Module.finrank ℤ T ≤ 3) {U : Module.End ℤ L} (hU : U ∈ T)
    {a b c : ℤ} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) {va vb vc : L} (hva : va ≠ 0) (hvb : vb ≠ 0) (hvc : vc ≠ 0)
    (ea : U va = a • va) (eb : U vb = b • vb) (ec : U vc = c • vc) : Nonempty (ThreeSys T U a b c) := by
  have D := fun t : T => dep3 T hR hU hab hac hbc hva hvb hvc ea eb ec t
  have hex : ∀ {x : ℤ} {y : L}, y ≠ 0 → U y = x • y → ∀ t : T, ∃ n : ℤ, (t : Module.End ℤ L) y = n • y := by
    intro x y hy ey t
    obtain ⟨k, α, β, γ, hk, hform⟩ := D t
    exact (scalar_on_eig U hk hform hy ey).imp fun _ h => h.1
  let la : T →+* ℤ := evHom hva (hex hva ea)
  let lb : T →+* ℤ := evHom hvb (hex hvb eb)
  let lc : T →+* ℤ := evHom hvc (hex hvc ec)
  have hall : ∀ {x : ℤ} {y : L} (hy : y ≠ 0) (ey : U y = x • y) (t : T),
      ∃ n : ℤ, (t : Module.End ℤ L) y = n • y ∧ ∀ z : L, U z = x • z → (t : Module.End ℤ L) z = n • z := by
    intro x y hy ey t
    obtain ⟨k, α, β, γ, hk, hform⟩ := D t
    exact scalar_on_eig U hk hform hy ey
  refine ⟨{ la := la, lb := lb, lc := lc, hUa := ?_, hUb := ?_, hUc := ?_, acta := ?_, actb := ?_, actc := ?_, inj := ?_ }⟩
  · intro h; exact evHom_eq hva (hex hva ea) _ ea
  · intro h; exact evHom_eq hvb (hex hvb eb) _ eb
  · intro h; exact evHom_eq hvc (hex hvc ec) _ ec
  · intro t z ez
    obtain ⟨n, hn, h⟩ := hall hva ea t
    rw [h z ez, evHom_eq hva (hex hva ea) t hn]
  · intro t z ez
    obtain ⟨n, hn, h⟩ := hall hvb eb t
    rw [h z ez, evHom_eq hvb (hex hvb eb) t hn]
  · intro t z ez
    obtain ⟨n, hn, h⟩ := hall hvc ec t
    rw [h z ez, evHom_eq hvc (hex hvc ec) t hn]
  · intro t h0a h0b h0c
    obtain ⟨k, α, β, γ, hk, hform⟩ := D t
    -- `α + β x + γ x²` vanishes at `a, b, c`
    have hact : ∀ {x : ℤ} {y : L}, y ≠ 0 → U y = x • y → (t : Module.End ℤ L) y = 0 → α + β * x + γ * x ^ 2 = 0 := by
      intro x y hy ey ht0
      have h1 := congrArg (fun F : Module.End ℤ L => F y) hform
      simp only [LinearMap.smul_apply, LinearMap.add_apply, Module.End.one_apply, Module.End.mul_apply, ht0, smul_zero] at h1
      rw [ey, map_zsmul, ey] at h1
      have h2 : (α + β * x + γ * x ^ 2) • y = 0 := by
        have e : (α + β * x + γ * x ^ 2) • y = α • y + β • x • y + γ • x • x • y := by module
        rw [e, ← h1]
      exact (smul_eq_zero.mp h2).resolve_right hy
    have hta : (t : Module.End ℤ L) va = 0 := by
      obtain ⟨n, hn, -⟩ := hall hva ea t
      rw [hn, show n = la t from (evHom_eq hva (hex hva ea) t hn).symm, h0a, zero_smul]
    have htb : (t : Module.End ℤ L) vb = 0 := by
      obtain ⟨n, hn, -⟩ := hall hvb eb t
      rw [hn, show n = lb t from (evHom_eq hvb (hex hvb eb) t hn).symm, h0b, zero_smul]
    have htc : (t : Module.End ℤ L) vc = 0 := by
      obtain ⟨n, hn, -⟩ := hall hvc ec t
      rw [hn, show n = lc t from (evHom_eq hvc (hex hvc ec) t hn).symm, h0c, zero_smul]
    have h0 := hact hva ea hta
    have h1 := hact hvb eb htb
    have h2 := hact hvc ec htc
    have hγ : γ = 0 := by
      have : γ * ((a - b) * (a - c) * (b - c)) = 0 := by linear_combination (b - c) * h0 - (a - c) * h1 + (a - b) * h2
      have hne : (a - b) * (a - c) * (b - c) ≠ 0 :=
        mul_ne_zero (mul_ne_zero (sub_ne_zero.mpr hab) (sub_ne_zero.mpr hac)) (sub_ne_zero.mpr hbc)
      exact (mul_eq_zero.mp this).resolve_right hne
    have hβ : β = 0 := by
      rw [hγ] at h0 h1
      have : β * (a - b) = 0 := by linear_combination h0 - h1
      exact (mul_eq_zero.mp this).resolve_right (sub_ne_zero.mpr hab)
    have hα : α = 0 := by rw [hβ, hγ] at h0; simpa using h0
    rw [hα, hβ, hγ] at hform
    simp only [zero_smul, add_zero] at hform
    exact end_zsmul_cancel hk hform

/-! ### Pair congruence modules -/

/-- `{n : (n, 0) ∈ image of (l₁, l₂)}`. -/
def pairIdeal {T : Subalgebra ℤ (Module.End ℤ L)} (l₁ l₂ : T →+* ℤ) : Ideal ℤ where
  carrier := {n | ∃ t : T, l₁ t = n ∧ l₂ t = 0}
  add_mem' := by
    rintro _ _ ⟨t, rfl, ht⟩ ⟨s, rfl, hs⟩; exact ⟨t + s, by simp, by simp [ht, hs]⟩
  zero_mem' := ⟨0, by simp, by simp⟩
  smul_mem' := by
    rintro c _ ⟨t, rfl, ht⟩; exact ⟨(c : T) * t, by simp, by simp [ht]⟩

lemma mem_pairIdeal {T : Subalgebra ℤ (Module.End ℤ L)} {l₁ l₂ : T →+* ℤ} {n : ℤ} :
    n ∈ pairIdeal l₁ l₂ ↔ ∃ t : T, l₁ t = n ∧ l₂ t = 0 := Iff.rfl

/-- `t − l₂(t)` lies over `(l₁ t − l₂ t, 0)`: every difference is in the ideal (01 Prop. 5.3, the upper bound). -/
lemma sub_mem_pairIdeal {T : Subalgebra ℤ (Module.End ℤ L)} (l₁ l₂ : T →+* ℤ) (t : T) : l₁ t - l₂ t ∈ pairIdeal l₁ l₂ :=
  ⟨t - ((l₂ t : ℤ) : T), by simp, by simp⟩

/-- **The pair fibre product** ([U, lem:u-fibre], `A = ℤ`): `(x, y)` comes from `T` iff `g ∣ x − y`. -/
theorem fibre_iff {T : Subalgebra ℤ (Module.End ℤ L)} (l₁ l₂ : T →+* ℤ) (g : ℤ) (hlow : ∀ t, g ∣ l₁ t - l₂ t)
    (hup : g ∈ pairIdeal l₁ l₂) (x y : ℤ) : (∃ t : T, l₁ t = x ∧ l₂ t = y) ↔ g ∣ x - y := by
  constructor
  · rintro ⟨t, rfl, rfl⟩; exact hlow t
  · rintro ⟨k, hk⟩
    obtain ⟨t0, h0, h0'⟩ := hup
    refine ⟨(y : T) + (k : T) * t0, ?_, ?_⟩
    · simp only [map_add, map_mul, map_intCast, h0, Int.cast_id]; linarith
    · simp [h0']

/-! ### A read at a root against the other systems -/

/-- **Pair read** ([C] `prop:heckeZcong`(5), elementwise): from `m • A = P − Q`, `z P = κ P`, `z Q = 0` one gets
`m • z (A v) = κ • P v`, so a coordinate of `P v` of `ℓ`-valuation exactly `s` and `ℓ^a ∣ m` give `ℓ^{a−s} ∣ κ`. -/
theorem dvd_of_read_pair {ι : Type*} (b : Module.Basis ι ℤ L) (ℓ : ℕ) (hℓ : ℓ.Prime) {m κ : ℤ}
    {A P Q z : Module.End ℤ L} (hAPQ : m • A = P - Q) (hzP : z * P = κ • P) (hzQ : z * Q = 0) {a s : ℕ}
    (hm : (ℓ : ℤ) ^ a ∣ m) (v : L) (i : ι) (hs : (ℓ : ℤ) ^ s ∣ b.repr (P v) i)
    (hs1 : ¬ (ℓ : ℤ) ^ (s + 1) ∣ b.repr (P v) i) : (ℓ : ℤ) ^ (a - s) ∣ κ := by
  have hkey : m • z (A v) = κ • P v := by
    have h1 := congrArg (fun F : Module.End ℤ L => F v) hAPQ
    simp only [LinearMap.smul_apply, LinearMap.sub_apply] at h1
    have h2 := congrArg (fun F : Module.End ℤ L => F v) hzP
    have h3 := congrArg (fun F : Module.End ℤ L => F v) hzQ
    simp only [Module.End.mul_apply, LinearMap.smul_apply, LinearMap.zero_apply] at h2 h3
    calc m • z (A v) = z (m • A v) := by rw [map_zsmul]
      _ = z (P v - Q v) := by rw [h1]
      _ = κ • P v := by rw [map_sub, h2, h3, sub_zero]
  have hk : m * b.repr (z (A v)) i = κ * b.repr (P v) i := by
    have := congrArg (fun w => b.repr w i) hkey
    simpa only [map_zsmul, Finsupp.smul_apply, smul_eq_mul] using this
  have h0 : (ℓ : ℤ) ^ a ∣ κ * b.repr (P v) i := by rw [← hk]; exact hm.mul_right _
  obtain ⟨y, hy⟩ := hs
  have hy' : ¬ (ℓ : ℤ) ∣ y := by rintro ⟨w, rfl⟩; exact hs1 ⟨w, by rw [hy, pow_succ]; ring⟩
  rcases le_or_gt a s with has | has
  · rw [Nat.sub_eq_zero_of_le has, pow_zero]; exact one_dvd _
  have h1 : (ℓ : ℤ) ^ (a - s) * (ℓ : ℤ) ^ s ∣ (κ * y) * (ℓ : ℤ) ^ s := by
    rw [← pow_add, Nat.sub_add_cancel has.le]
    rw [hy] at h0
    exact h0.trans (dvd_of_eq (by ring))
  have hℓ0 : (ℓ : ℤ) ^ s ≠ 0 := pow_ne_zero _ (by exact_mod_cast hℓ.ne_zero)
  have h2 := (mul_dvd_mul_iff_right hℓ0).1 h1
  have hirr : Irreducible (ℓ : ℤ) := (Nat.prime_iff_prime_int.mp hℓ).irreducible
  have hcop : IsCoprime ((ℓ : ℤ) ^ (a - s)) y := ((hirr.coprime_iff_not_dvd).2 hy').pow_left
  exact hcop.dvd_of_dvd_mul_right h2

/-! ### The read for the systems `b` against `c`, when `f(U) = 0` -/

/-- **The pair read on a third system** (T4: `ℓ = 17`, `a, b, c = 1, 128, −8`). With `(U − a)(U − b)(U − c) = 0` and
`P = (U − a)(U − c)`, `Q = (U − a)(U − b)`: `z = t − l_c(t)` satisfies `z P = (l_b t − l_c t) P` and `z Q = 0`, so a coordinate
of `P v` of `ℓ`-valuation `s` and `ℓ^e ∣ b − c` give `ℓ^{e−s} ∣ l_b(t) − l_c(t)`. -/
theorem read_b_c {T : Subalgebra ℤ (Module.End ℤ L)} {U : Module.End ℤ L} (hU : U ∈ T) {a b c : ℤ}
    (S : ThreeSys T U a b c) (hf : (U - (a : Module.End ℤ L)) * (U - (b : Module.End ℤ L)) * (U - (c : Module.End ℤ L)) = 0)
    (t : T) {ι : Type*} (bb : Module.Basis ι ℤ L) (ℓ : ℕ) (hℓ : ℓ.Prime) {e s : ℕ} (hm : (ℓ : ℤ) ^ e ∣ b - c) (v : L) (i : ι)
    (hs : (ℓ : ℤ) ^ s ∣ bb.repr (((U - (a : Module.End ℤ L)) * (U - (c : Module.End ℤ L))) v) i)
    (hs1 : ¬ (ℓ : ℤ) ^ (s + 1) ∣ bb.repr (((U - (a : Module.End ℤ L)) * (U - (c : Module.End ℤ L))) v) i) :
    (ℓ : ℤ) ^ (e - s) ∣ S.lb t - S.lc t := by
  have hAPQ : (b - c) • (U - (a : Module.End ℤ L)) =
      (U - (a : Module.End ℤ L)) * (U - (c : Module.End ℤ L)) - (U - (a : Module.End ℤ L)) * (U - (b : Module.End ℤ L)) := by
    have h := congrArg (aeval U) (show (C b - C c) * (Polynomial.X - C a) =
      (Polynomial.X - C a) * (Polynomial.X - C c) - (Polynomial.X - C a) * (Polynomial.X - C b) by ring)
    simp only [map_mul, map_sub, aeval_X, aeval_C, eq_intCast, map_intCast] at h
    rw [← h, zsmul_eq_mul, Int.cast_sub]
  let z : Module.End ℤ L := (t : Module.End ℤ L) - ((S.lc t : ℤ) : Module.End ℤ L)
  have hzP : z * ((U - (a : Module.End ℤ L)) * (U - (c : Module.End ℤ L))) =
      (S.lb t - S.lc t) • ((U - (a : Module.End ℤ L)) * (U - (c : Module.End ℤ L))) := by
    ext w
    have hy : U (((U - (a : Module.End ℤ L)) * (U - (c : Module.End ℤ L))) w) =
        b • ((U - (a : Module.End ℤ L)) * (U - (c : Module.End ℤ L))) w := by
      have e := congrArg (aeval U) (show (Polynomial.X - C b) * ((Polynomial.X - C a) * (Polynomial.X - C c)) =
        (Polynomial.X - C a) * (Polynomial.X - C b) * (Polynomial.X - C c) by ring)
      simp only [map_mul, map_sub, aeval_X, aeval_C, eq_intCast, map_intCast] at e
      have h1 := congrArg (fun F : Module.End ℤ L => F w) (e.trans hf)
      simp only [Module.End.mul_apply (U - (b : Module.End ℤ L)), LinearMap.zero_apply, LinearMap.sub_apply U (b : Module.End ℤ L),
        Module.End.intCast_apply] at h1
      exact sub_eq_zero.mp h1
    have h2 := S.actb t _ hy
    simp only [z, Module.End.mul_apply ((t : Module.End ℤ L) - _), LinearMap.sub_apply ((t : Module.End ℤ L)), Module.End.intCast_apply,
      LinearMap.smul_apply, h2]
    rw [sub_smul]
  have hzQ : z * ((U - (a : Module.End ℤ L)) * (U - (b : Module.End ℤ L))) = 0 := by
    ext w
    have hy : U (((U - (a : Module.End ℤ L)) * (U - (b : Module.End ℤ L))) w) =
        c • ((U - (a : Module.End ℤ L)) * (U - (b : Module.End ℤ L))) w := by
      have e := congrArg (aeval U) (show (Polynomial.X - C c) * ((Polynomial.X - C a) * (Polynomial.X - C b)) =
        (Polynomial.X - C a) * (Polynomial.X - C b) * (Polynomial.X - C c) by ring)
      simp only [map_mul, map_sub, aeval_X, aeval_C, eq_intCast, map_intCast] at e
      have h1 := congrArg (fun F : Module.End ℤ L => F w) (e.trans hf)
      simp only [Module.End.mul_apply (U - (c : Module.End ℤ L)), LinearMap.zero_apply, LinearMap.sub_apply U (c : Module.End ℤ L),
        Module.End.intCast_apply] at h1
      exact sub_eq_zero.mp h1
    have h2 := S.actc t _ hy
    simp only [z, Module.End.mul_apply ((t : Module.End ℤ L) - _), LinearMap.sub_apply ((t : Module.End ℤ L)), Module.End.intCast_apply,
      LinearMap.zero_apply, h2, sub_self]
  exact dvd_of_read_pair bb ℓ hℓ hAPQ hzP hzQ hm v i hs hs1

/-! ### Full-level helpers -/

variable {X U : ℕ → Module.End ℤ L} {N k d : ℕ}

/-- `𝕋^fr` is commutative (`hES_N`). -/
lemma tfull_comm (hF : FullRankLevelN X U N k d) : ∀ x ∈ Tfull X U, ∀ y ∈ Tfull X U, x * y = y * x := by
  have hcomm : ∀ a ∈ Set.range X ∪ Set.range U, ∀ b ∈ Set.range X ∪ Set.range U, a * b = b * a := by
    rintro _ (⟨i, rfl⟩ | ⟨i, rfl⟩) _ (⟨j, rfl⟩ | ⟨j, rfl⟩)
    · exact hF.commX i j
    · exact hF.commU i j
    · exact (hF.commU j i).symm
    · exact hF.commUU i j
  letI := Algebra.adjoinCommRingOfComm ℤ hcomm
  intro x hx y hy
  exact congrArg Subtype.val (mul_comm (⟨x, hx⟩ : Algebra.adjoin ℤ (Set.range X ∪ Set.range U)) ⟨y, hy⟩)

lemma Tfr_le_Tfull : Bridge.Tfr X ≤ Tfull X U :=
  Algebra.adjoin_mono Set.subset_union_left

/-- `T = T^an + U_{p₀} T` when `p₀` is the only prime with `U_p ≠ 0` (`N = p₀^r`). -/
lemma tfull_gen (hF : FullRankLevelN X U N k d) (p₀ : ℕ) (hp₀ : ∀ p, p ≠ p₀ → U p = 0) :
    ∀ t ∈ Tfull X U, ∃ s ∈ Bridge.Tfr X, ∃ y ∈ Tfull X U, t = s + U p₀ * y := by
  have hc := tfull_comm hF
  have hUmem : U p₀ ∈ Tfull X U := Algebra.subset_adjoin (Or.inr ⟨p₀, rfl⟩)
  intro t ht
  induction ht using Algebra.adjoin_induction with
  | mem x hx =>
    rcases hx with ⟨i, rfl⟩ | ⟨j, rfl⟩
    · exact ⟨X i, Algebra.subset_adjoin ⟨i, rfl⟩, 0, Subalgebra.zero_mem _, by simp⟩
    · by_cases hj : j = p₀
      · subst hj; exact ⟨0, Subalgebra.zero_mem _, 1, Subalgebra.one_mem _, by simp⟩
      · rw [hp₀ j hj]; exact ⟨0, Subalgebra.zero_mem _, 0, Subalgebra.zero_mem _, by simp⟩
  | algebraMap r =>
    exact ⟨algebraMap ℤ _ r, Subalgebra.algebraMap_mem _ r, 0, Subalgebra.zero_mem _, by simp⟩
  | add x y _ _ hx hy =>
    obtain ⟨s1, hs1, y1, hy1, rfl⟩ := hx
    obtain ⟨s2, hs2, y2, hy2, rfl⟩ := hy
    refine ⟨s1 + s2, Subalgebra.add_mem _ hs1 hs2, y1 + y2, Subalgebra.add_mem _ hy1 hy2, ?_⟩
    rw [mul_add]; abel
  | mul x y hxm hym hx hy =>
    obtain ⟨s1, hs1, y1, hy1, rfl⟩ := hx
    obtain ⟨s2, hs2, y2, hy2, rfl⟩ := hy
    have hu1 : s1 * U p₀ = U p₀ * s1 := hc s1 (Tfr_le_Tfull hs1) _ hUmem
    refine ⟨s1 * s2, Subalgebra.mul_mem _ hs1 hs2, y1 * s2 + s1 * y2 + y1 * (U p₀ * y2), ?_, ?_⟩
    · exact Subalgebra.add_mem _ (Subalgebra.add_mem _ (Subalgebra.mul_mem _ hy1 (Tfr_le_Tfull hs2))
        (Subalgebra.mul_mem _ (Tfr_le_Tfull hs1) hy2)) (Subalgebra.mul_mem _ hy1 (Subalgebra.mul_mem _ hUmem hy2))
    · have e1 : U p₀ * y1 * (U p₀ * y2) = U p₀ * (y1 * (U p₀ * y2)) := by rw [mul_assoc]
      rw [add_mul, mul_add, mul_add, e1, ← mul_assoc s1 (U p₀), hu1]
      noncomm_ring

/-- **Full-level structure from `hES_N`, `hES_N_an`** (TB-int provides the eigenvector identities).
`N = p₀^r`, `U_{p₀}` with three distinct integer eigenvalues `a, b, c` on `va, vb, vc`; `X i₀` scalar `e` on `va, vb` and `g ≠ e` on `vc`; anemic rank `≤ 2`, full rank `≤ 3`:
three systems; the pair `(la, lb)` congruent mod `b − a` on all of `𝕋` with the exact fibre product. -/
theorem full_three (hF : FullRankLevelN X U N k d) (hd : d ≤ 3) (hAn : Module.finrank ℤ (Bridge.Tfr X) ≤ 2) (p₀ : ℕ)
    (hp₀ : ∀ p, p ≠ p₀ → U p = 0) (i₀ : ℕ) {a b c e g : ℤ} (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (heg : e ≠ g)
    {va vb vc : L} (hva : va ≠ 0) (hvb : vb ≠ 0) (hvc : vc ≠ 0)
    (ea : U p₀ va = a • va) (eb : U p₀ vb = b • vb) (ec : U p₀ vc = c • vc)
    (ta : X i₀ va = e • va) (tb : X i₀ vb = e • vb) (tc : X i₀ vc = g • vc) :
    ∃ S : ThreeSys (Tfull X U) (U p₀) a b c,
      (∀ n h, S.la ⟨X n, h⟩ = S.lb ⟨X n, h⟩) ∧
      (∀ t : Tfull X U, (b - a) ∣ S.la t - S.lb t) ∧
      (∀ x y, (∃ t : Tfull X U, S.la t = x ∧ S.lb t = y) ↔ (b - a) ∣ x - y) := by
  have hUmem : U p₀ ∈ Tfull X U := Algebra.subset_adjoin (Or.inr ⟨p₀, rfl⟩)
  have hR3 : Module.finrank ℤ (Tfull X U) ≤ 3 := by rw [hF.rank]; exact hd
  obtain ⟨S⟩ := exists_threeSys (Tfull X U) hR3 hUmem hab hac hbc hva hvb hvc ea eb ec
  have hXeq : ∀ n h, S.la ⟨X n, h⟩ = S.lb ⟨X n, h⟩ := by
    intro n h
    obtain ⟨n', hn, hall⟩ := scalar_on X hAn i₀ heg.symm hvc hva tc ta (y := va) (a := e) hva ta (⟨X n, Algebra.subset_adjoin ⟨n, rfl⟩⟩ : Bridge.Tfr X)
    have e1 : (X n) va = S.la ⟨X n, h⟩ • va := S.acta ⟨X n, h⟩ va ea
    have e2 : (X n) vb = S.lb ⟨X n, h⟩ • vb := S.actb ⟨X n, h⟩ vb eb
    have e3 : (X n) vb = n' • vb := hall vb tb
    have e4 : (X n) va = n' • va := hn
    have h1 : S.la ⟨X n, h⟩ = n' := by
      have : (S.la ⟨X n, h⟩ - n') • va = 0 := by rw [sub_smul, ← e1, ← e4, sub_self]
      exact sub_eq_zero.mp ((smul_eq_zero.mp this).resolve_right hva)
    have h2 : S.lb ⟨X n, h⟩ = n' := by
      have : (S.lb ⟨X n, h⟩ - n') • vb = 0 := by rw [sub_smul, ← e2, ← e3, sub_self]
      exact sub_eq_zero.mp ((smul_eq_zero.mp this).resolve_right hvb)
    rw [h1, h2]
  have hUdiff : ∀ h, S.la ⟨U p₀, h⟩ - S.lb ⟨U p₀, h⟩ = a - b := fun h => by rw [S.hUa, S.hUb]
  have hlow : ∀ t : Tfull X U, (b - a) ∣ S.la t - S.lb t := by
    rintro ⟨t, ht⟩
    induction ht using Algebra.adjoin_induction with
    | mem x hx =>
      rcases hx with ⟨i, rfl⟩ | ⟨j, rfl⟩
      · have := hXeq i (Algebra.subset_adjoin (Or.inl ⟨i, rfl⟩))
        rw [this, sub_self]; exact dvd_zero _
      · by_cases hj : j = p₀
        · subst hj; rw [hUdiff]; exact ⟨-1, by ring⟩
        · have h0 : ∀ h, S.la ⟨U j, h⟩ = 0 ∧ S.lb ⟨U j, h⟩ = 0 := fun h => by
            have ez : (⟨U j, h⟩ : Tfull X U) = 0 := Subtype.ext (hp₀ j hj)
            rw [ez]; simp
          rw [(h0 _).1, (h0 _).2, sub_self]; exact dvd_zero _
    | algebraMap r =>
      have e : (⟨algebraMap ℤ _ r, Subalgebra.algebraMap_mem _ r⟩ : Tfull X U) = (r : Tfull X U) := by
        apply Subtype.ext; simp
      rw [e, map_intCast, map_intCast, sub_self]; exact dvd_zero _
    | add x y hxm hym hx hy =>
      have ex : (⟨x + y, Subalgebra.add_mem _ hxm hym⟩ : Tfull X U) = ⟨x, hxm⟩ + ⟨y, hym⟩ := rfl
      rw [ex, map_add, map_add, show ∀ p q r s : ℤ, p + q - (r + s) = (p - r) + (q - s) by intros; ring]
      exact dvd_add hx hy
    | mul x y hxm hym hx hy =>
      have ex : (⟨x * y, Subalgebra.mul_mem _ hxm hym⟩ : Tfull X U) = ⟨x, hxm⟩ * ⟨y, hym⟩ := rfl
      rw [ex, map_mul, map_mul, show ∀ p q r s : ℤ, p * q - r * s = (p - r) * q + r * (q - s) by intros; ring]
      exact dvd_add (hx.mul_right _) (hy.mul_left _)
  have hup : (b - a) ∈ pairIdeal S.la S.lb := by
    refine ⟨((b : Tfull X U) - ⟨U p₀, hUmem⟩), ?_, ?_⟩
    · simp [map_sub, S.hUa]
    · simp [map_sub, S.hUb]
  exact ⟨S, hXeq, hlow, fibre_iff S.la S.lb (b - a) hlow hup⟩

end

end GL2.HeckeZ.Three
