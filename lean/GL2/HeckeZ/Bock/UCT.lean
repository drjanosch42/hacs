/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import Mathlib

/-!
# HeckeZ 03 B1 (LB-1): the universal-coefficient sequence modulo `N = ℓ^K`, elementary (L-HZ2, M-HZ3)

`docs/implementation/slot24/heckez/03-bockstein-tor.md` §2.1 (B1(1), (3)), §8.1 (LB-1); `07-lean-plan.md` §2.3 (LB-1, the
**elementary route**, F07-10); [C] `prop:heckeZuct`(1), (3).

A window `A →d₀ B →d₁ C →d₂ D` of a cochain complex of `ℤ`-modules (`d₁ d₀ = 0`, `d₂ d₁ = 0`) with `C`, `D` without `N`-torsion
(true for a complex of free modules). On `B` (degree `q`):
* `H^q = Z/B` with `Z = ker d₁`, `B = im d₀`;
* `H^q(C/N) = Z_N/B_N` with `Z_N = {x : d₁ x ∈ N C}` (cocycles modulo `N`) and `B_N = im d₀ + N B` — the cohomology of `C/N`
  computed on representatives, without forming the quotient complex;
* `H^{q+1} = Z'/B'` with `Z' = ker d₂`, `B' = im d₁` (on `C`).

`red : Z → Z_N` is the inclusion, `β : Z_N → Z'` is `x ↦ N⁻¹ d₁ x` (`bock`). The hand-written snake lemma, elementwise:
* `bock_spec`, `bock_mem_ker`: `β` is well defined, with values in `Z'`, and `N β x = d₁ x ∈ B'` (`β` lands in `H^{q+1}[N]`);
* `bock_of_mem_BN`: `β (B_N) ⊆ B'` (well defined on classes);
* `exact_red`: `ker(H^q → H^q(C/N)) = N H^q` (`red z ∈ B_N ⟺ z ∈ N Z + B`);
* `exact_mid`: `im red = ker β` (`β x ∈ B' ⟺ x ∈ Z + B_N`);
* `exact_bock`: `β` is onto `H^{q+1}[N]` (`N y ∈ B'` for `y ∈ Z'` ⟹ `y = β x`);
* `bock_natural`: naturality in a cochain endomorphism `F` (B1(3): the induced map on `H^q(C/N)` commutes with `β`; it is
  `T_p` there only if `F` is a cochain-level lift of `T_p`, a hypothesis, A3-m1);
* `Caveat.*`: A3-m1's example — `F` is the identity on `H^•(C)` but not on `H^0(C/ℓ)`.
The quotient modules and the cardinality `|H^q(C/N)| = |H^q/N| · |H^{q+1}[N]|` are not assembled here (LB-2 holds the type count).
-/

namespace GL2.HeckeZ.Bock

section UCT

variable {A B C D : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D]
  (d₀ : A →+ B) (d₁ : B →+ C) (d₂ : C →+ D) (N : ℤ)

/-- `x` is a cocycle modulo `N`: `d₁ x ∈ N C`. -/
def ZN (x : B) : Prop := ∃ y : C, d₁ x = N • y

/-- `x` is a coboundary modulo `N`: `x ∈ im d₀ + N B`. -/
def BN (x : B) : Prop := ∃ (a : A) (b : B), x = d₀ a + N • b

variable {d₀ d₁ d₂ N}

/-- The Bockstein `β x = N⁻¹ d₁ x`. -/
noncomputable def bock {x : B} (hx : ZN d₁ N x) : C := Classical.choose hx

theorem bock_spec {x : B} (hx : ZN d₁ N x) : d₁ x = N • bock hx := Classical.choose_spec hx

variable (hC : ∀ y : C, N • y = 0 → y = 0) (hD : ∀ z : D, N • z = 0 → z = 0)
include hC

/-- `β x` is the unique `y` with `d₁ x = N y`. -/
theorem bock_unique {x : B} (hx : ZN d₁ N x) {y : C} (hy : d₁ x = N • y) : bock hx = y := by
  have h := bock_spec hx
  exact sub_eq_zero.mp (hC _ (by rw [smul_sub, ← h, ← hy, sub_self]))

omit hC in
include hD in
/-- `β x ∈ Z' = ker d₂`. -/
theorem bock_mem_ker (h21 : ∀ b, d₂ (d₁ b) = 0) {x : B} (hx : ZN d₁ N x) : d₂ (bock hx) = 0 :=
  hD _ (by rw [← map_zsmul, ← bock_spec hx, h21])

omit hC in
/-- `N β x = d₁ x ∈ B'`: `β` lands in `H^{q+1}[N]`. -/
theorem smul_bock_mem_range {x : B} (hx : ZN d₁ N x) : ∃ b, d₁ b = N • bock hx := ⟨x, bock_spec hx⟩

/-- `β (B_N) ⊆ B'`: `β` is well defined on classes. -/
theorem bock_of_mem_BN (h10 : ∀ a, d₁ (d₀ a) = 0) {x : B} (hx : ZN d₁ N x) (hb : BN d₀ N x) :
    ∃ b, d₁ b = bock hx := by
  obtain ⟨a, b, rfl⟩ := hb
  exact ⟨b, (bock_unique hC hx (by rw [map_add, h10, zero_add, map_zsmul])).symm⟩

/-- **Exactness at `H^q/N`**: a cocycle `z` is a coboundary modulo `N` iff `z ∈ N Z + B` (`ker red = N H^q`). -/
theorem exact_red (h10 : ∀ a, d₁ (d₀ a) = 0) {z : B} (hz : d₁ z = 0) :
    BN d₀ N z ↔ ∃ (c : B) (a : A), d₁ c = 0 ∧ z = N • c + d₀ a := by
  constructor
  · rintro ⟨a, b, rfl⟩
    refine ⟨b, a, hC _ ?_, by abel⟩
    rw [← map_zsmul, ← add_sub_cancel_left (d₀ a) (N • b), map_sub, hz, h10, sub_zero]
  · rintro ⟨c, a, -, rfl⟩
    exact ⟨a, c, by abel⟩

/-- **Exactness at `H^q(C/N)`** (`im red = ker β`): `β x ∈ B'` iff `x` is a cocycle plus a coboundary modulo `N`. -/
theorem exact_mid (h10 : ∀ a, d₁ (d₀ a) = 0) {x : B} (hx : ZN d₁ N x) :
    (∃ w, d₁ w = bock hx) ↔ ∃ z, d₁ z = 0 ∧ BN d₀ N (x - z) := by
  constructor
  · rintro ⟨w, hw⟩
    refine ⟨x - N • w, ?_, ⟨0, w, by simp⟩⟩
    rw [map_sub, map_zsmul, hw, bock_spec hx, sub_self]
  · rintro ⟨z, hz, a, b, he⟩
    refine ⟨b, (bock_unique hC hx ?_).symm⟩
    have : x = z + (d₀ a + N • b) := by rw [← he]; abel
    rw [this, map_add, map_add, hz, h10, map_zsmul, zero_add, zero_add]

/-- **Exactness at `H^{q+1}[N]`** (`β` onto): a cocycle `y` of degree `q+1` with `N y ∈ B'` is `β x`. -/
theorem exact_bock {y : C} {x : B} (hxy : d₁ x = N • y) : ∃ hx : ZN d₁ N x, bock hx = y :=
  ⟨⟨y, hxy⟩, bock_unique hC ⟨y, hxy⟩ hxy⟩

/-- **Naturality** in a cochain map `F = (F_B, F_C)` (`F_C d₁ = d₁ F_B`): `β (F_B x) = F_C (β x)`. -/
theorem bock_natural (FB : B →+ B) (FC : C →+ C) (hF : ∀ b, d₁ (FB b) = FC (d₁ b)) {x : B} (hx : ZN d₁ N x)
    (hx' : ZN d₁ N (FB x)) : bock hx' = FC (bock hx) :=
  bock_unique hC hx' (by rw [hF, bock_spec hx, map_zsmul])

end UCT

/-! ### A3-m1's example: identity on `H^•(C)`, not on `H^0(C/ℓ)` -/

namespace Caveat

/-- `C⁰ = ℤ a ⊕ ℤ b`, `C¹ = ℤ c`, `d a = 0`, `d b = ℓ c`. -/
def d (ℓ : ℤ) : ℤ × ℤ →+ ℤ := (AddMonoidHom.mulLeft ℓ).comp (AddMonoidHom.snd ℤ ℤ)

/-- `F a = a`, `F b = b + a`, `F c = c`. -/
def F : ℤ × ℤ →+ ℤ × ℤ where
  toFun v := (v.1 + v.2, v.2)
  map_zero' := rfl
  map_add' v w := by ext <;> simp <;> ring

/-- `F` is a cochain map: `d F = d`. -/
theorem d_F (ℓ : ℤ) (v : ℤ × ℤ) : d ℓ (F v) = d ℓ v := rfl

/-- `F` is the identity on `H⁰(C) = ker d = ℤ a` (for `ℓ ≠ 0`) and on `H¹(C) = C¹/ℓ` (`F_C = id`). -/
theorem F_id_H0 {ℓ : ℤ} (hℓ : ℓ ≠ 0) {v : ℤ × ℤ} (hv : d ℓ v = 0) : F v = v := by
  have : v.2 = 0 := by simpa [d, hℓ] using hv
  ext <;> simp [F, this]

/-- `F` is **not** the identity on `H⁰(C/ℓ) = C⁰/ℓ C⁰` (every element is a cocycle modulo `ℓ`, no coboundaries): `F b − b = a`
is not in `ℓ C⁰` for `ℓ ∤ 1`. -/
theorem F_ne_id_H0_mod {ℓ : ℤ} (hℓ : ¬ ℓ ∣ 1) : ¬ ∃ w : ℤ × ℤ, F (0, 1) - (0, 1) = ℓ • w := by
  rintro ⟨w, hw⟩
  have := congrArg Prod.fst hw
  simp [F] at this
  exact hℓ ⟨w.1, by linarith⟩

end Caveat

end GL2.HeckeZ.Bock
