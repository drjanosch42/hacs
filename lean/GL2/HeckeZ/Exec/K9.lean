/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.K10

/-!
# HeckeZ `HZExec`: the executable K9 checker, the Hida one-weight certificate (L-K9, M-HZ4) — Mathlib-free

`docs/implementation/slot24/heckez/07-lean-plan.md` §5.9 (K9), §11.1 (its fields); [U] 0.7.0-pre part 6 `sec:u-eislambda`,
`prop:u-H11` (cuspidal rank one: the generator test), `prop:u-H12` ((D1), (D3) at `(547, 486)`); data `slot25/H1_report.md` §5
(`docs/implementation/slot25/h1/k9_data.tsv`). Brief `slot25/LK9_brief.md`. This module imports only `GL2.HeckeZ.Exec.K10` (exact
list polynomials over `ℤ`, trial-division `primeB`, `powMod`), so `k9Check c = true` is decided by `decide +kernel` without Mathlib.
Soundness: `GL2/HeckeZ/Check/K9.lean`.

**K9 certificate** (one irregular pair `(ℓ, a)` at one weight `k ≡ a mod (ℓ − 1)`; every residue below `ℓ²`):
* `flo`: the lower coefficients of `f_k = flo + x^g` modulo `ℓ²` (`g = flo.length`, the cuspidal rank of the ordinary Eisenstein
  block): **trust input TB-χ (block form)**, the characteristic polynomial of `T₂` on the block cut out by the full residual system
  (H1's `HeckeHidaEisLambda::ReadBlock`);
* `e`: `e_k = 1 + 2^{k−1} mod ℓ²` (the Eisenstein eigenvalue of `T₂`), **recomputed** by the checker;
* `bk`: `B_k/k mod ℓ²` (**trust input TB-B**, the C++ residue);
* `gk`: `G_a(x_k) mod ℓ²`, which is `−(1 − ℓ^{k−1}) B_k/k ≡ −B_k/k` (`k ≥ 3`; checked against `bk`);
* `gnext`: `G_a(x_{k+ℓ−1}) mod ℓ²` (TB-B at the weight `k + ℓ − 1`);
* `lam`: the λ-residue `(G_a(x_k) − G_a(x_{k+ℓ−1}))/ℓ mod ℓ`.

The checks: `ℓ` prime, `ℓ ≥ 5`, `k ≥ 3`, `(ℓ − 1) ∤ k` (the class `a ≢ 0`), `g ≥ 1`; `e = (1 + 2^{k−1}) mod ℓ²`; `f ≡ (x − e)^g (mod ℓ)`;
`ℓ ∥ f(e)` and `ℓ ∥ bk` at precision `ℓ²` (so `v(f_k(e_k)) = v(B_k/k) = 1`); for `g = 2`, `ℓ ∥ disc f = f₁² − 4f₀`; `gk ≡ −bk (mod ℓ²)`;
the λ-test `gk − gnext ≡ ℓ·lam (mod ℓ²)` with `lam ≢ 0 (mod ℓ)`.
-/

noncomputable section

namespace GL2.HeckeZ.Exec.K9

open GL2.L23.SmithPK (powMod)
open GL2.HeckeZ.Exec.K10 (peval plin coeffD peqMod primeB rounds)

/-- A K9 certificate (07 §5.9). -/
structure K9Cert where
  /-- the irregular prime `ℓ` -/
  ell : Nat
  /-- the weight `k` (in the class `a`) -/
  k : Nat
  /-- the lower coefficients of `f_k = flo + x^g` (residues mod `ℓ²`; TB-χ, block form) -/
  flo : List Int
  /-- `e_k = 1 + 2^{k−1} mod ℓ²` (recomputed) -/
  e : Nat
  /-- `B_k/k mod ℓ²` (TB-B) -/
  bk : Nat
  /-- `G_a(x_k) mod ℓ²` -/
  gk : Nat
  /-- `G_a(x_{k+ℓ−1}) mod ℓ²` (TB-B at weight `k + ℓ − 1`) -/
  gnext : Nat
  /-- the λ-residue `(G_a(x_k) − G_a(x_{k+ℓ−1}))/ℓ mod ℓ` -/
  lam : Nat

/-- `f_k = flo + x^g`. -/
def K9Cert.F (c : K9Cert) : List Int := c.flo ++ [1]

/-- The cuspidal rank `g = deg f_k`. -/
def K9Cert.g (c : K9Cert) : Nat := c.flo.length

/-- `e_k = (1 + 2^{k−1}) mod ℓ²`, by `powMod` (exact for `k − 1 < 2^12`, checked). -/
def eisRoot (ell k : Nat) : Nat := (1 + powMod rounds 2 (k - 1) (ell ^ 2)) % (ell ^ 2)

/-- `ℓ ∥ z` at precision `ℓ²`: `ℓ ∣ z` and `ℓ² ∤ z`. -/
def v1B (ell : Nat) (z : Int) : Bool := z % (ell : Int) == 0 && z % ((ell : Int) ^ 2) != 0

/-- The guards: `ℓ` prime, `ℓ ≥ 5`, `k ≥ 3`, `(ℓ − 1) ∤ k`, `g ≥ 1`, `k − 1 < 2^12`. -/
def guardB (c : K9Cert) : Bool :=
  primeB c.ell && decide (5 ≤ c.ell) && decide (3 ≤ c.k) && c.k % (c.ell - 1) != 0 && decide (1 ≤ c.g) &&
  decide (c.k - 1 < 2 ^ rounds)

/-- `e` is `1 + 2^{k−1} mod ℓ²`. -/
def rootB (c : K9Cert) : Bool := c.e == eisRoot c.ell c.k

/-- `f ≡ (x − e)^g (mod ℓ)`. -/
def shapeB (c : K9Cert) : Bool := peqMod c.ell c.F (plin (c.e : Int) c.g)

/-- `ℓ ∥ f(e)` (precision `ℓ²`). -/
def valB (c : K9Cert) : Bool := v1B c.ell (peval c.F (c.e : Int))

/-- `ℓ ∥ B_k/k` (precision `ℓ²`). -/
def bernB (c : K9Cert) : Bool := v1B c.ell (c.bk : Int)

/-- For `g = 2`: `ℓ ∥ f₁² − 4 f₀`. -/
def discB (c : K9Cert) : Bool := c.g != 2 || v1B c.ell (coeffD c.F 1 ^ 2 - 4 * coeffD c.F 0)

/-- `G_a(x_k) ≡ −B_k/k (mod ℓ²)`. -/
def signB (c : K9Cert) : Bool := (c.gk + c.bk) % (c.ell ^ 2) == 0

/-- The λ-test: `gk − gnext ≡ ℓ · lam (mod ℓ²)`, `lam ≢ 0 (mod ℓ)`. -/
def lamB (c : K9Cert) : Bool :=
  ((c.gk : Int) - c.gnext - c.ell * c.lam) % ((c.ell : Int) ^ 2) == 0 && c.lam % c.ell != 0

/-- **The K9 checker.** -/
def k9Check (c : K9Cert) : Bool :=
  guardB c && rootB c && shapeB c && valB c && bernB c && discB c && signB c && lamB c

end GL2.HeckeZ.Exec.K9

end
