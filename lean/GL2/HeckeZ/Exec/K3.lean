/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/

/-!
# HeckeZ `HZExec`: the executable K3 checker (L-HZ0, M-HZ1) — Mathlib-free

`docs/implementation/slot24/heckez/07-lean-plan.md` §5 ("common shape"), §5.3 (K3, the Eisenstein certificate of
01 Props. 5.3 / 5.4 at `r = 1`), §7 (T1). This module imports **nothing** (Lean core only), so the per-cell data
theorems `k3Check c = true` are decided by `decide +kernel` without loading Mathlib (the LV pattern). The soundness
theorem `GL2.HeckeZ.K3.k3Check_sound` and the bridge to the congruence module live in `GL2/HeckeZ/Check/K3.lean`.

**What a K3 certificate is** (the two-system, rational case, `r = 1`):
* `sys`: one entry `(p, λ_p, μ_p)` per Hecke operator `T_p`, `p ∈ P`: the eigenvalue `λ_p` of the rational Eisenstein
  system and `μ_p` of the rest, which is **one rational system** here (`c_{S',p} = x − μ_p`, the T1 shape:
  `χ_ℤ(T_p) = (x − μ_p)² (x − λ_p)`, TB-χ). The check `λ_p = 1 + p^{k−1}` ties the Eisenstein column to `σ_{k−1}`.
* `gP = gcd_p (λ_p − μ_p)` (the upper bound of 01 Prop. 5.3) and the claim `gP = ℓ^e`, so one read at `ℓ` closes it.
* The at-root read (01 Prop. 5.4, `r = 1`) at `p = readP`: the dumped matrix of `T_p` on `L/ℓ^K` (TB-mat, the
  `HECKE_DUMP_SAT_CERT` block, column convention), the claimed valuation `v` of `c(T_p) = T_p − μ_p`, and a witness
  entry: every entry of `c(T_p) mod ℓ^K` is divisible by `ℓ^v`, the witness entry is not divisible by `ℓ^{v+1}`, `v < K`,
  and `ℓ^{v+e} ∣ c(λ_p) = λ_p − μ_p` (01's check 5, `v_ℓ c(λ) − v ≥ v_ℓ(g_P)`).

Primality of `ℓ` is not checked here (it is a hypothesis of the soundness theorem, discharged per cell by `norm_num`).
-/

namespace GL2.HeckeZ.Exec

/-- Entry `(i, j)` of a list matrix (`0` when absent). -/
def entry (M : List (List Nat)) (i j : Nat) : Nat := (M.getD i []).getD j 0

/-- `∀ i j < f, P i j`. -/
def allFin (f : Nat) (P : Nat → Nat → Bool) : Bool :=
  (List.range f).all fun i => (List.range f).all fun j => P i j

/-- `gcd` of a list of integers (as a natural number), `0` for the empty list. -/
def gcdList : List Int → Nat
  | [] => 0
  | a :: l => Nat.gcd a.natAbs (gcdList l)

/-- The residue of an integer in `[0, m)`, as a natural number. -/
def resid (m : Nat) (a : Int) : Nat := (a % (m : Int)).toNat

/-- The first entry of `sys` for the operator `p`. -/
def lookup (sys : List (Nat × Int × Int)) (p : Nat) : Option (Nat × Int × Int) :=
  sys.find? fun s => s.1 == p

/-- A K3 certificate (07 §5.3), two-system rational case, `r = 1`. -/
structure K3Cert where
  /-- the weight `k` (level one) -/
  k : Nat
  /-- the prime `ℓ` of the read -/
  ell : Nat
  /-- the precision `K` of the dump -/
  K : Nat
  /-- the rank `f` of the dumped block -/
  f : Nat
  /-- `(p, λ_p, μ_p)`, one per `p ∈ P` -/
  sys : List (Nat × Int × Int)
  /-- `g_P = gcd_p (λ_p − μ_p)` -/
  gP : Nat
  /-- `g_P = ℓ^e` -/
  e : Nat
  /-- the operator `T_p` of the at-root read -/
  readP : Nat
  /-- the dumped `T_{readP}` on `L/ℓ^K`, `f × f`, column convention, entries mod `ℓ^K` (TB-mat) -/
  dump : List (List Nat)
  /-- the claimed valuation `v` of `c(T_{readP}) = T_{readP} − μ` -/
  v : Nat
  /-- the witness entry `(i, j)` of valuation exactly `v` -/
  wit : Nat × Nat
  deriving Repr

/-- The modulus `ℓ^K`. -/
def K3Cert.m (c : K3Cert) : Nat := c.ell ^ c.K

/-- Entry `(i, j)` of `c(T_p) = T_p − μ·I` modulo `ℓ^K`, as a natural number `< ℓ^K`. -/
def cEntry (c : K3Cert) (mu : Int) (i j : Nat) : Nat :=
  (entry c.dump i j + (if i = j then resid c.m (-mu) else 0)) % c.m

/-- The read at `readP`, given its `(λ, μ)`. -/
def readOK (c : K3Cert) (lam mu : Int) : Bool :=
  allFin c.f (fun i j => cEntry c mu i j % c.ell ^ c.v == 0) &&
  cEntry c mu c.wit.1 c.wit.2 % c.ell ^ (c.v + 1) != 0 &&
  decide (c.wit.1 < c.f) && decide (c.wit.2 < c.f) &&
  (lam - mu) % ((c.ell : Int) ^ (c.v + c.e)) == 0

/-- **The K3 checker.** -/
def k3Check (c : K3Cert) : Bool :=
  decide (2 ≤ c.ell) && decide (1 ≤ c.e) && decide (c.v + 1 ≤ c.K) &&
  c.dump.length == c.f && c.dump.all (fun r => r.length == c.f) &&
  c.sys.all (fun s => s.2.1 == 1 + (s.1 : Int) ^ (c.k - 1)) &&
  c.gP == gcdList (c.sys.map fun s => s.2.1 - s.2.2) &&
  c.gP == c.ell ^ c.e &&
  match lookup c.sys c.readP with
  | some s => readOK c s.2.1 s.2.2
  | none => false

/-! ### The TB-χ / TB-mat cross-check (M-HZ1 audit m1; 06 §5 Q10, 07 §8)

Not part of `k3Check` and not used by the soundness theorem: it is the in-Lean guard that the two trust inputs agree.
The dumped matrix `D` of `T_{readP}` (TB-mat) must have, modulo `ℓ^K`, the characteristic polynomial recorded by the
descent (TB-χ). In the T1 shape `χ_ℤ(T_p) = (x − μ)² (x − λ) = x³ − (λ + 2μ) x² + (2λμ + μ²) x − λμ²`, and for a
`3 × 3` matrix `χ(D) = x³ − tr D · x² + c₂(D) · x − det D` (`c₂` = the sum of the principal `2 × 2` minors). -/

/-- Entry `(i, j)` of a list matrix, as an integer. -/
def entZ (M : List (List Nat)) (i j : Nat) : Int := (entry M i j : Int)

/-- The trace of a `3 × 3` list matrix. -/
def tr3 (M : List (List Nat)) : Int := entZ M 0 0 + entZ M 1 1 + entZ M 2 2

/-- The sum of the principal `2 × 2` minors of a `3 × 3` list matrix. -/
def c2of3 (M : List (List Nat)) : Int :=
  (entZ M 0 0 * entZ M 1 1 - entZ M 0 1 * entZ M 1 0) +
  (entZ M 0 0 * entZ M 2 2 - entZ M 0 2 * entZ M 2 0) +
  (entZ M 1 1 * entZ M 2 2 - entZ M 1 2 * entZ M 2 1)

/-- The determinant of a `3 × 3` list matrix (cofactor expansion along row `0`). -/
def det3 (M : List (List Nat)) : Int :=
  entZ M 0 0 * (entZ M 1 1 * entZ M 2 2 - entZ M 1 2 * entZ M 2 1) -
  entZ M 0 1 * (entZ M 1 0 * entZ M 2 2 - entZ M 1 2 * entZ M 2 0) +
  entZ M 0 2 * (entZ M 1 0 * entZ M 2 1 - entZ M 1 1 * entZ M 2 0)

/-- `χ(M) ≡ x³ − a₁ x² + a₂ x − a₃ (mod m)` for a `3 × 3` list matrix `M`. -/
def charpoly3OK (m : Nat) (M : List (List Nat)) (a1 a2 a3 : Int) : Bool :=
  M.length == 3 && M.all (fun r => r.length == 3) &&
  (tr3 M - a1) % (m : Int) == 0 && (c2of3 M - a2) % (m : Int) == 0 && (det3 M - a3) % (m : Int) == 0

/-- **The TB-χ cross-check** (T1 shape, `f = 3`): the dumped `T_{readP}` has characteristic polynomial
`(x − μ)² (x − λ)` modulo `ℓ^K`, with `(λ, μ)` the certificate's entry for `readP`. -/
def chiCheck (c : K3Cert) : Bool :=
  c.f == 3 &&
  match lookup c.sys c.readP with
  | some s => charpoly3OK c.m c.dump (s.2.1 + 2 * s.2.2) (2 * s.2.1 * s.2.2 + s.2.2 ^ 2) (s.2.1 * s.2.2 ^ 2)
  | none => false

end GL2.HeckeZ.Exec
