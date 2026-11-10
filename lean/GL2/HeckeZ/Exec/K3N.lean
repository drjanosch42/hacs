/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.Cert
import GL2.HeckeZ.Exec.K10

/-!
# HeckeZ `HZExec`: the level-`N` K3 checker K3N (L-HZ3, M-HZ4b, partly) — Mathlib-free

`docs/implementation/slot24/heckez/07-lean-plan.md` §5.3 (K3) at level `N`, anemic, in the **two-system** case: the anemic
algebra `𝕋^{an,fr}` has exactly two systems `λ`, `μ` (rank `2`, `hES_N_an`), separated by one operator `θ = T_p`, `p ∤ N`.
This module imports only Lean core and the Mathlib-free `Exec.Cert` (`allFin`, `entry`, `resid`, `pevalMat`, `mzeroB`) and
`Exec.K10` (`primeB`), so the per-cell theorems `k3nCheck c = true` are decided by `decide +kernel` without Mathlib.
Soundness and the bridge to the congruence module: `GL2/HeckeZ/Check/K3N.lean`.

**What a K3N certificate is.** One operator `T_p` (`p` prime, `p ∤ N`) with the values `λ = λ(T_p) = 1 + p^{k−1}` (checked: the
system with this value at `T_p` is called `λ`; that it is the system of an Eisenstein series is *not* part of any statement) and
`μ = μ(T_p)` (the other root of the recorded `χ_ℤ(T_p)`, TB-χ), and a list of **reads** at pairwise distinct primes `ℓ`. A read
is 01 Prop. 5.4 at a root, `r` arbitrary, read entrywise (07 §5.3 checks 2–5): the dumped matrix `D` of `T_p` on `L/ℓ^K` (TB-mat,
`f × f`, column convention, the free block of the slot-25 EXPORT), a claimed valuation `v < K` of `c(T_p) = T_p − μ` (every entry
divisible by `ℓ^v`, the witness entry not by `ℓ^{v+1}`), and a claimed exponent `e` with `ℓ^{v+e} ∣ λ − μ`. The read gives the
lower bound `ℓ^e`; the product `G = ∏ ℓ^e` is the certified lower bound, `|λ − μ|` the upper bound (01 Prop. 5.3), and the
certificate is **complete** iff `G = |λ − μ|`. A read with `e = 0` records that the read reaches nothing at `ℓ` (T4 at `ℓ = 2`).

The TB-χ / TB-mat cross-check `chiReadsOK` (`χ(D) ≡ 0 mod ℓ^K` for the recorded `χ_ℤ(T_p)`) is a guard against a wrong file or
cell, not against a forger, and no soundness theorem uses it.
-/

namespace GL2.HeckeZ.Exec.K3N

open GL2.HeckeZ.Exec

/-- One read at a prime `ℓ` (07 §5.3). -/
structure K3Read where
  /-- the prime `ℓ` -/
  ell : Nat
  /-- the precision `K` of the dump -/
  K : Nat
  /-- the rank `f` of `L` -/
  f : Nat
  /-- the dumped `T_p` on `L/ℓ^K`, `f × f`, column convention, entries mod `ℓ^K` (TB-mat) -/
  dump : List (List Nat)
  /-- the claimed valuation `v` of `c(T_p) = T_p − μ` -/
  v : Nat
  /-- the witness entry `(i, j)` of valuation exactly `v` -/
  wit : Nat × Nat
  /-- the claimed lower-bound exponent: `ℓ^{v+e} ∣ λ − μ` -/
  e : Nat
  deriving Repr

/-- A K3N certificate (level `N`, anemic, two systems, one operator). -/
structure K3NCert where
  /-- the level `N` (interpretive: `p ∤ N` is checked) -/
  N : Nat
  /-- the weight `k = n + 2` -/
  k : Nat
  /-- the operator `T_p` -/
  p : Nat
  /-- `λ(T_p)`, checked `= 1 + p^{k−1}` -/
  lam : Int
  /-- `μ(T_p)` (TB-χ) -/
  mu : Int
  /-- the reads, at pairwise distinct primes -/
  reads : List K3Read
  /-- the recorded `χ_ℤ(T_p)`, coefficients low → high (TB-χ; used only by the cross-check `chiReadsOK`) -/
  chi : List Int
  deriving Repr

/-- The modulus `ℓ^K` of a read. -/
def K3Read.m (r : K3Read) : Nat := r.ell ^ r.K

/-- Entry `(i, j)` of `c(T_p) = D − μ·I` modulo `ℓ^K`, in `[0, ℓ^K)`. -/
def rEntry (r : K3Read) (mu : Int) (i j : Nat) : Nat :=
  (entry r.dump i j + (if i = j then resid r.m (-mu) else 0)) % r.m

/-- The shape of a read: `f × f`, a prime `ℓ`, `v < K`, the witness inside. -/
def shapeB (r : K3Read) : Bool :=
  K10.primeB r.ell && decide (r.v + 1 ≤ r.K) &&
  r.dump.length == r.f && r.dump.all (fun row => row.length == r.f) &&
  decide (r.wit.1 < r.f) && decide (r.wit.2 < r.f)

/-- Every entry of `c(T_p)` is divisible by `ℓ^v`. -/
def allDivB (r : K3Read) (mu : Int) : Bool := allFin r.f (fun i j => rEntry r mu i j % r.ell ^ r.v == 0)

/-- The witness entry is not divisible by `ℓ^{v+1}`. -/
def witB (r : K3Read) (mu : Int) : Bool := rEntry r mu r.wit.1 r.wit.2 % r.ell ^ (r.v + 1) != 0

/-- `ℓ^{v+e} ∣ λ − μ` (01's check 5). -/
def gapB (r : K3Read) (lam mu : Int) : Bool := (lam - mu) % ((r.ell : Int) ^ (r.v + r.e)) == 0

/-- One read. -/
def readOK (r : K3Read) (lam mu : Int) : Bool := shapeB r && allDivB r mu && witB r mu && gapB r lam mu

/-- No repetitions in a list of naturals. -/
def nodupB : List Nat → Bool
  | [] => true
  | a :: l => !(l.contains a) && nodupB l

/-- The operator guard: `p` prime, `p ∤ N`, `N ≥ 1`, `λ = 1 + p^{k−1}`, `λ ≠ μ`. -/
def opB (c : K3NCert) : Bool :=
  K10.primeB c.p && decide (1 ≤ c.N) && c.p % c.N != 0 && c.lam == 1 + (c.p : Int) ^ (c.k - 1) && c.lam != c.mu

/-- **The K3N checker.** -/
def k3nCheck (c : K3NCert) : Bool :=
  opB c && nodupB (c.reads.map K3Read.ell) && c.reads.all (fun r => readOK r c.lam c.mu)

/-- The certified lower bound `G = ∏ ℓ^e`. -/
def K3NCert.G (c : K3NCert) : Nat := (c.reads.map fun r => r.ell ^ r.e).foldr (· * ·) 1

/-- The upper bound `|λ − μ|`. -/
def K3NCert.gUp (c : K3NCert) : Nat := (c.lam - c.mu).natAbs

/-- The certificate closes: lower bound = upper bound. -/
def K3NCert.complete (c : K3NCert) : Bool := c.G == c.gUp

/-- **The TB-χ / TB-mat cross-check** (guard only): `χ(D) ≡ 0 mod ℓ^K` for every read. -/
def chiReadsOK (c : K3NCert) : Bool := c.reads.all fun r => mzeroB r.f r.m (pevalMat r.f r.m r.dump c.chi)

end GL2.HeckeZ.Exec.K3N
