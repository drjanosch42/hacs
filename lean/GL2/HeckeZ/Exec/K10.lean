/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.SmithPKExec

/-!
# HeckeZ `HZExec`: the executable K10 and K10′ checkers (L-K10, M-HZ4a) — Mathlib-free

`docs/implementation/slot24/heckez/07-lean-plan.md` §5.10 (K10, K10′, the A7-M1 note v0.4.1), [C] `prop:heckeZeiscert`(3) (the
Mazur route, prime level `N`, weight 2), `slot25/H3_report.md` / `H3_referee.md` note 5 (the valuation check is the certificate,
goodness is only how `q` is chosen). Brief `slot25/LK10_brief.md`. This module imports only the Mathlib-free
`GL2.L23.SmithPKExec` (for `powMod`), so the per-cell theorems `k10pCheck c = true`, `merelCheck c = true` are decided by
`decide +kernel` without Mathlib (the LV pattern). Soundness: `GL2/HeckeZ/Check/K10.lean`.

**Polynomials** are lists of integer coefficients, lowest degree first, computed **exactly** over `ℤ` (no reduction inside
the arithmetic); congruences are checked coefficientwise at the end (`peqMod`). So the soundness proofs are about `ℤ[X]`.

**K10′ certificate** (one Mazur cell `(N, ℓ)` and the recorded prime `q`):
* `h`: the record's `c_q ∈ ℤ[x]` (HZ-RAT / H3: `χ_ℤ(T_q) = (x − 1 − q) · c_q²`, so `c_q = h_q` is the characteristic polynomial of
  `T_q` on `S₂(Γ₀(N))`; trust input TB-χ), exact integers;
* `flo`: the lower coefficients of the Hensel factor `F = flo + x^e` (variable `x`, residues mod `ℓ^K`), `e = flo.length`;
* `H`: the cofactor, residues mod `ℓ^K`.

The checks: `N`, `ℓ`, `q` prime (trial division), `5 ≤ N`, `5 ≤ ℓ`, `q ≠ N`, `ℓ ∣ N − 1`; `ℓ^v ∥ ν := (N−1)/gcd(N−1, 12)`
(`= num((N−1)/12)`) and `v < K`; **goodness** (the choice rule): `q mod ℓ ≠ 1` and `q^{(N−1)/ℓ} mod N ≠ 1`; `F ≡ (x − a)^e (mod ℓ)` with
`a = 1 + q`; `h ≡ F · H (mod ℓ^K)`; `H(a) ≢ 0 (mod ℓ)`; and **the valuation check** `ℓ^v ∣ F(a)`, `ℓ^{v+1} ∤ F(a)` (A7-M1).

**K10 certificate** (Merel's criterion, [C] LN5; `orc:`, no soundness theorem): `N`, `ℓ`, the residue `P = ∏_{i ≤ (N−1)/2} i^i mod N`,
and the cell's rank `e`. The check recomputes `P` and compares `[P^{(N−1)/ℓ} ≡ 1 (mod N)]` with `[e ≥ 2]`.
-/

noncomputable section

namespace GL2.HeckeZ.Exec.K10

open GL2.L23.SmithPK (powMod)

/-! ### Integer polynomials as coefficient lists -/

/-- Coefficientwise sum. -/
def padd : List Int → List Int → List Int
  | [], b => b
  | a, [] => a
  | x :: a, y :: b => (x + y) :: padd a b

/-- Product (schoolbook). -/
def pmul : List Int → List Int → List Int
  | [], _ => []
  | x :: a, b => padd (b.map (x * ·)) (0 :: pmul a b)

/-- Horner evaluation. -/
def peval (p : List Int) (t : Int) : Int := p.foldr (fun c acc => c + t * acc) 0

/-- `(x − a)^e`. -/
def plin (a : Int) : Nat → List Int
  | 0 => [1]
  | e + 1 => pmul [-a, 1] (plin a e)

/-- Coefficient `i` (`0` beyond the list). -/
def coeffD (p : List Int) (i : Nat) : Int := p.getD i 0

/-- `a ≡ b` coefficientwise modulo `m`. -/
def peqMod (m : Nat) (a b : List Int) : Bool :=
  (List.range (max a.length b.length)).all fun i => (coeffD a i - coeffD b i) % (m : Int) == 0

/-- Trial-division primality (`q < 2^12` in practice; `O(q)` kernel steps). -/
def primeB (q : Nat) : Bool := decide (2 ≤ q) && (List.range q).all fun d => d < 2 || q % d != 0

/-- Mazur's `n = num((N − 1)/12)`, as `(N − 1)/gcd(N − 1, 12)`. -/
def nuNat (N : Nat) : Nat := (N - 1) / Nat.gcd (N - 1) 12

/-- The rounds of `powMod` used here (exact for exponents `< 2^12`; every cell has `N < 2^12`, checked). -/
abbrev rounds : Nat := 12

/-! ### K10′ -/

/-- A K10′ certificate (07 §5.10 with A7-M1). -/
structure K10pCert where
  /-- the prime level `N` -/
  N : Nat
  /-- the Eisenstein prime `ℓ` (Mazur's `p`) -/
  ell : Nat
  /-- the recorded prime `q` (Mazur's good `ℓ`) -/
  q : Nat
  /-- the precision `K` of `F`, `H` -/
  K : Nat
  /-- `v = v_ℓ(ν)` -/
  v : Nat
  /-- `h = c_q` (exact integers, lowest degree first; TB-χ) -/
  h : List Int
  /-- the lower coefficients of `F` (residues mod `ℓ^K`); `F = flo + x^e`, `e = flo.length` -/
  flo : List Int
  /-- the cofactor `H` (residues mod `ℓ^K`) -/
  H : List Int


/-- `F = flo + x^e`. -/
def K10pCert.F (c : K10pCert) : List Int := c.flo ++ [1]

/-- The rank `e = deg F`. -/
def K10pCert.e (c : K10pCert) : Nat := c.flo.length

/-- `a = 1 + q`. -/
def K10pCert.a (c : K10pCert) : Int := 1 + (c.q : Int)

/-- Goodness in Mazur's sense (case (a), `ℓ` odd): `q ≢ 1 (mod ℓ)` and `q^{(N−1)/ℓ} ≢ 1 (mod N)`. The choice rule for `q`. -/
def goodB (N ell q : Nat) : Bool :=
  q % ell != 1 && powMod rounds q ((N - 1) / ell) N != 1

/-- The arithmetic guards of a Mazur cell. -/
def cellB (c : K10pCert) : Bool :=
  primeB c.N && primeB c.ell && primeB c.q && decide (5 ≤ c.N) && decide (5 ≤ c.ell) && c.q != c.N &&
  (c.N - 1) % c.ell == 0 && decide ((c.N - 1) / c.ell < 2 ^ rounds) &&
  nuNat c.N % c.ell ^ c.v == 0 && nuNat c.N % c.ell ^ (c.v + 1) != 0 && decide (1 ≤ c.v) && decide (c.v + 1 ≤ c.K)

/-- The Hensel factorisation at `a = 1 + q`: `F ≡ (x − a)^e (mod ℓ)`, `h ≡ F·H (mod ℓ^K)`, `H(a) ≢ 0 (mod ℓ)`. -/
def henselB (c : K10pCert) : Bool :=
  peqMod c.ell c.F (plin c.a c.e) && peqMod (c.ell ^ c.K) c.h (pmul c.F c.H) && peval c.H c.a % (c.ell : Int) != 0

/-- **The valuation check** (A7-M1): `ℓ^v ∣ F(a)` and `ℓ^{v+1} ∤ F(a)`. -/
def valB (c : K10pCert) : Bool :=
  peval c.F c.a % ((c.ell : Int) ^ c.v) == 0 && peval c.F c.a % ((c.ell : Int) ^ (c.v + 1)) != 0

/-- **The K10′ checker**: the guards, goodness, the Hensel factorisation, the valuation check. -/
def k10pCheck (c : K10pCert) : Bool := cellB c && goodB c.N c.ell c.q && henselB c && valB c

/-! ### K10 (Merel's criterion, `orc:`) -/

/-- `∏_{i ≤ k} i^i mod N` (`i` from `1`). -/
def merelProd (N : Nat) : Nat → Nat
  | 0 => 1 % N
  | k + 1 => merelProd N k * powMod rounds (k + 1) (k + 1) N % N

/-- Merel's residue `P = ∏_{i ≤ (N−1)/2} i^i mod N`. -/
def merelP (N : Nat) : Nat := merelProd N ((N - 1) / 2)

/-- A K10 certificate: `(N, ℓ, P, e)`. -/
structure K10Cert where
  N : Nat
  ell : Nat
  P : Nat
  /-- the rank `e` of the cell (from K10′: `deg F`) -/
  e : Nat


/-- **The K10 checker** ([C] LN5): `P` recomputed, and `[P^{(N−1)/ℓ} ≡ 1] = [e ≥ 2]`. -/
def merelCheck (c : K10Cert) : Bool :=
  decide (c.N < 2 ^ rounds) && c.P == merelP c.N && (c.N - 1) % c.ell == 0 &&
  (powMod rounds c.P ((c.N - 1) / c.ell) c.N == 1) == decide (2 ≤ c.e)

end GL2.HeckeZ.Exec.K10

end
