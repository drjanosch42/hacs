/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.Cert

/-!
# HeckeZ `HZExec`: the executable Prop. 5.6 certificate (LC-11, slot 25) — Mathlib-free

`docs/implementation/slot24/heckez/01-math-core.md` Prop. 5.6 (`prop:heckeZeiscert`) with the read of Prop. 5.4;
`07-lean-plan.md` §2.2 LC-11, §10 (M-HZ2). The soundness is `GL2/HeckeZ/Check/Sturm.lean`, the mathematics
`GL2/HeckeZ/Sturm.lean`.

**A Prop. 5.6 certificate** (`SturmCert`) sits on top of a cell certificate `cc : CellCert` (same `L`, same TB-χ items):
* the weight `k = cc.n + 2`; every prime `p ≤ k/12` lies in `cc.P` (Sturm's bound at level one);
* per item `p` (a K2g item of `cc`, `p` prime): `c` with `(x − λ_p) c = f` (`f` the item's squarefree part, `λ_p = 1 + p^{k−1}`;
  then `c(T_p)` is 01's `c_{cusp,p}(T_p)`), and `r` with `(x − λ_p) r = χ`, `r(λ_p) ≠ 0` (`λ_p` a simple root of TB-χ);
* `g := gcd_p c(λ_p)` (01's `g_P`), recomputed here;
* **reads** (01 Prop. 5.4 entrywise) at primes `ℓ ∣ g`: TB-mat, the matrix `M` of `T_{p₀}` on `L/ℓ^{s+1}` (in practice the slot-25
  EXPORT `DescentRecord.module(ℓ, s+1)`); the check: `c(M) ≡ 0 (mod ℓ^s)`, `c(M) ≢ 0 (mod ℓ^{s+1})`, `ℓ^{s+v} ∣ c(λ_{p₀})`,
  `ℓ^{v+1} ∤ g`;
* the Prop. 5.6 cofactor `cof56` with `g ∣ cof56 · (∏ ℓ_read)^ex`, and the combined cofactor `cof = gcd(cc.cof, cof56)`.

Nothing here is trusted: a wrong certificate can only make `sturmCheck` return `false`.
-/

namespace GL2.HeckeZ.Exec

/-- `λ_p = 1 + p^{k−1}`, the Eisenstein eigenvalue of `T_p` at level one. -/
def eisLamE (k p : Nat) : Int := 1 + (p : Int) ^ (k - 1)

/-- Evaluation of a little-endian coefficient list (Horner). -/
def peval (a : Int) : List Int → Int
  | [] => 0
  | c :: p => c + a * peval a p

/-- Primality by trial division up to a bound `b` with `n < (b + 1)²` (the bound is supplied by the certificate; `b = n`
always works for small `n`). -/
def primeB (n b : Nat) : Bool :=
  decide (2 ≤ n) && decide (n < (b + 1) * (b + 1)) &&
  (List.range (b + 1)).all (fun m => decide (m < 2) || decide (n < m * m) || n % m != 0)

/-- A Prop. 5.6 item: `θ = T_p` with `(x − λ_p) c = f` and `(x − λ_p) r = χ`. -/
structure SItem where
  p : Nat
  /-- `c` with `(x − λ_p) c = f` (the cuspidal polynomial `c_{cusp,p}`) -/
  c : List Int
  /-- `r` with `(x − λ_p) r = χ` -/
  r : List Int
  deriving Repr

/-- A read at a root (01 Prop. 5.4): `T_{p₀}` on `L/ℓ^{s+1}` (TB-mat). -/
structure SRead where
  ell : Nat
  s : Nat
  v : Nat
  /-- a bound `sq` with `ℓ < (sq + 1)²`, for the primality check -/
  sq : Nat
  /-- the Hecke index `p₀` (an item of the certificate) -/
  p : Nat
  /-- TB-mat: the matrix of `T_{p₀}` on `L/ℓ^{s+1}` (column convention), entries in `[0, ℓ^{s+1})` -/
  mat : List (List Nat)
  deriving Repr

/-- A Prop. 5.6 certificate on top of a cell certificate. -/
structure SturmCert where
  k : Nat
  items : List SItem
  reads : List SRead
  /-- Prop. 5.6 alone: `𝕋'_(ℓ) = 𝕋^fr_(ℓ)` at every prime `ℓ ∤ cof56` -/
  cof56 : Nat
  ex : Nat
  /-- with Thm. 4.1 (`cc`): at every prime `ℓ ∤ cof = gcd(cc.cof, cof56)` -/
  cof : Nat
  deriving Repr

/-- `c(λ_p)` of an item. -/
def SItem.val (k : Nat) (it : SItem) : Int := peval (eisLamE k it.p) it.c

/-- `g = gcd_p c(λ_p)`. -/
def SturmCert.g (sc : SturmCert) : Nat := gcdList (sc.items.map (SItem.val sc.k))

/-- `∏ ℓ` over the reads. -/
def SturmCert.rprod (sc : SturmCert) : Nat := (sc.reads.map SRead.ell).foldr (· * ·) 1

/-- The item check against the cell's K2g item with the same `p`. -/
def sitemOK (cc : CellCert) (k : Nat) (it : SItem) : Bool :=
  primeB it.p it.p && cc.P.contains it.p &&
  cc.items.any (fun ci => ci.p == it.p && peq (pmul [-(eisLamE k it.p), 1] it.c) ci.f &&
    peq (pmul [-(eisLamE k it.p), 1] it.r) ci.chi) &&
  peval (eisLamE k it.p) it.r != 0

/-- The read check (`b1 = rank L`, `g` the certificate's `g`); it includes the TB-χ / TB-mat cross-check
`χ_{p₀}(M) ≡ 0 (mod ℓ^{s+1})` (as `chiMatOK` for K1). -/
def sreadOK (cc : CellCert) (k g : Nat) (items : List SItem) (rd : SRead) : Bool :=
  let b1 := cc.b1
  primeB rd.ell rd.sq &&
  cc.items.any (fun ci => ci.p == rd.p &&
    mzeroB b1 (rd.ell ^ (rd.s + 1)) (pevalMat b1 (rd.ell ^ (rd.s + 1)) rd.mat ci.chi)) &&
  items.any (fun it => it.p == rd.p &&
    (let C := pevalMat b1 (rd.ell ^ (rd.s + 1)) rd.mat it.c
     allFin b1 (fun i j => ment C i j % rd.ell ^ rd.s == 0) &&
     !(allFin b1 (fun i j => ment C i j % rd.ell ^ (rd.s + 1) == 0)) &&
     SItem.val k it % ((rd.ell : Int) ^ (rd.s + rd.v)) == 0)) &&
  g % rd.ell ^ (rd.v + 1) != 0

/-- **The Prop. 5.6 checker.** -/
def sturmCheck (cc : CellCert) (sc : SturmCert) : Bool :=
  sc.k == cc.n + 2 && !sc.items.isEmpty &&
  (List.range (sc.k / 12 + 1)).all (fun m => !primeB m m || cc.P.contains m) &&
  sc.items.all (sitemOK cc sc.k) &&
  sc.reads.all (sreadOK cc sc.k sc.g sc.items) &&
  sc.g != 0 && decide (1 ≤ sc.cof56) &&
  (sc.cof56 * sc.rprod ^ sc.ex) % sc.g == 0 &&
  sc.cof == Nat.gcd cc.cof sc.cof56

end GL2.HeckeZ.Exec
