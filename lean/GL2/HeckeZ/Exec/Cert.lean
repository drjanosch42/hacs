/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.K3

/-!
# HeckeZ `HZExec`: the executable Thm. 4.1 checkers K2g and K1 (L-HZ1, M-HZ2) — Mathlib-free

`docs/implementation/slot24/heckez/07-lean-plan.md` §5 (common shape), §5.1 (K1), §5.2 (K2g), §10 (M-HZ2); 01 Thm. 4.1
(`thm:heckeZcert`). This module imports only the Mathlib-free `GL2.HeckeZ.Exec.K3` (for `allFin`, `gcdList`), so the per-cell data theorems
`cellCheck c = true` are decided by `decide +kernel` without loading Mathlib (the LV pattern). The soundness theorems
live in `GL2/HeckeZ/Check/Cert.lean`, the mathematics in `GL2/HeckeZ/Cert.lean`.

**A cell certificate** (`CellCert`, level 1, weight `k = n + 2`) consists of:
* `b1 = rank L`, `d = (b1 + 1)/2` (the rank of `𝕋^fr` by Eichler–Shimura, `hES`; never from the data);
* `P`: the Hecke indices generating `𝕋' = ℤ[T_p : p ∈ P]`;
* **K2g items**, one per `θ = T_p`: `χ` (TB-χ: the integral characteristic polynomial of the descent), `f` (monic of
  degree `d`), a cofactor `q` with `f · q = χ`, and a Bezout identity `u f + v f' = D`, `D ≠ 0`;
* **K1 certificates** (the (C3) read), one per covered modulus `q`: the dumped matrices of some `T_g` on `L/q`
  (TB-mat, `HECKE_DUMP_SAT_CERT`, the stable blocks assembled block-diagonally, reduced mod `q`), `d` monomials in them,
  `d` pivot positions and the inverse modulo `q` of the `d × d` pivot matrix: the `d` monomials are linearly independent
  modulo every prime dividing `q`;
* the cofactor `cof` and an exponent `ex` with `gcd_j D_j ∣ cof · (∏ q)^ex`;
* (slot 25, LC-11, M-HZ2 audit F3) the cross-check `χ_g(M_g) ≡ 0 (mod q)` of TB-χ against TB-mat (`chiMatOK`).

Polynomials are little-endian coefficient lists (`[a₀, a₁, …]` is `a₀ + a₁ x + …`); matrices are lists of rows.
Nothing here is trusted: a wrong certificate can only make `cellCheck` return `false`.
-/

namespace GL2.HeckeZ.Exec

/-! ### Integer polynomials as coefficient lists -/

/-- Sum of two coefficient lists. -/
def padd : List Int → List Int → List Int
  | [], q => q
  | a :: p, [] => a :: p
  | a :: p, b :: q => (a + b) :: padd p q

/-- Scalar multiple. -/
def pscale (c : Int) (p : List Int) : List Int := p.map (c * ·)

/-- Product. -/
def pmul : List Int → List Int → List Int
  | [], _ => []
  | a :: p, q => padd (pscale a q) (0 :: pmul p q)

/-- The formal derivative: coefficient `i` is `(i + 1) · a_{i+1}`. -/
def pderiv (p : List Int) : List Int :=
  (List.range (p.length - 1)).map fun i => ((i + 1 : Nat) : Int) * p.getD (i + 1) 0

/-- All coefficients vanish. -/
def pzero (p : List Int) : Bool := p.all (· == 0)

/-- Equality as polynomials (up to trailing zeros). -/
def peq (p q : List Int) : Bool := pzero (padd p (pscale (-1) q))

/-! ### Matrices modulo `m` as lists of rows -/

/-- Entry `(i, j)` of a list matrix (`0` when absent). -/
def ment (M : List (List Nat)) (i j : Nat) : Nat := (M.getD i []).getD j 0

/-- `Σ_{l < f} A_{il} B_{lj}`. -/
def mulEnt (f : Nat) (A B : List (List Nat)) (i j : Nat) : Nat :=
  ((List.range f).map fun l => ment A i l * ment B l j).sum

/-- The product mod `m` of two `f × f` list matrices. -/
def mmul (f m : Nat) (A B : List (List Nat)) : List (List Nat) :=
  (List.range f).map fun i => (List.range f).map fun j => mulEnt f A B i j % m

/-- The `f × f` identity. -/
def mone (f : Nat) : List (List Nat) :=
  (List.range f).map fun i => (List.range f).map fun j => if i = j then 1 else 0

/-! ### K2g: one `θ = T_p` -/

/-- A K2g item (07 §5.2): `θ = T_p`, TB-χ, the squarefree part `f`, `χ = f q`, `u f + v f' = D`. -/
structure K2Item where
  /-- the Hecke index `p` of `θ = T_p` -/
  p : Nat
  /-- TB-χ: `χ_ℤ(T_p)` as recorded by the descent (ed.1), little-endian -/
  chi : List Int
  /-- the squarefree part of `χ`: monic of degree `d` -/
  f : List Int
  /-- `χ = f · q` -/
  q : List Int
  /-- Bezout `u` -/
  u : List Int
  /-- Bezout `v` -/
  v : List Int
  /-- `u f + v f' = D` -/
  D : Int
  deriving Repr

/-- The K2g check of one item at rank `d`. -/
def k2OK (d : Nat) (it : K2Item) : Bool :=
  it.f.length == d + 1 && it.f.getD d 0 == 1 &&
  peq (pmul it.f it.q) it.chi &&
  peq (padd (pmul it.u it.f) (pmul it.v (pderiv it.f))) [it.D] &&
  it.D != 0

/-! ### K1: `d` monomials independent modulo `q` -/

/-- A K1 certificate (07 §5.1 at level 1, the (C3) read of 01 Thm. 4.1): matrices of `T_g` (`g ∈ gens`) on `L/q`. -/
structure K1Cert where
  /-- the modulus `q` (in practice the prime `ℓ`; soundness holds at every prime dividing `q`) -/
  q : Nat
  /-- the Hecke indices of the dumped generators -/
  gens : List Nat
  /-- TB-mat: their `b1 × b1` matrices modulo `q` (column convention), the stable blocks assembled block-diagonally -/
  mats : List (List (List Nat))
  /-- `d` monomials, each a list of positions in `gens` (the product `T_{g_{w₀}} T_{g_{w₁}} ⋯`) -/
  words : List (List Nat)
  /-- `d` pivot positions `(row, col)` -/
  pos : List (Nat × Nat)
  /-- the inverse modulo `q` of the pivot matrix `C_{km} = (word k)_{pos m}` -/
  cinv : List (List Nat)
  deriving Repr

/-- The matrix of a monomial: `M_{w₀} M_{w₁} ⋯ · I` modulo `q`. -/
def wordMat (b1 q : Nat) (mats : List (List (List Nat))) (w : List Nat) : List (List Nat) :=
  w.foldr (fun j acc => mmul b1 q (mats.getD j []) acc) (mone b1)

/-- The pivot matrix `C_{km} = (word k)_{pos m}`. -/
def pivMat (b1 : Nat) (c : K1Cert) : List (List Nat) :=
  c.words.map fun w => c.pos.map fun ij => ment (wordMat b1 c.q c.mats w) ij.1 ij.2

/-- The K1 check at `(b1, d)`. -/
def k1OK (b1 d : Nat) (c : K1Cert) : Bool :=
  decide (2 ≤ c.q) && c.mats.length == c.gens.length &&
  c.words.length == d && c.pos.length == d &&
  c.words.all (fun w => w.all (fun j => decide (j < c.gens.length))) &&
  c.pos.all (fun ij => decide (ij.1 < b1) && decide (ij.2 < b1)) &&
  allFin d (fun i j => mulEnt d (pivMat b1 c) c.cinv i j % c.q == (if i = j then 1 else 0) % c.q)

/-! ### K1m: the (C3) read from the dumped blocks plus an étale complement -/

/-- Equality of coefficient lists modulo `m`. -/
def peqMod (m : Nat) (p q : List Int) : Bool := (padd p (pscale (-1) q)).all (fun a => a % (m : Int) == 0)

/-- `p(M)` for a list matrix `M` modulo `m` (Horner: `p = a₀ + x (a₁ + x (…))`). -/
def pevalMat (f m : Nat) (M : List (List Nat)) : List Int → List (List Nat)
  | [] => (List.range f).map fun _ => (List.range f).map fun _ => 0
  | a :: p => madd f m (mscal f m a (mone f)) (mmul f m M (pevalMat f m M p))
where
  /-- sum mod `m` -/
  madd (f m : Nat) (A B : List (List Nat)) : List (List Nat) :=
    (List.range f).map fun i => (List.range f).map fun j => (ment A i j + ment B i j) % m
  /-- integer multiple mod `m` -/
  mscal (f m : Nat) (a : Int) (A : List (List Nat)) : List (List Nat) :=
    (List.range f).map fun i => (List.range f).map fun j => ((a % (m : Int)).toNat * ment A i j) % m

/-- `A^e` modulo `m`. -/
def mpow (f m : Nat) (A : List (List Nat)) : Nat → List (List Nat)
  | 0 => mone f
  | e + 1 => mmul f m A (mpow f m A e)

/-- A K1m certificate at a prime `q` (01 Prop. 2.2(1) blockwise, plus the étale part; 01 §4 "blockwise"): `L/q = B ⊕ B'`
with `B` the dumped stable blocks (`fB × fB`, TB-mat) and `B'` unknown; `r_B` monomials independent on `B` (pivots), and
`s = d − r_B` elements `θ^i f₁(θ)^m` (`θ = T_{gens[th]}`) killing `B` (`f₁(D_θ)^m ≡ 0`) and independent on `L/q` because
`f₂ ∣ χ_θ (mod q)` is squarefree, coprime to `f₁`, of degree `s`. -/
structure K1mCert where
  /-- the prime `q` -/
  q : Nat
  /-- the rank of the dumped part `B` -/
  fB : Nat
  /-- Hecke indices of the dumped generators -/
  gens : List Nat
  /-- TB-mat: their `fB × fB` matrices on `B`, modulo `q` -/
  mats : List (List (List Nat))
  /-- `r_B` monomials (positions in `gens`) -/
  words : List (List Nat)
  /-- `r_B` pivot positions in `fB × fB` -/
  pos : List (Nat × Nat)
  /-- the inverse mod `q` of the `r_B × r_B` pivot matrix -/
  cinv : List (List Nat)
  /-- the position in `gens` of `θ` -/
  th : Nat
  /-- TB-χ of `θ`: the K2 item's `χ` (checked equal by `cellCheck`) -/
  chi : List Int
  /-- `f₁` with `f₁(D_θ)^m ≡ 0` -/
  f1 : List Int
  m : Nat
  /-- `f₂`, monic of degree `s` modulo `q` -/
  f2 : List Int
  /-- `f₂ c₂ ≡ χ` -/
  c2 : List Int
  /-- `u₂ f₂ + v₂ f₂' ≡ 1` -/
  u2 : List Int
  v2 : List Int
  /-- `s₁ f₂ + t₁ f₁ ≡ 1` -/
  s1 : List Int
  t1 : List Int
  deriving Repr

/-- The dumped part of a K1m certificate, read as a K1 certificate (for `pivMat`). -/
def K1mCert.toK1 (c : K1mCert) : K1Cert :=
  { q := c.q, gens := c.gens, mats := c.mats, words := c.words, pos := c.pos, cinv := c.cinv }

/-- The K1m check at `d`. -/
def k1mOK (d : Nat) (c : K1mCert) : Bool :=
  decide (2 ≤ c.q) && c.mats.length == c.gens.length && decide (c.th < c.gens.length) &&
  c.words.length + (c.f2.length - 1) == d && c.pos.length == c.words.length &&
  c.words.all (fun w => w.all (fun j => decide (j < c.gens.length))) &&
  c.pos.all (fun ij => decide (ij.1 < c.fB) && decide (ij.2 < c.fB)) &&
  allFin c.words.length (fun i j =>
    mulEnt c.words.length (pivMat c.fB c.toK1) c.cinv i j % c.q == (if i = j then 1 else 0) % c.q) &&
  allFin c.fB (fun i j => ment (mpow c.fB c.q (pevalMat c.fB c.q (c.mats.getD c.th []) c.f1) c.m) i j == 0) &&
  decide (1 ≤ c.f2.length) && c.f2.getD (c.f2.length - 1) 0 == 1 &&
  peqMod c.q (pmul c.f2 c.c2) c.chi &&
  peqMod c.q (padd (pmul c.u2 c.f2) (pmul c.v2 (pderiv c.f2))) [1] &&
  peqMod c.q (padd (pmul c.s1 c.f2) (pmul c.t1 c.f1)) [1]

/-! ### TB-χ against TB-mat (M-HZ2 audit F3) -/

/-- `M ≡ 0 (mod m)` entrywise on `f × f`. -/
def mzeroB (f m : Nat) (M : List (List Nat)) : Bool := allFin f (fun i j => ment M i j % m == 0)

/-- **TB-χ and TB-mat agree modulo `q`** ([LP] v1.9 "Trusted C++ outputs" (2); M-HZ2 audit F3): `χ_g(M_g) ≡ 0 (mod q)` for every
dumped generator `g` that has a K2g item (for a K1m block, `M_g` is the dumped block, whose characteristic polynomial divides
`χ_g` modulo `q`). A guard against a wrong file or cell, not against a forger: a fabricated matrix can satisfy it. -/
def chiMatOK (f q : Nat) (items : List K2Item) (gens : List Nat) (mats : List (List (List Nat))) : Bool :=
  (List.range gens.length).all fun j =>
    items.all fun it => it.p != gens.getD j 0 || mzeroB f q (pevalMat f q (mats.getD j []) it.chi)

/-! ### The cell -/

/-- A level-1 cell certificate for 01 Thm. 4.1 (M-HZ2). -/
structure CellCert where
  /-- the weight is `k = n + 2` -/
  n : Nat
  /-- `rank L = rank H¹(SL₂(ℤ); Sym^n ℤ)/tors` -/
  b1 : Nat
  /-- `d = (b1 + 1)/2 = rank 𝕋^fr` (by `hES`) -/
  d : Nat
  /-- the generator set of `𝕋'` -/
  P : List Nat
  /-- the K2g items -/
  items : List K2Item
  /-- the K1 certificates (all of `L/q` dumped) -/
  k1 : List K1Cert
  /-- the K1m certificates (dumped blocks plus an étale complement) -/
  k1m : List K1mCert
  /-- the cofactor: the conclusion holds at every prime not dividing it -/
  cof : Nat
  /-- an exponent with `gcd_j D_j ∣ cof · (∏ q)^ex` -/
  ex : Nat
  deriving Repr

/-- `∏ q` over the K1 and K1m certificates. -/
def CellCert.qprod (c : CellCert) : Nat := (c.k1.map K1Cert.q ++ c.k1m.map K1mCert.q).foldr (· * ·) 1

/-- `gcd_j D_j`. -/
def CellCert.G (c : CellCert) : Nat := gcdList (c.items.map K2Item.D)

/-- **The cell checker.** -/
def cellCheck (c : CellCert) : Bool :=
  c.d == (c.b1 + 1) / 2 && decide (1 ≤ c.cof) &&
  c.items.all (k2OK c.d) && c.items.all (fun it => c.P.contains it.p) &&
  c.k1.all (k1OK c.b1 c.d) && c.k1.all (fun k => k.gens.all (fun g => c.P.contains g)) &&
  c.k1m.all (k1mOK c.d) && c.k1m.all (fun k => k.gens.all (fun g => c.P.contains g)) &&
  c.k1m.all (fun k => decide (k.fB ≤ c.b1) &&
    c.items.any (fun it => it.p == k.gens.getD k.th 0 && it.chi == k.chi)) &&
  !c.items.isEmpty &&
  (c.cof * c.qprod ^ c.ex) % c.G == 0 &&
  c.k1.all (fun k => chiMatOK c.b1 k.q c.items k.gens k.mats) &&
  c.k1m.all (fun k => chiMatOK k.fB k.q c.items k.gens k.mats)

end GL2.HeckeZ.Exec
