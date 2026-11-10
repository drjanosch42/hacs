/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.Cert

/-!
# HeckeZ `HZExec`: the Bockstein / Tor checkers K5, K6, K7 and the consistency check K8 (L-HZ2, M-HZ3) — Mathlib-free

`docs/implementation/slot24/heckez/07-lean-plan.md` §5.5–§5.8, §10 (M-HZ3); `03-bockstein-tor.md` §8.2 (B2, B7, B12, B13, B10).
This module imports only the Mathlib-free `GL2.HeckeZ.Exec.Cert` (for `padd`, `pmul`, `peqMod`), so the per-cell data theorems
`k5Check c = true`, `k6Check c = true`, `k7Check c = true`, `k8Check c = true` are decided by `decide +kernel` without loading
Mathlib (the LV pattern). The soundness theorems live in `GL2/HeckeZ/Check/Bock.lean`, the mathematics in `GL2/HeckeZ/Bock/`.

Conventions: polynomials are little-endian coefficient lists of integers; a **coordinate vector** `v : List Int` stands for
the element `Σ_i v_i g_i` of a module with chosen elements `g_i`; a matrix is a list of rows, read in the **column convention**
of the C++ export (`DescentRecord.module(ell, K)`, slot-25 EXPORT): the image of generator `j` has coordinate `i` equal to
entry `(i, j)`. Nothing here is trusted: a wrong certificate can only make a checker return `false`.
-/

namespace GL2.HeckeZ.Exec

/-! ### Coordinate vectors -/

/-- Entry `(i, j)` of an integer list matrix (`0` when absent). -/
def ientry (M : List (List Int)) (i j : Nat) : Int := (M.getD i []).getD j 0

/-- `M v` in the column convention, on `t` coordinates: coordinate `i` is `Σ_{j<t} M_{ij} v_j`. -/
def matVec (t : Nat) (M : List (List Int)) (v : List Int) : List Int :=
  (List.range t).map fun i => ((List.range t).map fun j => ientry M i j * v.getD j 0).sum

/-- `v + w` coordinatewise, on `t` coordinates. -/
def vadd (t : Nat) (v w : List Int) : List Int := (List.range t).map fun i => v.getD i 0 + w.getD i 0

/-- `a v`, on `t` coordinates. -/
def vscale (t : Nat) (a : Int) (v : List Int) : List Int := (List.range t).map fun i => a * v.getD i 0

/-- Coordinate `i` reduced modulo `ℓ^{σ_i}` (harmless: `ℓ^{σ_i} g_i = 0`). -/
def vred (t ell : Nat) (sig : List Nat) (v : List Int) : List Int :=
  (List.range t).map fun i => v.getD i 0 % ((ell ^ sig.getD i 0 : Nat) : Int)

/-- `P(M) v` by Horner (`P = a₀ + x (a₁ + x (…))`), reducing after each step. -/
def polyVec (t ell : Nat) (sig : List Nat) (M : List (List Int)) : List Int → List Int → List Int
  | [], _ => (List.range t).map fun _ => 0
  | a :: P, v => vred t ell sig (vadd t (vscale t a v) (matVec t M (polyVec t ell sig M P v)))

/-- `P(M)^m v`. -/
def polyPowVec (t ell : Nat) (sig : List Nat) (M : List (List Int)) (P : List Int) : Nat → List Int → List Int
  | 0, v => v
  | m + 1, v => polyVec t ell sig M P (polyPowVec t ell sig M P m v)

/-- The unit vector `e_j` on `t` coordinates. -/
def unitVec (t j : Nat) : List Int := (List.range t).map fun i => if i = j then 1 else 0

/-- Every coordinate `i` of `v` is divisible by `ℓ^{σ_i}` (so `Σ v_i g_i = 0`). -/
def vanishes (t ell : Nat) (sig : List Nat) (v : List Int) : Bool :=
  (List.range t).all fun i => v.getD i 0 % ((ell ^ sig.getD i 0 : Nat) : Int) == 0

/-! ### K5: universal coefficients (B2(1) at `K = 1`) -/

/-- A K5 certificate at `(N, n, ℓ)`, `K = 1` (03 §8.2, A3-N5): the free ranks `b_q`, the torsion multiplicities `μ_q(j)`
(`mu[q][j-1]` = the number of summands `ℤ/ℓ^j` of `H^q ⊗ ℤ_(ℓ)`), `q = 0 … 3`, from the descent record, and the emitted
`h^q = dim_{𝔽_ℓ} H^q(Γ; Sym^n 𝔽_ℓ)` (the Lean-kernel tables `lean_dims_l{2,3}.tsv`). Periodicity `t_4 := t_2`
([C, thm:descentcert](6)) supplies the `q = 3` Tor term. -/
structure K5Cert where
  /-- the prime `ℓ` -/
  ell : Nat
  /-- the weight parameter `n` (for the record) -/
  n : Nat
  /-- `b_0, …, b_3` -/
  b : List Nat
  /-- `μ_q(j)`, `q = 0 … 3`, `j = 1, 2, …` -/
  mu : List (List Nat)
  /-- the emitted `h^0, …, h^3` -/
  h : List Nat
deriving Repr

/-- `t_q = Σ_j μ_q(j)`, the number of `ℓ`-primary cyclic summands, with `t_4 := t_2`. -/
def K5Cert.t (c : K5Cert) (q : Nat) : Nat := (c.mu.getD (if q = 4 then 2 else q) []).sum

/-- The B2(1) prediction `h^q = b_q + t_q + t_{q+1}`. -/
def K5Cert.pred (c : K5Cert) (q : Nat) : Nat := c.b.getD q 0 + c.t q + c.t (q + 1)

/-- **K5**: the emitted `h^q` equal the universal-coefficient prediction for `q = 0 … 3`. -/
def k5Check (c : K5Cert) : Bool :=
  c.b.length == 4 && c.mu.length == 4 && c.h.length == 4 &&
  (List.range 4).all fun q => c.h.getD q 0 == c.pred q

/-! ### K6: separation by one operator (B7) -/

/-- A K6 certificate (03 §3.1, §8.2; A3-m4): at `ℓ`, for `θ = T_p`,
* `chi`: **TB-χ**, the integral characteristic polynomial of `T_p` on `L = H¹/tors` (exact coefficients);
* `sig`, `C`: **TB-mat**, the exponents `σ_i` of the export's torsion generators `g_i` of `tors_ℓ` and the matrix of `T_p`
  on them (`T_p g_j = Σ_i C_{ij} g_i`);
* `P` (an integer lift of `∏_j P_j`), `m`: `P(T_p)^m` kills every `g_j` (checked on the coordinates);
* `u`, `v`: a Bezout pair over `𝔽_ℓ`, `u χ + v P ≡ 1 (mod ℓ)`. -/
structure K6Cert where
  /-- the prime `ℓ` -/
  ell : Nat
  /-- the Hecke index `p` of `θ = T_p` -/
  p : Nat
  /-- TB-χ: `χ_ℤ(T_p | L)` -/
  chi : List Int
  /-- the torsion exponents `σ_i` -/
  sig : List Nat
  /-- TB-mat: `T_p` on the torsion generators -/
  C : List (List Int)
  /-- the torsion polynomial (a lift of `∏ P_j`) -/
  P : List Int
  /-- the nilpotency exponent -/
  m : Nat
  /-- Bezout `u` -/
  u : List Int
  /-- Bezout `v` -/
  v : List Int
deriving Repr

/-- **K6**: the Bezout identity modulo `ℓ`, and `P(C)^m e_j ≡ 0` (row `i` modulo `ℓ^{σ_i}`) for every torsion generator. -/
def k6Check (c : K6Cert) : Bool :=
  decide (2 ≤ c.ell) &&
  peqMod c.ell (padd (pmul c.u c.chi) (pmul c.v c.P)) [1] &&
  (List.range c.sig.length).all fun j =>
    vanishes c.sig.length c.ell c.sig (polyPowVec c.sig.length c.ell c.sig c.C c.P c.m (unitVec c.sig.length j))

/-! ### K7: split / Hom bound on `H¹_ℓ/ℓ^e` with cyclic torsion (B12, B13) -/

/-- A K7 certificate (03 §8.2; 07 §5.7), for `tors_ℓ = ℤ/ℓ^e` cyclic with generator `g` and free generators `f_0 … f_{r-1}`
of `H¹/ℓ^e` (the export at `K ≥ e`, reduced modulo `q = ℓ^e`; **TB-mat**):
`T_p f_j ≡ Σ_i B_p[i][j] f_i + β_p[j] g`, `T_p g = c_p g`, for `p ∈ P` (index `k`).
The checked claims:
* `split`: a section `h` with `Σ_i h_i B_p[i][j] − c_p h_j ≡ β_p[j]` (the graph of `h` is a `𝕋'`-stable complement);
* the Hom bound: with the system `A[i][(k,j)] = B_k[i][j] − c_k δ_{ij}` (`Hom_{𝕋'}(L/ℓ^e, tors) = {h : h A = 0}`), a Smith-type
  certificate `A W ≡ G · diag(ℓ^{v_0}, …, ℓ^{v_{r-1}})` with `G G' ≡ 1` (all mod `q`); then `h ↦ h G` is injective and
  `ℓ^{v_k} (h G)_k = 0`: `|Hom_{𝕋'}| ≤ ℓ^{Σ v_k}` (status `zero` iff all `v_k = 0`). -/
structure K7Cert where
  /-- the prime `ℓ` -/
  ell : Nat
  /-- the torsion exponent `e` (`q = ℓ^e`) -/
  e : Nat
  /-- `r = rank L` -/
  r : Nat
  /-- the Hecke indices `P` of `𝕋'` -/
  P : List Nat
  /-- TB-mat: per `p ∈ P`, the free block `B_p` (`r × r`) -/
  B : List (List (List Int))
  /-- TB-mat: per `p ∈ P`, the torsion coordinates `β_p` of the `T_p f_j` -/
  beta : List (List Int)
  /-- TB-mat: per `p ∈ P`, the scalar `c_p` of `T_p` on `g` -/
  c : List Int
  /-- `true`: the certificate claims a split and carries the section `h` -/
  isSplit : Bool
  /-- the section -/
  h : List Int
  /-- the elimination matrix (`r·|P|` rows, `r` columns) -/
  W : List (List Int)
  /-- the invertible `r × r` matrix `G` -/
  G : List (List Int)
  /-- its inverse `G'` modulo `q` -/
  Gi : List (List Int)
  /-- the valuations `v_k ≤ e` -/
  vals : List Nat
deriving Repr

/-- `q = ℓ^e`. -/
def K7Cert.q (c : K7Cert) : Nat := c.ell ^ c.e

/-- The system matrix `A[i][k r + j] = B_k[i][j] − c_k δ_{ij}`. -/
def K7Cert.A (c : K7Cert) (i col : Nat) : Int :=
  ientry (c.B.getD (col / c.r) []) i (col % c.r) - (if i = col % c.r then c.c.getD (col / c.r) 0 else 0)

/-- `(A W)[i][k]`. -/
def K7Cert.AW (c : K7Cert) (i k : Nat) : Int :=
  ((List.range (c.r * c.P.length)).map fun col => c.A i col * ientry c.W col k).sum

/-- The split identity `Σ_i h_i B_k[i][j] − c_k h_j ≡ β_k[j] (mod q)` for all `k`, `j`. -/
def K7Cert.splitOK (c : K7Cert) : Bool :=
  (List.range c.P.length).all fun k => (List.range c.r).all fun j =>
    (((List.range c.r).map fun i => c.h.getD i 0 * ientry (c.B.getD k []) i j).sum - c.c.getD k 0 * c.h.getD j 0
      - (c.beta.getD k []).getD j 0) % (c.q : Int) == 0

/-- The Hom-bound certificate: `A W ≡ G diag(ℓ^v)` and `G G' ≡ 1` modulo `q`, `v_k ≤ e`. -/
def K7Cert.homOK (c : K7Cert) : Bool :=
  c.vals.length == c.r && c.vals.all (fun v => decide (v ≤ c.e)) &&
  (List.range c.r).all (fun i => (List.range c.r).all fun k =>
    (c.AW i k - ientry c.G i k * ((c.ell ^ c.vals.getD k 0 : Nat) : Int)) % (c.q : Int) == 0) &&
  (List.range c.r).all (fun i => (List.range c.r).all fun k =>
    (((List.range c.r).map fun l => ientry c.G i l * ientry c.Gi l k).sum - (if i = k then 1 else 0)) % (c.q : Int) == 0)

/-- **K7**: shapes, the split identity when claimed, and the Hom bound. -/
def k7Check (c : K7Cert) : Bool :=
  decide (2 ≤ c.ell) && decide (1 ≤ c.e) && c.B.length == c.P.length && c.beta.length == c.P.length &&
  c.c.length == c.P.length && (!c.isSplit || c.splitOK) && c.homOK

/-! ### K8: the Frobenius residue (B10; an emitter consistency check, `orc:`) -/

/-- A K8 item (03 §8.2, B10(2)): on the cyclic torsion summand of order `ℓ^{j+1}` at `n = ℓ^j m`, with the Dickson monomial
`D_L^a D_D^b` of degree `m = a(ℓ+1) + b(ℓ²−ℓ)`, the emitted residues of `T_p` (`p ≠ ℓ`, from the export) are
`(1+p) p^{a ℓ^j} mod ℓ^{j+1}`. -/
structure K8Item where
  /-- `ℓ` -/
  ell : Nat
  /-- `n` -/
  n : Nat
  /-- `j` (the summand has order `ℓ^{j+1}`) -/
  j : Nat
  /-- the Dickson exponent `a` -/
  a : Nat
  /-- the Dickson exponent `b` -/
  b : Nat
  /-- `(p, emitted residue)` pairs -/
  res : List (Nat × Nat)
deriving Repr

/-- **K8**: the degree identity and every residue (`p ≠ ℓ`). -/
def k8Check (c : K8Item) : Bool :=
  decide (2 ≤ c.ell) && c.n == c.ell ^ c.j * (c.a * (c.ell + 1) + c.b * (c.ell * c.ell - c.ell)) &&
  c.res.all fun pr => pr.1 != c.ell && ((1 + pr.1) * pr.1 ^ (c.a * c.ell ^ c.j)) % c.ell ^ (c.j + 1) == pr.2 % c.ell ^ (c.j + 1)

end GL2.HeckeZ.Exec
