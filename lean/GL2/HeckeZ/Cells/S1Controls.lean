/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Cells.S1DataA
import GL2.HeckeZ.Cells.S1DataB

/-!
# HeckeZ S1: negative controls for the K2g, K1 and K1m checkers (L-HZ1, M-HZ2) — Mathlib-free

[LP] §2.5 mutation discipline, 07 §10 ("one tampered-certificate negative control per checker"). Each control changes one
field of a certificate that `cellCheck` accepts (`GL2.HeckeZ.Cells.S1DataA.cert_n10`, T1 = `(1, 10)`, with a K1 certificate
at 691; `GL2.HeckeZ.Cells.S1DataB.cert_n22`, T2 = `(1, 22)`, with K1m certificates) and is rejected by `decide +kernel`.
-/

namespace GL2.HeckeZ.Cells.S1Controls

open GL2.HeckeZ.Exec GL2.HeckeZ.Cells.S1DataA GL2.HeckeZ.Cells.S1DataB

/-- K2g / TB-χ: the constant coefficient of every recorded `χ` shifted by one (`f · q ≠ χ`). -/
def ctl_chi : CellCert :=
  { cert_n10 with items := cert_n10.items.map fun it => { it with chi := it.chi.set 0 (it.chi.getD 0 0 + 1) } }

theorem ctl_chi_rejected : cellCheck ctl_chi = false := by decide +kernel

/-- K2g: the Bezout constant `D` shifted by one (`u f + v f' ≠ D`). -/
def ctl_bezout : CellCert :=
  { cert_n10 with items := cert_n10.items.map fun it => { it with D := it.D + 1 } }

theorem ctl_bezout_rejected : cellCheck ctl_bezout = false := by decide +kernel

/-- K1: one entry of the claimed inverse pivot matrix shifted by one (`C · C⁻¹ ≢ 1`). -/
def ctl_k1_cinv : CellCert :=
  { cert_n10 with k1 := cert_n10.k1.map fun k =>
      { k with cinv := k.cinv.set 0 ((k.cinv.getD 0 []).set 0 ((k.cinv.getD 0 []).getD 0 0 + 1)) } }

theorem ctl_k1_cinv_rejected : cellCheck ctl_k1_cinv = false := by decide +kernel

/-- K1 / TB-mat: one dumped matrix replaced by the zero matrix (the monomials are no longer independent). -/
def ctl_k1_mat : CellCert :=
  { cert_n10 with k1 := cert_n10.k1.map fun k => { k with mats := k.mats.map fun M => M.map fun r => r.map fun _ => 0 } }

theorem ctl_k1_mat_rejected : cellCheck ctl_k1_mat = false := by decide +kernel

/-- K1m: `m = 0`, so `f₁(D_θ)^m = 1 ≠ 0` (the étale elements no longer vanish on the dumped blocks). -/
def ctl_k1m_nil : CellCert :=
  { cert_n22 with k1m := cert_n22.k1m.map fun k => { k with m := 0 } }

theorem ctl_k1m_nil_rejected : cellCheck ctl_k1m_nil = false := by decide +kernel

/-- K1m: `f₂ c₂ ≢ χ` (the cofactor `c₂` shifted). -/
def ctl_k1m_c2 : CellCert :=
  { cert_n22 with k1m := cert_n22.k1m.map fun k => { k with c2 := k.c2.set 0 (k.c2.getD 0 0 + 1) } }

theorem ctl_k1m_c2_rejected : cellCheck ctl_k1m_c2 = false := by decide +kernel

/-- The cell: T2 claimed complete (`cof = 1`) although `gcd D` has prime factors without a certificate (2, 3, 5, …). -/
def ctl_cof : CellCert := { cert_n22 with cof := 1 }

theorem ctl_cof_rejected : cellCheck ctl_cof = false := by decide +kernel

/-! ### The M-HZ2 audit's forgeries (F5) and the TB-χ / TB-mat cross-check (F3), slot 25 (LC-11)

FG2 and FG7 are copied from the auditor's scratch file `AudForge.lean` (outside the repository)
(sha256 `63b982b8…c573`, `MHZ2_audit.md` §3). -/

/-- FG2: K1m at 103 whose étale factor `f₂` is multiplied by a linear factor of `f₁` (a root of the dumped block):
one monomial dropped (count still `d`), `c₂`, `u₂`, `v₂` recomputed and valid; only `s₁ f₂ + t₁ f₁ ≡ 1` can fail. -/
def fg2k : K1mCert := { q := 103, fB := 3, gens := [2], mats := [[[65, 0, 10], [0, 83, 0], [50, 0, 101]]], words := [[]], pos := [(0, 0)], cinv := [[1]], th := 0, chi := [-3514568411249837604864, -370462152587526144, 333668184988608, 18079624368, -8390769, 1], f1 := [20, 1], m := 3, f2 := [42, 53, 1], c2 := [16, 72, 73, 1], u2 := [53], v2 := [96, 25], s1 := [8], t1 := [95] }
def fg2 : CellCert := { cert_n22 with k1m := fg2k :: cert_n22.k1m.filter (fun k => k.q != 103) }
theorem fg2_rejected : cellCheck fg2 = false := by decide +kernel

/-- FG7: T2 with the rank read off the data (`d := 1`, `f = x − σ₂₃(p)`, a true monic factor of `χ`; `u f + v f' = 1`),
no K1/K1m, `cof := 1`: "𝕋^fr = ℤ[T₂,T₃,T₅]" without any certificate at 2, 3, 5, 103, 144169, 2294797. -/
def fg7 : CellCert := { cert_n22 with d := 1, k1m := [], cof := 1, ex := 0, items := [{ p := 2, chi := [-3514568411249837604864, -370462152587526144, 333668184988608, 18079624368, -8390769, 1], f := [-8388609, 1], q := [418969153437696, 44212469760, -39771072, -2160, 1], u := [0], v := [1], D := 1 }, { p := 3, chi := [-34057798836439199919157167787008, -1215756995544361380161366784, -7268440872026960436096, 63919529903436192, -94143857788, 1], f := [-94143178828, 1], q := [361765974555235143936, 12913918697514240, 77206377312, -678960, 1], u := [0], v := [1], D := 1 }, { p := 5, chi := [-134682737975321206750768657008087015434287500000, -5855639013492530998374503912919063150000, 16491650999069172969732248859600, 1742101191090945466678440, -11920929101216166, 1], f := [-11920928955078126, 1], q := [11298006932416831827446006250000, 491206603684701006900000, -1383419913834600, -146138040, 1], u := [0], v := [1], D := 1 } ] }
theorem fg7_rejected : cellCheck fg7 = false := by decide +kernel

/-- F3: the audit's FG4 (a K1 certificate at the uncovered prime 5 with an invented matrix, `5²` removed from `cof`) with
the invented matrix replaced by the nilpotent shift: K1's pivots and inverse still pass (`1, M, M²` give `e₀, e₁, e₂`), and
only the new conjunct `χ_{T_2}(M) ≡ 0 (mod 5)` fails (`χ_{T_2}(0) ≢ 0 mod 5`). FG4 itself, whose matrix is the companion
matrix of `χ_{T_2} mod 5`, still passes: the trust boundary of TB-mat (audit §3, item 2). -/
def ctl_k1_chi : CellCert :=
  let fg4 : CellCert := { cert_n22 with k1 := [{ q := 5, gens := [2], mats := [[[0, 0, 0, 0, 4], [1, 0, 0, 0, 4], [0, 1, 0, 0, 2], [0, 0, 1, 0, 2], [0, 0, 0, 1, 4]]], words := [[], [0], [0, 0]], pos := [(0, 0), (1, 0), (2, 0)], cinv := [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }], cof := cert_n22.cof / 25, ex := max cert_n22.ex 2 }
  { fg4 with k1 := fg4.k1.map fun k =>
      { k with mats := [[[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]]] } }

theorem ctl_k1_chi_rejected : cellCheck ctl_k1_chi = false := by decide +kernel

end GL2.HeckeZ.Cells.S1Controls
