/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Cells.LevelNData
import GL2.HeckeZ.Cells.MazurDataA

/-!
# L-HZ3: negative controls for K3N, and the cross-check of T3 with M-HZ4a's Mazur cell `(11, 5)` — Mathlib-free

Each control tampers **one** field of a real certificate and records the whole vector of K3N's six sub-checks (`checks`:
operator guard, distinct primes, read shape, every entry divisible by `ℓ^v`, the witness, `ℓ^{v+e} ∣ λ − μ`), so that the
rejection is seen to come from exactly the intended check (the L-K10 audit's MINOR-5 lesson). All by `decide +kernel`.
-/

namespace GL2.HeckeZ.Cells.LevelNControls

open GL2.HeckeZ.Exec GL2.HeckeZ.Exec.K3N GL2.HeckeZ.Cells.LevelNData

/-- The six sub-checks of `k3nCheck` (their conjunction is `k3nCheck`, `checks_and`). -/
def checks (c : K3NCert) : Bool × Bool × Bool × Bool × Bool × Bool :=
  (opB c, nodupB (c.reads.map K3Read.ell), c.reads.all shapeB, c.reads.all (fun r => allDivB r c.mu),
    c.reads.all (fun r => witB r c.mu), c.reads.all (fun r => gapB r c.lam c.mu))

/-- `List.all` of a four-fold conjunction. -/
theorem all_and4 (l : List K3Read) (a b c d : K3Read → Bool) :
    l.all (fun r => a r && b r && c r && d r) = (l.all a && l.all b && l.all c && l.all d) := by
  induction l with
  | nil => rfl
  | cons r l ih =>
    simp only [List.all_cons, ih]
    cases a r <;> cases b r <;> cases c r <;> cases d r <;> simp

/-- `k3nCheck` is the conjunction of the six sub-checks recorded by `checks`. -/
theorem checks_and (c : K3NCert) :
    k3nCheck c = (opB c && nodupB (c.reads.map K3Read.ell) && c.reads.all shapeB &&
      c.reads.all (fun r => allDivB r c.mu) && c.reads.all (fun r => witB r c.mu) &&
      c.reads.all (fun r => gapB r c.lam c.mu)) := by
  unfold k3nCheck Exec.K3N.readOK
  rw [all_and4 c.reads shapeB (fun r => allDivB r c.mu) (fun r => witB r c.mu) (fun r => gapB r c.lam c.mu)]
  simp only [Bool.and_assoc]

/-- The real certificates pass all six. -/
theorem certT3_checks : checks certT3 = (true, true, true, true, true, true) := by decide +kernel
theorem certT4_checks : checks certT4 = (true, true, true, true, true, true) := by decide +kernel

/-- The single read of `certT3`. -/
def r3 : K3Read := certT3.reads.headD { ell := 0, K := 0, f := 0, dump := [], v := 0, wit := (0, 0), e := 0 }

/-- **Gap**: `e = 2` claims `25 ∣ λ − μ = 5`; fails the gap check alone. -/
def ctlGap : K3NCert := { certT3 with reads := [{ r3 with e := 2 }] }
theorem ctlGap_only : checks ctlGap = (true, true, true, true, true, false) := by decide +kernel
theorem ctlGap_rejected : k3nCheck ctlGap = false := by decide +kernel

/-- **Valuation too high**: `v = 1` (with `e = 0`, so the gap still holds) claims every entry of `c(T₂)` divisible by `5`; the
entry `24` is not; fails the divisibility check alone. -/
def ctlVal : K3NCert := { certT3 with reads := [{ r3 with v := 1, e := 0 }] }
theorem ctlVal_only : checks ctlVal = (true, true, true, false, true, true) := by decide +kernel
theorem ctlVal_rejected : k3nCheck ctlVal = false := by decide +kernel

/-- **Witness**: the entry `(0, 0)` of `c(T₂)` is `25 ≡ 0`; fails the witness check alone. -/
def ctlWit : K3NCert := { certT3 with reads := [{ r3 with wit := (0, 0) }] }
theorem ctlWit_only : checks ctlWit = (true, true, true, true, false, true) := by decide +kernel
theorem ctlWit_rejected : k3nCheck ctlWit = false := by decide +kernel

/-- **Tampered dump** (TB-mat): the unit entry `24` of `T₂` set to `0`; the witness entry vanishes; fails the witness check
alone. The TB-χ cross-check does *not* see this tamper (`ctlDump_chi`: the tampered matrix still has characteristic polynomial
`(x + 2)²(x − 3)` mod `25`): the cross-check is a guard against a wrong file, not a forger. -/
def ctlDump : K3NCert := { certT3 with reads := [{ r3 with dump := [[23, 0, 0], [0, 23, 0], [0, 0, 3]] }] }
theorem ctlDump_only : checks ctlDump = (true, true, true, true, false, true) := by decide +kernel
theorem ctlDump_rejected : k3nCheck ctlDump = false := by decide +kernel
theorem ctlDump_chi : chiReadsOK ctlDump = true := by decide +kernel

/-- **Not a prime**: `ℓ = 25`, `K = 1` (same modulus `25`, same dump), `e = 0`; fails the read shape (primality) alone. -/
def ctlPrime : K3NCert := { certT3 with reads := [{ r3 with ell := 25, K := 1, e := 0 }] }
theorem ctlPrime_only : checks ctlPrime = (true, true, false, true, true, true) := by decide +kernel
theorem ctlPrime_rejected : k3nCheck ctlPrime = false := by decide +kernel

/-- **Wrong weight**: `k = 3` makes `1 + 2^{k−1} = 5 ≠ λ = 3`; fails the operator guard alone. -/
def ctlLam : K3NCert := { certT3 with k := 3 }
theorem ctlLam_only : checks ctlLam = (false, true, true, true, true, true) := by decide +kernel
theorem ctlLam_rejected : k3nCheck ctlLam = false := by decide +kernel

/-- **Bad operator**: level `N = 2` with `T_2` (`p ∣ N`, `T_2` is not anemic); fails the operator guard alone. -/
def ctlPN : K3NCert := { certT3 with N := 2 }
theorem ctlPN_only : checks ctlPN = (false, true, true, true, true, true) := by decide +kernel
theorem ctlPN_rejected : k3nCheck ctlPN = false := by decide +kernel

/-- **Repeated prime**: T4's read at `17` twice (which would square the lower bound); fails the distinct-primes check alone. -/
def ctlDup : K3NCert := { certT4 with reads := certT4.reads ++ certT4.reads }
theorem ctlDup_only : checks ctlDup = (true, false, true, true, true, true) := by decide +kernel
theorem ctlDup_rejected : k3nCheck ctlDup = false := by decide +kernel

/-- **T4, gap**: `e = 2` at `17` claims `17² ∣ 2176`; fails the gap check alone. -/
def ctlGap4 : K3NCert :=
  { certT4 with reads := certT4.reads.map fun r => { r with e := 2 } }
theorem ctlGap4_only : checks ctlGap4 = (true, true, true, true, true, false) := by decide +kernel
theorem ctlGap4_rejected : k3nCheck ctlGap4 = false := by decide +kernel

/-- **TB-χ guard**: a wrong cuspidal root, `(x + 1)²(x − 3) = x³ − x² − 5x − 3` instead of the record's `(x + 2)²(x − 3)`, is
rejected by `chiReadsOK` (on `e₀`, `p(T₂) e₀ = p(−2) e₀ = −5 e₀ ≢ 0 mod 25`). -/
theorem ctlChi_rejected : chiReadsOK { certT3 with chi := [-3, -5, -1, 1] } = false := by decide +kernel

/-! ### Cross-check with M-HZ4a (L-K10): the Mazur cell `(11, 5)` -/

/-- **T3 against M-HZ4a's `(11, 5)`** (07 §10 "T3 (at-root read, cross-check of 4a)"): L-K10's certificate `c11_5` uses the same
operator (`q = 2 = p`), its cuspidal factor `h = c_2 = x + 2` is `x − μ(T₂)`, its Eisenstein root `1 + q = 3` is `λ(T₂)`, and its
valuation `v = v_5(num((11 − 1)/12)) = 1` is T3's read exponent `e = 1`. The objects differ — Mazur's completed cuspidal ring
`𝐓_𝔓` at the Eisenstein ideal (K10′, under `hMazurEis`) versus the anemic algebra `𝕋^{an,fr}` of `H¹(Γ₀(11); ℤ)/tors` with its
Eisenstein system (T3, under `hES_N_an`) — and the two certificates agree. -/
theorem mazur11_xcheck : GL2.HeckeZ.Cells.MazurData.c11_5.N = certT3.N ∧ GL2.HeckeZ.Cells.MazurData.c11_5.ell = 5 ∧
    GL2.HeckeZ.Cells.MazurData.c11_5.q = certT3.p ∧ GL2.HeckeZ.Cells.MazurData.c11_5.h = [-certT3.mu, 1] ∧
    1 + (GL2.HeckeZ.Cells.MazurData.c11_5.q : Int) = certT3.lam ∧
    certT3.reads.map K3Read.e = [GL2.HeckeZ.Cells.MazurData.c11_5.v] ∧ certT3.reads.map K3Read.ell = [5] := by
  decide +kernel

end GL2.HeckeZ.Cells.LevelNControls
