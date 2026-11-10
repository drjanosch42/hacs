/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.HeckeZ.Exec.K3

/-!
# HeckeZ T1 = `(1, 10)`, Ramanujan's 691: the K3 certificate data (M-HZ1) — Mathlib-free

`07-lean-plan.md` §7. Written by hand from two emitted files (re-typeset only; nothing here is computed by the author):

* **TB-mat.** `T₂` on the torsion-free block `e_𝔪 H¹(SL₂(ℤ); Sym¹⁰ ℤ/691²) = (ℤ/691²)³` (`f = 3`, one block, verdict
  `stable`, `K = 2`): the `gen 2` matrix of `satcert_n10_ell691_b1.txt`, dumped by
  `HECKE_DUMP_SAT_CERT=<scratch> HECKE_DUMP_SAT_CERT_CELLS=10:691:2 ./cohomology_app HeckeSatCertDumpValidation`
  (2026-11-04; binary `s24/fmtmp/build/cohomology_app`, sha256
  `78d19608d6dcdd4d4cdc8f1e6ad82a71c92122ddc192a262199239fbe9cb10a8`, built from `3a8169cb`; dump sha256
  `29ffd6a3fed7d568eda29fd0ba31ab8cca1dc0f83ac6ba0dfbf6b777a1e49eca`). Column convention, entries mod `691² = 477481`;
  `477457 ≡ −24`.
* **TB-χ.** `χ_ℤ(T₂) = (x + 24)² (x − 2049)`, `χ_ℤ(T₃) = (x − 252)² (x − 177148)`: the `charpoly_factored` of
  `descent(1, 10, [2,3,5,7])` (02's `t1_descentj`, output sha256
  `9840ce51ca19c0abad0c38c7a20301e0b8bb3e5bde5d5f12ae155f8f8079a1cf`, same binary), entering the theorem
  `GL2.HeckeZ.T1.T1_691` as its hypotheses `hχ2`, `hχ3`; here only their roots `λ_p` (simple, Eisenstein) and `μ_p`
  (double, `Δ`) are listed.

The dump, the console script and the descent output are committed under
`docs/implementation/slot24/heckez/mhz1-data/` (with `SHA256SUMS`); the exact command lines are in
`MHZ1-report.md` §2. `certT1_chi` below checks TB-χ against TB-mat (audit m1).

`g_P = gcd(2049 + 24, 177148 − 252) = gcd(2073, 176896) = 691 = 691¹`; the at-root read is at `p = 2`, `v = 0`, witness
entry `(0, 2)`: `c(T₂) = T₂ + 24` has the entry `4860 = 7·691 + 23`, a unit mod 691.
-/

namespace GL2.HeckeZ.T1Data

open GL2.HeckeZ.Exec

/-- The K3 certificate of T1. -/
def certT1 : K3Cert where
  k := 12
  ell := 691
  K := 2
  f := 3
  sys := [(2, 2049, -24), (3, 177148, 252)]
  gP := 691
  e := 1
  readP := 2
  dump := [[477457, 0, 4860], [0, 477457, 0], [0, 0, 2049]]
  v := 0
  wit := (0, 2)

/-- The checker accepts T1's certificate (kernel evaluation, no `native_decide`). -/
theorem certT1_check : k3Check certT1 = true := by decide +kernel

/-! ### Negative controls ([LP] §2.5 mutation discipline): one tamper per kind of check. -/

/-- "T1 with 692": the prime and `g_P` both claimed to be 692. -/
def certT1_692 : K3Cert := { certT1 with ell := 692, gP := 692 }

theorem tampered_692_rejected : k3Check certT1_692 = false := by decide +kernel

/-- The witness entry `4860` replaced by `691·7 = 4837` (then `c(T₂) ≡ 0 mod 691`: the read fails). -/
def certT1_wit : K3Cert := { certT1 with dump := [[477457, 0, 4837], [0, 477457, 0], [0, 0, 2049]] }

theorem tampered_wit_rejected : k3Check certT1_wit = false := by decide +kernel

/-- The cusp eigenvalue `μ₃` shifted to 253 (the upper bound `g_P` no longer matches). -/
def certT1_mu : K3Cert := { certT1 with sys := [(2, 2049, -24), (3, 177148, 253)] }

theorem tampered_mu_rejected : k3Check certT1_mu = false := by decide +kernel

/-- The Eisenstein eigenvalue `λ₂` shifted to 2050 (not `1 + 2¹¹`). -/
def certT1_lam : K3Cert := { certT1 with sys := [(2, 2050, -24), (3, 177148, 252)], gP := 1 }

theorem tampered_lam_rejected : k3Check certT1_lam = false := by decide +kernel

/-- The claimed valuation `v = 1` (with `K = 2`): the witness entry is a unit mod 691, so not every entry is
divisible by 691. -/
def certT1_v : K3Cert := { certT1 with v := 1 }

theorem tampered_v_rejected : k3Check certT1_v = false := by decide +kernel

/-! ### TB-χ against TB-mat (audit m1; 06 §5 Q10, 07 §8)

The dumped `T₂` has characteristic polynomial `(x + 24)² (x − 2049) = x³ − 2001 x² − 97776 x − 1180224` modulo
`691²`, i.e. the polynomial of the hypothesis `hχ2` of `GL2.HeckeZ.T1.T1_691` (the expansion is
`GL2.HeckeZ.T1.chi2_expand`). So the two emitted-data trust inputs do not contradict each other modulo the certifying
prime power, and `T1_691` is not vacuous for that reason. -/

/-- The dumped `T₂` (TB-mat) has the descent's characteristic polynomial (TB-χ) modulo `691²`. -/
theorem certT1_chi : chiCheck certT1 = true := by decide +kernel

/-- Negative control: the diagonal entry `2049` of the dump replaced by `2050`. `k3Check` does not look at that entry
(it still accepts), the cross-check rejects. -/
def certT1_diag : K3Cert := { certT1 with dump := [[477457, 0, 4860], [0, 477457, 0], [0, 0, 2050]] }

theorem tampered_diag_k3_accepted : k3Check certT1_diag = true := by decide +kernel

theorem tampered_diag_chi_rejected : chiCheck certT1_diag = false := by decide +kernel

end GL2.HeckeZ.T1Data
