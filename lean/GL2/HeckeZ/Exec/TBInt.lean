/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/

/-!
# HeckeZ TB-int: exact integer matrices, the checks (L-HZ4, M-HZ5; Mathlib-free)

Lists of rows; the checks of `GL2.HeckeZ.Check.TBInt` (soundness) and of the per-cell identities, evaluated by `decide +kernel`.
-/

namespace GL2.HeckeZ.Exec.TBInt

/-- A matrix as a list of rows of integers. -/
abbrev IMat := List (List Int)

/-- Entry `(i, j)` of a list matrix (`0` when absent). -/
def ient (M : IMat) (i j : Nat) : Int := (M.getD i []).getD j 0

/-- `Σ_{l < m} A_{il} B_{lj}`. -/
def imulEnt (m : Nat) (A B : IMat) (i j : Nat) : Int :=
  ((List.range m).map fun l => ient A i l * ient B l j).foldr (· + ·) 0

/-- The product of an `a × m` and an `m × b` list matrix. -/
def imul (a b m : Nat) (A B : IMat) : IMat :=
  (List.range a).map fun i => (List.range b).map fun j => imulEnt m A B i j

/-- Entrywise equality on `[0, a) × [0, b)`. -/
def ieq (a b : Nat) (A B : IMat) : Bool :=
  (List.range a).all fun i => (List.range b).all fun j => ient A i j == ient B i j

/-- `A - c • 1` for an `n × n` matrix. -/
def ishift (n : Nat) (A : IMat) (c : Int) : IMat :=
  (List.range n).map fun i => (List.range n).map fun j => ient A i j - (if i == j then c else 0)

/-- `A + B` for `a × b` matrices. -/
def iadd (a b : Nat) (A B : IMat) : IMat :=
  (List.range a).map fun i => (List.range b).map fun j => ient A i j + ient B i j

/-- The column `v` (a list) as an `n × 1` matrix. -/
def icol (v : List Int) : IMat := v.map fun x => [x]

/-- `A v = c v` (`A` `n × n`, `v` a column). -/
def ieigen (n : Nat) (A : IMat) (v : List Int) (c : Int) : Bool :=
  ieq n 1 (imul n 1 n A (icol v)) (icol (v.map (c * ·)))

/-- A non-zero column. -/
def inonzero (v : List Int) : Bool := v.any (· != 0)

end GL2.HeckeZ.Exec.TBInt
