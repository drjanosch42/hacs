/-
SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
SPDX-License-Identifier: Apache-2.0
-/
import GL2.L23.CellsF2Dims

/-!
# LG-2 cells (slot 23): the closed form `h⁰ = ⌊(⌊n/3⌋ + 1)/2⌋` at the certified odd cells

[C] `prop:oddnltwo`(2); [GT] `gt_part_MP.tex`, row `chk:lg-h0-cells`.  The general closed form needs the eigenvalue count of
`U` over `F₄` (or Dickson's theorem for `GL₂(F₂)`), which is not formalised; here it is checked **per cell**, at the odd `n`
of `GL2.L23.Cells` (`n ≤ 63`, `n = 101, 129, 255`), from the kernel-decided `dims_n`.  With `V2_odd_cU` each cell also gives
`dim M^U = 2 ⌊(⌊n/3⌋ + 1)/2⌋`.  Written from the `dims_n` list (no literal data, no `native_decide`).
-/

namespace GL2.L23.OddDimsCells

open GL2.L23.TreeF2 GL2.L23.Cells Module

/-- `n = 1`: `h⁰ = ⌊(⌊1/3⌋ + 1)/2⌋ = 0`. -/
theorem h0_1 : finrank (ZMod 2) (groupCohomology (V2 1) 0) = (1 / 3 + 1) / 2 :=
  dims_1.1.trans (by norm_num)

/-- `n = 3`: `h⁰ = ⌊(⌊3/3⌋ + 1)/2⌋ = 1`. -/
theorem h0_3 : finrank (ZMod 2) (groupCohomology (V2 3) 0) = (3 / 3 + 1) / 2 :=
  dims_3.1.trans (by norm_num)

/-- `n = 5`: `h⁰ = ⌊(⌊5/3⌋ + 1)/2⌋ = 1`. -/
theorem h0_5 : finrank (ZMod 2) (groupCohomology (V2 5) 0) = (5 / 3 + 1) / 2 :=
  dims_5.1.trans (by norm_num)

/-- `n = 7`: `h⁰ = ⌊(⌊7/3⌋ + 1)/2⌋ = 1`. -/
theorem h0_7 : finrank (ZMod 2) (groupCohomology (V2 7) 0) = (7 / 3 + 1) / 2 :=
  dims_7.1.trans (by norm_num)

/-- `n = 9`: `h⁰ = ⌊(⌊9/3⌋ + 1)/2⌋ = 2`. -/
theorem h0_9 : finrank (ZMod 2) (groupCohomology (V2 9) 0) = (9 / 3 + 1) / 2 :=
  dims_9.1.trans (by norm_num)

/-- `n = 11`: `h⁰ = ⌊(⌊11/3⌋ + 1)/2⌋ = 2`. -/
theorem h0_11 : finrank (ZMod 2) (groupCohomology (V2 11) 0) = (11 / 3 + 1) / 2 :=
  dims_11.1.trans (by norm_num)

/-- `n = 13`: `h⁰ = ⌊(⌊13/3⌋ + 1)/2⌋ = 2`. -/
theorem h0_13 : finrank (ZMod 2) (groupCohomology (V2 13) 0) = (13 / 3 + 1) / 2 :=
  dims_13.1.trans (by norm_num)

/-- `n = 15`: `h⁰ = ⌊(⌊15/3⌋ + 1)/2⌋ = 3`. -/
theorem h0_15 : finrank (ZMod 2) (groupCohomology (V2 15) 0) = (15 / 3 + 1) / 2 :=
  dims_15.1.trans (by norm_num)

/-- `n = 17`: `h⁰ = ⌊(⌊17/3⌋ + 1)/2⌋ = 3`. -/
theorem h0_17 : finrank (ZMod 2) (groupCohomology (V2 17) 0) = (17 / 3 + 1) / 2 :=
  dims_17.1.trans (by norm_num)

/-- `n = 19`: `h⁰ = ⌊(⌊19/3⌋ + 1)/2⌋ = 3`. -/
theorem h0_19 : finrank (ZMod 2) (groupCohomology (V2 19) 0) = (19 / 3 + 1) / 2 :=
  dims_19.1.trans (by norm_num)

/-- `n = 21`: `h⁰ = ⌊(⌊21/3⌋ + 1)/2⌋ = 4`. -/
theorem h0_21 : finrank (ZMod 2) (groupCohomology (V2 21) 0) = (21 / 3 + 1) / 2 :=
  dims_21.1.trans (by norm_num)

/-- `n = 23`: `h⁰ = ⌊(⌊23/3⌋ + 1)/2⌋ = 4`. -/
theorem h0_23 : finrank (ZMod 2) (groupCohomology (V2 23) 0) = (23 / 3 + 1) / 2 :=
  dims_23.1.trans (by norm_num)

/-- `n = 25`: `h⁰ = ⌊(⌊25/3⌋ + 1)/2⌋ = 4`. -/
theorem h0_25 : finrank (ZMod 2) (groupCohomology (V2 25) 0) = (25 / 3 + 1) / 2 :=
  dims_25.1.trans (by norm_num)

/-- `n = 27`: `h⁰ = ⌊(⌊27/3⌋ + 1)/2⌋ = 5`. -/
theorem h0_27 : finrank (ZMod 2) (groupCohomology (V2 27) 0) = (27 / 3 + 1) / 2 :=
  dims_27.1.trans (by norm_num)

/-- `n = 29`: `h⁰ = ⌊(⌊29/3⌋ + 1)/2⌋ = 5`. -/
theorem h0_29 : finrank (ZMod 2) (groupCohomology (V2 29) 0) = (29 / 3 + 1) / 2 :=
  dims_29.1.trans (by norm_num)

/-- `n = 31`: `h⁰ = ⌊(⌊31/3⌋ + 1)/2⌋ = 5`. -/
theorem h0_31 : finrank (ZMod 2) (groupCohomology (V2 31) 0) = (31 / 3 + 1) / 2 :=
  dims_31.1.trans (by norm_num)

/-- `n = 33`: `h⁰ = ⌊(⌊33/3⌋ + 1)/2⌋ = 6`. -/
theorem h0_33 : finrank (ZMod 2) (groupCohomology (V2 33) 0) = (33 / 3 + 1) / 2 :=
  dims_33.1.trans (by norm_num)

/-- `n = 35`: `h⁰ = ⌊(⌊35/3⌋ + 1)/2⌋ = 6`. -/
theorem h0_35 : finrank (ZMod 2) (groupCohomology (V2 35) 0) = (35 / 3 + 1) / 2 :=
  dims_35.1.trans (by norm_num)

/-- `n = 37`: `h⁰ = ⌊(⌊37/3⌋ + 1)/2⌋ = 6`. -/
theorem h0_37 : finrank (ZMod 2) (groupCohomology (V2 37) 0) = (37 / 3 + 1) / 2 :=
  dims_37.1.trans (by norm_num)

/-- `n = 39`: `h⁰ = ⌊(⌊39/3⌋ + 1)/2⌋ = 7`. -/
theorem h0_39 : finrank (ZMod 2) (groupCohomology (V2 39) 0) = (39 / 3 + 1) / 2 :=
  dims_39.1.trans (by norm_num)

/-- `n = 41`: `h⁰ = ⌊(⌊41/3⌋ + 1)/2⌋ = 7`. -/
theorem h0_41 : finrank (ZMod 2) (groupCohomology (V2 41) 0) = (41 / 3 + 1) / 2 :=
  dims_41.1.trans (by norm_num)

/-- `n = 43`: `h⁰ = ⌊(⌊43/3⌋ + 1)/2⌋ = 7`. -/
theorem h0_43 : finrank (ZMod 2) (groupCohomology (V2 43) 0) = (43 / 3 + 1) / 2 :=
  dims_43.1.trans (by norm_num)

/-- `n = 45`: `h⁰ = ⌊(⌊45/3⌋ + 1)/2⌋ = 8`. -/
theorem h0_45 : finrank (ZMod 2) (groupCohomology (V2 45) 0) = (45 / 3 + 1) / 2 :=
  dims_45.1.trans (by norm_num)

/-- `n = 47`: `h⁰ = ⌊(⌊47/3⌋ + 1)/2⌋ = 8`. -/
theorem h0_47 : finrank (ZMod 2) (groupCohomology (V2 47) 0) = (47 / 3 + 1) / 2 :=
  dims_47.1.trans (by norm_num)

/-- `n = 49`: `h⁰ = ⌊(⌊49/3⌋ + 1)/2⌋ = 8`. -/
theorem h0_49 : finrank (ZMod 2) (groupCohomology (V2 49) 0) = (49 / 3 + 1) / 2 :=
  dims_49.1.trans (by norm_num)

/-- `n = 51`: `h⁰ = ⌊(⌊51/3⌋ + 1)/2⌋ = 9`. -/
theorem h0_51 : finrank (ZMod 2) (groupCohomology (V2 51) 0) = (51 / 3 + 1) / 2 :=
  dims_51.1.trans (by norm_num)

/-- `n = 53`: `h⁰ = ⌊(⌊53/3⌋ + 1)/2⌋ = 9`. -/
theorem h0_53 : finrank (ZMod 2) (groupCohomology (V2 53) 0) = (53 / 3 + 1) / 2 :=
  dims_53.1.trans (by norm_num)

/-- `n = 55`: `h⁰ = ⌊(⌊55/3⌋ + 1)/2⌋ = 9`. -/
theorem h0_55 : finrank (ZMod 2) (groupCohomology (V2 55) 0) = (55 / 3 + 1) / 2 :=
  dims_55.1.trans (by norm_num)

/-- `n = 57`: `h⁰ = ⌊(⌊57/3⌋ + 1)/2⌋ = 10`. -/
theorem h0_57 : finrank (ZMod 2) (groupCohomology (V2 57) 0) = (57 / 3 + 1) / 2 :=
  dims_57.1.trans (by norm_num)

/-- `n = 59`: `h⁰ = ⌊(⌊59/3⌋ + 1)/2⌋ = 10`. -/
theorem h0_59 : finrank (ZMod 2) (groupCohomology (V2 59) 0) = (59 / 3 + 1) / 2 :=
  dims_59.1.trans (by norm_num)

/-- `n = 61`: `h⁰ = ⌊(⌊61/3⌋ + 1)/2⌋ = 10`. -/
theorem h0_61 : finrank (ZMod 2) (groupCohomology (V2 61) 0) = (61 / 3 + 1) / 2 :=
  dims_61.1.trans (by norm_num)

/-- `n = 63`: `h⁰ = ⌊(⌊63/3⌋ + 1)/2⌋ = 11`. -/
theorem h0_63 : finrank (ZMod 2) (groupCohomology (V2 63) 0) = (63 / 3 + 1) / 2 :=
  dims_63.1.trans (by norm_num)

/-- `n = 101`: `h⁰ = ⌊(⌊101/3⌋ + 1)/2⌋ = 17`. -/
theorem h0_101 : finrank (ZMod 2) (groupCohomology (V2 101) 0) = (101 / 3 + 1) / 2 :=
  dims_101.1.trans (by norm_num)

/-- `n = 129`: `h⁰ = ⌊(⌊129/3⌋ + 1)/2⌋ = 22`. -/
theorem h0_129 : finrank (ZMod 2) (groupCohomology (V2 129) 0) = (129 / 3 + 1) / 2 :=
  dims_129.1.trans (by norm_num)

/-- `n = 255`: `h⁰ = ⌊(⌊255/3⌋ + 1)/2⌋ = 43`. -/
theorem h0_255 : finrank (ZMod 2) (groupCohomology (V2 255) 0) = (255 / 3 + 1) / 2 :=
  dims_255.1.trans (by norm_num)

end GL2.L23.OddDimsCells
