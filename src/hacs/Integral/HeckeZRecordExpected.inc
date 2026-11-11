// SPDX-FileCopyrightText: 2024-2026 Fabian Januszewski
// SPDX-License-Identifier: LGPL-3.0-only WITH LicenseRef-CUDA-exception
// HZ-REC frozen status table and record pins (HZREC_report.md Sec. 3; re-frozen 2026-11-06 at the schema-audit fix round: G4-G6 per
// torsion point, split_on_P (F1), no G8 (F2), the cuspidal block (F5), searched theta (F6); each entry {signature, record_fnv1a64} (F14)).
// Re-pinned 2026-11-06 at the re-audit fix round (HZREC_schema_reaudit.md): every hash moves with the settings text (R2 second_route,
// R8 irreducible cap 256, R9 theta_search over P u {T_ell}); (2,6) anemic and (8,10) anemic also by R1 (Eisenstein mult = e_k).
// Re-pinned 2026-11-08 by slot-26 Z1 (UP-Z (a); Z1_report.md, slot26/golden_ledger.md "non-golden lines moved"): the two PENDING full
// records (2,6,full), (8,10,full) are recomputed with U_p (the full (C1) on G1.P's T_p then U_p, the ell >= 5 reads with U_p); the five
// other pins are unchanged (route (i): settings text and fingerprint unchanged).  (8,10,full) under [C] 1.26.0-pre.3 item 8 (d')(ii):
// the non-reduced branch (C1 = refused(non_reduced), C2 = not_run, amendment A2).
// Re-pinned 2026-11-08 by slot-26 Z2 (UP-Z (b); Z2_report.md, slot26/golden_ledger.md): the same two full records, the reads at ell in {2, 3} on the tree
// route with U_p, the G3 words of [C] 1.26.0-pre.9 (d')(ii) ("The G3 words from Z2 on"); (8,10,full) is now final (the non-reduced branch has no pending field left).
// Re-pinned 2026-11-10 by slot-26 HZ-EISUP (HZEISUP_report.md, slot26/golden_ledger.md "non-golden lines moved"): (2,6,full) only, its two pending
// fields moved as [C] 1.27.0-pre.1 item 8 (d')(ii) says (G1.components blocked(U_p) -> certified with the rows G1.components[0..2].*;
// G2.ell=5,17,127.maximal_ideals -> certified with their m-rows, ell=3 -> not_run), hence G0.status pending -> final; no other path moved.
{"1,10,full",
 {"(1,10,full) C1=certified redQ=true C2=complete gen=[2:certified(saturation);3:certified(saturation);691:certified(saturation)] C_T=certified:0x2b3 fp=certified G6=[2.pt[0]=split_on_P;3.pt[0]=split_on_P] status=final",
  "0x5e84b4c2540ff22a"}},
{"1,22,full",
 {"(1,22,full) C1=certified redQ=true C2=complete gen=[2:certified(eis_single);3:certified(eis_single);5:certified(saturation);103:certified(saturation);144169:certified(saturation);2294797:certified(saturation)] C_T=certified:0xe16a13b fp=certified G6=[2.pt[0]=split_on_P;3.pt[0]=split_on_P] status=final",
  "0xa68f6f1daf281d13"}},
{"11,0,anemic",
 {"(11,0,anemic) C1=certified redQ=true C2=complete gen=[2:certified(saturation);5:certified(saturation)] C_T=certified:0x5 fp=certified G6=[] status=final",
  "0xca27c0b164bf9518"}},
{"2,6,anemic",
 {"(2,6,anemic) C1=certified redQ=true C2=complete gen=[2:open(sandwich);3:certified(saturation);5:certified(saturation);17:certified(saturation)] C_T=upper_bound_certified fp=not_certified G6=[2.pt[0]=split_on_P;3.pt[0]=split_on_P;5.pt[0]=split_on_P] status=final",
  "0x676c53b5f7f47d7d"}},
{"2,6,full",
 {"(2,6,full) C1=certified redQ=true C2=complete gen=[2:open(sandwich);3:certified(saturation);5:certified(saturation);17:certified(saturation);127:certified(saturation)] C_T=upper_bound_certified fp=not_certified G6=[2.pt[0]=split_on_P;3.pt[0]=split_on_P;5.pt[0]=split_on_P] status=final",
  "0xaa516b325e24eed5"}},
{"8,10,anemic",
 {"(8,10,anemic) C1=certified redQ=true C2=complete gen=[2:open(sandwich);3:open(sandwich);5:certified(saturation);7:certified(saturation);109:certified(saturation);691:certified(saturation)] C_T=upper_bound_certified fp=not_certified G6=[2.pt[0]=non_split;3.pt[0]=split_on_P] status=pending",
  "0x45bdc565b75b660b"}},
{"8,10,full",
 {"(8,10,full) C1=refused(non_reduced) redQ=non_reduced_certified C2=not_run gen=[2:open(sandwich);3:open(sandwich)] C_T=not_run fp=not_certified G6=[2.pt[0]=non_split;3.pt[0]=split_on_P] status=final",
  "0xe15555ccce92ed3e"}},
