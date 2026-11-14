# `HeckeExt/` — the Hecke extension layer

> **Registry paths (S24 CL-10, 2026-11-01).** Registry fragments `Registry/Slot<S>/<pkg>.cpp` and `cmake/slot<S>/<pkg>.cmake` named in this file for
> slots S <= 23 are historical: they are folded into `Registry/Slot<S>.cpp` (the package function `RegisterSlot<S><pkg>()` is kept there, called by
> `RegisterSlot<S>()` in the old order) and `cmake/sources/<Module>.cmake`; the old READMEs are in
> `docs/implementation/registry-slot-readmes-9-26.md`, the convention in force in `src/hacs/Registry/README.md`.

The engine of the **hecke-extensions** programme: the local-algebra / module-structure
layer that sits *on top of* the shipped pipeline and consumes it one-way.

Binding documents (cite by label, never by page):

| Tag | Document |
|---|---|
| `[C]` | `docs/latex/gl2_contract.tex` — the binding mathematical contract |
| `[E]` | `docs/latex/hecke_extensions.tex` — the extension programme's specification |
| `[X]` | `docs/implementation/hecke_extensions_implementation_plan.md` (v1.6) |
| `[SP]` | `docs/implementation/hecke_extensions_slot1_plan.md`, `..._slot2_plan.md` + their audits |

## Build placement and the one-way rule

These TUs are compiled **straight into `cohomology_app`**, not into a static library
([X] §1.1): the layer consumes `MatrixT`, `FpPolyT`, `HeckeModuleDecomposition`,
`HeckeOperatorFamily`, `EquivariantCohomologyOnK` and
`EquivariantCochainComplexMorphismOnK`, every one of which is already an executable TU, and
a `HeckeExt` library would first need the `Pipeline`-library refactor the console plan
records as an unscheduled follow-up.

There is therefore **no link-time barrier** for the layer's one-way dependency, and a grep
guard is it: `docs/matrix-engine/tools/check_heckeext_includes.sh`, run inside `--all` by
`HeckeExtArchitectureValidation`. Its four rules, each with its allow-list and a reason per
entry:

1. no `#include` of `HeckeExt/` outside `HeckeExt/`, `Campaign/`, `Golden/`, `Bench/`,
   `main.cpp`, `*Validation.*` and `ValidationRegistry.cpp`;
2. inside `HeckeExt/`, no `Console/`, no `src/Cuda/`, no `src/Engine/`
   and no `BlockFlatMatrix.h` — the layer is `Matrix`-sized and single-threaded ([X] §2.3);
3. `HeckeSturmBound.*` may include only `Hecke/SubspaceOps.h`, `Hecke/FpPoly.h`, `Matrix.h`,
   `ModArithmetic.h`, `EngineCheck.h` and `Fnv.h`;
4. no `unordered_` anywhere in the layer (determinism hazard 02 §6.5(5)).

The `Hecke/` → `HeckeExt/` direction is forbidden, which is why `SubspaceOps`,
`HeckeSettings`, `HeckeBoundaryDatum`, `HeckeBoundaryOperator` and `HeckePairHomotopy` live
under `Hecke/` and not here ([SP] §2.3 items 5 and 7, adopted into [X] v1.5 §1.1).

## Files → `[E]` sections

| File | Phase | `[E]` / `[X]` |
|---|---|---|
| `HeckeSturmBound.{h,cpp}` | **E1s** (slot 1, WP-3) | `[E]` `def:bprime`, `cl:bprime`, `cl:sturmmodl`, `cert:sturm` part 1; `[X]` §1.2 (`SturmBound`), §1.3 (`SturmBoundCertificate`), §1.3.1 R3 instance 1, §1.5 (`sturm.cap`, `sturm.unavailable`), §2.2 Stage 4(c) |
| `HeckeSturmBoundValidation.{h,cpp}` | E1s | `[X]` §5, row `HeckeSturmBoundValidation` |
| `HeckeExtArchitectureValidation.{h,cpp}` | E1s | `[X]` §5, row `HeckeExtArchitectureValidation` |
| `HeckeIdentityCertificate.{h,cpp}` | **E0a** (slot 1, WP-2); item 5 slot 5 WP-8; item 7 slot 6 WP-3; **item 6 slot 6 WP-9** | `[E]` §5.1 identity classes 1–4, 5 (Dickson `L·`) and **6 (the residue atlas: `[H^1(M_n)] = sum_i theta^{j_i}[H^1(M_{n_i})]` + the `H^0` / `H^2` corrections over `[E]` `cl:glover`'s factors, under the STANDARD-module convention quoted verbatim; `GloverConvention::kContragredient` is the negative control)**; item 7 = WP-3's socle ↔ Loewy under PD; `[X]` §1.3 `IdentityCertificate` |
| `HeckeIdentityValidation.{h,cpp}` | E0a | `[X]` §5, row `HeckeIdentityValidation` |
| `HeckeLocalAlgebra.{h,cpp}` | **E1a** (slot 2, WP-4) | `[E]` `sec:localalg`, `alg:stage3`, `rmk:coef`, `cl:resfield`, `cert:cyclic`; `[X]` §1.2 `LocalAlgebra`, §1.3 `CyclicityCertificate`, §2.2 Stage 3 + Stage 4(a), Stage 2's cross-block test ([XA M5]); `[C]` `obl:algebra` / `thm:relations` (commutativity), `cl:pd`(3) (`T_m`) |
| `HeckeGenerationCertificates.{h,cpp}` | **E1a** (slot 2, WP-5) | `[E]` `cert:comm`; `[X]` §1.2 `GenerationCertificate` (route b), §1.3 `CommutantCertificate`, §1.3.1 R3.2 + R3.3 item 3 (`CommutantState`), §2.2 Stage 4(b) (the v1.2 rework, `[XA B2]`) |
| `HeckeSturmSweep.{h,cpp}` | **E1b** (slot 3, WP-2) | `[E]` `cert:sturm` (the sweep proper) on the range of `cl:sturmmodl`, `cl:sep` (the `S_sep(ell)` additions deliberately left to E6); `[X]` §1.2 `SturmCertificate`, §1.3.1 R3 (the sweep's budget IS `sturm.cap`), §1.5 (`sturm.cap`, `sturm.unavailable`, `u_ell`), §2.2 **Stage 1** (the base operator set and the `u_ell` budget check) and **Stage 4(c)**; `[C]` `obl:algebra` (commutativity) |
| `HeckeExtSturmSweepValidation.{h,cpp}` | E1b | `[X]` §5, the REAL-BLOCK arms of the `HeckeExtAlgebraValidation` row (they cannot live there: that file's arms are constructed truths, and `HeckeSturmBoundValidation` consumes no pipeline by design) |
| `HeckeExtAlgebraValidation.{h,cpp}` | E1a | `[X]` §5, row `HeckeExtAlgebraValidation` (the constructed-truth arms of WP-4 **and** WP-5's commutant / socle-bound / R3 arms; the real-block arms are E1b's) |
| `HeckeBoundary.{h,cpp}` | **E4b** (slot 2, WP-7) | the ext-layer CONSUMER of `Hecke/HeckeBoundaryDatum` + `HeckeBoundaryOperator` ([X] §1.1, §6.2 E4b): `H^1_! = ker r^1` ([C] `cl:les`(2)) with its Hecke action `A_!` (exact solve; T_p-stable by `obl:taubdry`(vii)), the joint `1 + p^{n+1}` eigenspace / Eisenstein line (route (a) of [C] `obl:iota`(v)), Harder's criterion [E] `cl:harder` from a pinned `num(B_k/k)` table (`k <= 26`; `bernoulli_mod` stays E6) with a four-state verdict, the Harder split (`charpoly(A) = charpoly(A_!) charpoly(A_bd|im r^1)`, `H^1 = H^1_! (+) Eis`), the copyable `HeckeBoundaryCertificate` with `verify()`, the golden collectors `golden_objects(p, nu)` (15 records, GoldenHarness hashing, **never emitted** -- E2's add-only emission), the per-`(p, nu)` diagnostics of `Hecke/HeckeBoundaryDiagnostics`, and `to_json()` (`hecke_boundary_parabolic/v1`, the per-complex `q_max / truncated / q_max_source / degrees` of slot-2 confirmation 1). The interfaces WP-8 builds on: `pipeline(p, nu)` (cosets, bulk tau, tau_bd, both operators on one coset table), `restriction()`, `boundary_cohomology()`, `K_bd()` |
| `HeckeBoundaryParabolicValidation.{h,cpp}` | E4b (slot 2, WP-7) | `[X]` §5 row `HeckeBoundaryValidation` (E4b half): gates G1-G11 of its header; prints `BOUNDARY_DIAG ...` and `BOUNDARY_PARABOLIC ...` |
| *(under `Hecke/`)* `HeckePairHomotopy.{h,cpp}` | **E5-1** (slot 2, WP-8) | the eta-table of [C] `sec:cone` (`def:pairhomotopy`, `lem:etaexists`, `obl:eta`(i)-(iv)), the first third of the Hecke datum of the pair `(tau, tau_bd, eta)` that `HeckeConeAction` (slot 3) consumes: `eta(j, q, sigma)` on every stored slot, the (T1) helper `eta_on_translate(gamma, q, sigma)` (what the derived `h = eta~` of `def:coneaction` / `eq:derivedh` is transported from), `defect` / `defect_on_translate` (`j tau_bd - tau j`), `unique_fill` (the `L` of `lem:etaunique` and the independence arm of `obl:coneexact`(viii)(a)), `eta_zero`, `to_json()` (`hecke_pair_homotopy/v1`, the per-complex degree fields of slot-2 confirmation 1). Lives under `Hecke/` per the guard (it consumes the templated `HeckeTauTableT` / `HeckeBoundaryTauTableT`); validated by `Hecke/HeckePairHomotopyValidation` |
| `HeckePredictions.{h,cpp}` | **E6** (slot 4, WP-7) | `[X]` §2.1 "Bernoulli numbers mod `ell`" (the `F_ell` recursion `sum_j C(w+1, j) B_j = 0`, valid for `w < ell - 1` by von Staudt–Clausen; this is what settles `[PA A9]`), §2.2 Stage 6 (`HeckePredictions`); `[E]` `cl:harder` (the Kummer condition `ell \| num(B_k/k)`) and Claim 1.8's Kummer list. `bernoulli_mod(w, ell)` / `bernoulli_mod_table`, the named refusal `BernoulliOutOfRange` at `w >= ell - 1`, and `kummer_divides(k, ell)` / `kummer_verdict` with three provable branches (`von_staudt_clausen`, `direct`, `kummer_congruence`) plus the two degenerate ones. **It does NOT rewire `HeckeBoundary`'s pinned `k <= 26` table** — that moves `HeckeBoundaryParabolicValidation`'s recorded `harder_applicable` for `k > 26` and is slot 5's, with the record integration |
| `GloverFactors.{h,cpp}` | **E6** (slot 4, WP-7) | `[E]` §identities item 6 (the Ash–Stevens / Glover factors `Sym^{n_i} (x) det^{j_i}`, `0 <= n_i <= ell - 1`, `n_i + 2 j_i == n`); `[X]` §5 `HeckeAtlasValidation`'s Glover arm. The composition factors of `Sym^n(F_ell^2)` as an `F_ell[GL_2(F_ell)]`-module with multiplicities, by the `theta`-multiplication recursion (`theta = x^ell y - x y^ell`, the product of the `ell + 1` lines of `P^1(F_ell)`) whose cokernel is the principal series `Ind_B^G(chi)`; the SL_2 view collapses the det-twists. The convention is pinned to the pipeline's own `rho_n` = `ComputeSymmetricPowerMatrixFM` (`Sym^n` of the STANDARD module, not its dual — the two differ by `det^{-n}`) |
| `HeckePredictionsValidation.{h,cpp}` | E6 (slot 4, WP-7) | `[X]` §5, the two UNIT arms of the `HeckeAtlasValidation` row (they cannot live there: the atlas driver is slot 5's). Eight arms, 113 664 checks; see the header |
| `HeckeCanonicalField.{h,cpp}` | **E6** (slot 5, WP-7; planned as slot-4 WP-10, which did not run) | `[X]` §2.2 Stage 6 (`HeckeCanonicalField`), `[E]` `def:canon` / `cl:canon` / `alg:split`, `[OD]` OQ9 (the defining polynomial) and OQ10 (the default report form). `C_{ell,r}` = the `CanonicalLess`-least monic irreducible of degree `r`, by enumeration in that order + `hecke::CertifyIrreducible`; `CanonicalField` = the arithmetic of `F = F_ell[Y]/(C)`; `FqPoly` = univariate polynomials over `F` (the only new arithmetic Stage 6 needs); `FindCanonicalRoot` = `alg:split` step 1 with its FIXED candidate sequence; `SplitInCanonicalField` / `CertifyCanonicalSplit` / `to_json` (`hecke_canonical_split/v1`); `OwnFieldReportJson` = tier (i) of `[E]` `sec:field` (repointed at the slot-6 close from `sec:reportfield`, which never existed). Pipeline-free (`Hecke/FpPoly`, `Hecke/HeckeSettings`, `ModCtx.h`, `EngineCheck.h`). **NO REPORT CONSUMER in this slot** — nothing on a pipeline path calls it, no record gains a key; the integration is E6's, in slot 6 |
| `HeckeCanonicalFieldValidation.{h,cpp}` | E6 (slot 5, WP-7) | `[X]` §5; seven arms, 17 054 checks, `fast`. Every oracle independent of the unit: exhaustive trial division, the Moebius irreducible count, the `FpPoly`-mod-`C` model of `F`, an EXHAUSTIVE root search over `F`, and a nine-way tamper on the split certificate. Arm (F) is the `(2, 30)` stress case |
| `HeckeDicksonMultiplication.{h,cpp}` | **E1 identity class 5** (slot 5, WP-8) | `[X]` §1.1's inventory gap `[I G12]` / `[XA m10]`, §2.2 Stage 5 (`L·H^1` as a one-entry `FiberBlockTable`); `[E]` §identities item 6's `L·` half and `cl:Tlnil`. The `(n+1) × (n−ell)` matrix of `L· : Sym^{n−ell−1} → Sym^n`, `L = x^ell y − x y^ell`, in the monomial basis of `[C, def:rhon]` (`+1` at row `j+1`, `−1` at row `j+ell`), and `DicksonMultiplicationCertificate` whose `verify()` re-derives the shape from `(n, ell)`, every entry by an independent homogeneous convolution, the injectivity, and the det-twisted intertwining `ρ_n(A) L = det(A) L ρ_m(A)` entrywise on 24 seeded `A` — singular ones included, because the identity is proved in characteristic `ell` with no determinant hypothesis. **Typed `FlatMatrix`, not `ExtMatrix`**: it is a *fibre*-level object (the same kind as `ρ_n`) whose consumer is the cochain layer's `PerSimplexBlock::F`, so it includes no `Matrix.h` and needs no `check_matrix_includes.sh` entry. Pipeline-free (`FlatMatrix`, `ModCtx`, `EngineCheck`) |
| `HeckeM1Table.{h,cpp}` | **M1** (slot 5, WP-8) | `[X]` §6.2 E2's **M1** and §1.3.1 R3.2's fixed reason list. One definition of *an M1 row* (cell, group, ideal, algebra invariants, the three routes, `B'(k, ell)`, the `final` tri-state and its reason) and of *a well-formed M1 row* (`CheckM1Row`, six clauses — the load-bearing one is `IsFinalReason`), shared by the three case validations so that the deliverable is one object and not three. It computes nothing except the cell's `B'` through `HeckeSturmBound` |
| `HeckeExtKummerValidation.{h,cpp}` | M1 (slot 5, WP-8) | `[X]` §5, row `HeckeExtKummerValidation`, **`slow`, 37.5 s, 424 checks**; `[E]` §8.3's Kummer item. Arm 1 checks the LIST against `kummer_divides` / `bernoulli_mod` and sweeps every odd prime ≤ 20000 for completeness (7 hits, exactly `[E]` Claim 1.8's pairs in that range); arm 2 runs all ten cells through `HeckeExtDriver` at the default settings |
| `HeckeExtSmallEllValidation.{h,cpp}` | M1 (slot 5, WP-8) | `[X]` §5, row `HeckeExtSmallEllValidation`, **`standard`, 1.1 s, 225 checks**; `[E]` §8.3. The Dickson loci, the two Δ-multiplicity cells, the two `U_ell` self-extension cells, the Sturm counterexamples on the pipeline's own `H^1`, `(32, 131)`, and **arm 5**, the `final = false` half of M1 (`proof_range` as an unverified caller INPUT) |
| `HeckeAtlas.{h,cpp}` | **E6** (slot 6, WP-8) | `[E]` `sec:atlas` (Stage 6), `cl:sigma`, `cl:sep`, `sec:field` tier (ii); `[X]` §2.2 Stage 6; slot-6 owner ruling 4. For one `ell`: `HeckeExtDriver` on every even `n <= ell - 1` (weight `ell + 1` included) with the anemic pool `T = {q <= 31} \ {ell}`; every certified-local ideal put in its **canonical split form** through ONE `CanonicalField` per `(ell, r)` — **the canonical split field's first consumer** — and keyed by the least Frobenius conjugate of its anemic vector, which identifies systems across `n`; `a_ell` per occurrence where the `u_ell` verdict admitted `U_ell`, else `unknown` with the verdict's reason (never guessed); the `theta`-twist closure `Sigma(ell)`; the greedy `S_sep(ell)` separating `Sigma u {chi_{a,b}}` as Galois orbits (cross-degree pairs it cannot decide without an embedding are counted **unseparated**, never guessed). The store `hecke_atlas/v1` is byte-deterministic (no commit, no timing), has its own reader (`ParseAtlasStore`), and is written **only** under `HECKE_ATLAS_OUT=<dir>` (a directory under `docs/matrix-engine/results/` is refused). It enters no record and no golden object |
| `HeckeAtlasValidation.{h,cpp}` | E6 (slot 6, WP-8) | `HeckeAtlasSkeletonValidation` (`fast`, `ell in {5, 7}`: closure on keys **and** on polynomial data, `S_sep` separates, round trip, byte-determinism across two builds, the write gate closed, the `r >= 2` path on constructed data) and `HeckeAtlasValidation` (`slow`, `[X]` §5: `Sigma(5)`, `Sigma(7)`, `Sigma(11)`, `Sigma(13)`, the Eisenstein systems and the `chi` family, `Delta mod 11` / `mod 13` against `tau(q)`, `a_ell = tau(ell)`, not a lower-weight twist, the Kummer table + the Kummer/Ribet dichotomy on the atlas, the Glover/Brauer arm for `n <= 3 ell` at `ell in {5, 7}` through `GloverBrauerCrossCheck` — the same oracle as `HeckePredictionsValidation` arm [H] — and the per-cell `canonical_field_s / total_s` column) |
| `HeckeExtEndToEndValidation.{h,cpp}` | **E6** (slot 6, WP-9) | `[X]` §5, row `HeckeExtEndToEndValidation`, **`slow`, ~45 s, 5135 checks**. [A] the certificate's Glover recursion == `GloverFactors` (404 cells); [B] **identity 6** (certificate item 6, with items 1–5) on `ell in {5, 7, 11}`, `n in {0..60} u {80, 100, 120, 160, 200}`, pool `{q <= 31} \ {ell}`, `u_ell = off`, every `H^1` class located in `Sigma(ell)`; [C] the **negative control** — the contragredient reading fails item 6 on all 63 non-vacuous cells, named cell `(22, 5)`; [D] item 7 (WP-3's socle ↔ Loewy) asserted in the same certificate on 8 cells; [E] `ExtDriverOptions::s_sep` (default empty) on the 31 ext-subgrid cells and the M1 list — vacuous at `ell <= 13` (records byte-identical where `S_sep` is a subset of `S_0`), plus a constructed one-prime `S_0'` arm where Stage 2(3) acts; [F] the `proof_range` comparison, RECORDED only. `ell >= 691`: atlas unavailable |
| `HeckeExtSeparationValidation.{h,cpp}` | M1 (slot 5, WP-8) | `[X]` §5, row `HeckeExtSeparationValidation`, **`standard`, 3.6 s, 92 checks**; `[E]` §8.2 / `[Q17]` / `[Q13]`. `(160, 691)`'s seven blocks asserted, the separation claim MEASURED (see the report's finding 1), and `(198, 691)`'s five blocks with its one non-split `3:1`
| `HeckePresentation.{h,cpp}` | **E2 (a)** (slot 6, WP-2; `[OD]` OQ11) | `[X]` §2.2 Stage 5, "On-request presentation". Generators (the least-index Nakayama lift of a `K`-basis of `V / m_S V`) and a minimal set of relations (Nakayama on `R = ker(A_S^t -> V)`) of a certified-local block over `T_m = A_S\|_V`, relations as `t`-tuples over the stored RREF basis of `A_S`; `verify()` re-derives `A_S` / `m_S` (through the embedded `CyclicityCertificate`), the lift, the span, the vanishing, the generation of `R` and the count `dim R / m_S R`. `to_json()` = `hecke_presentation/v1`, **not a record key**. Reached through `ComputePresentation(LocalAlgebra)` or `ExtDriverOptions::with_presentation` (default off) — **no settings key in slot 6**: the `presentation = off \| on` key, the record key and the hash-invariance arm are slot 7's changed-line re-baseline (slot-6 plan §2.3 item 2, §4.4). Zero raw `mod_*` sites, no direct `Matrix.h` include |
| `HeckePresentationValidation.{h,cpp}` | E2 (a) (slot 6, WP-2) | `[X]` §5, row `HeckePresentationValidation`, **`fast`, ~1.9 s, 995 checks**. Constructed algebras (`A^dual` over `F[x,y]/(x,y)^2` with its three relations, cyclic `F[x]/(x^3)`, `A + A`, the `(10, 691)` shape `A + A/m`, its `f = 2` analogue, a field) against their known presentations up to the stored basis and conjugated, over `ell in {5, 13, 691, 134217757}`; an eleven-way tamper arm; at `(10, 691)` through the driver: `H^1_+` free of rank one, the unsplit `H^1_m` with `t = 2` and one minimal relation, and the unsplit and per-sign record texts byte-identical with the option on and off. The key arm is **deferred** (see the row above) |
| `HeckeAtlas.{h,cpp}` (slot-7 WP-5 changes) | **E6 cont.** (slot 7, WP-5) | `BuildAtlas` = `CollectAtlas` (the pipeline) + `CloseAtlas` (the `theta`-twist closure and the greedy `S_sep`, re-derivable from a store's collected prefix, so `CloseAtlas(ParseAtlasStore(json))` re-closes to `json` byte for byte). Rewritten to scale to `ell = 691` (`|Sigma u chi| ~ 10^6`) with the SAME output: twists as scaled conjugates of the base (`theta^j` commutes with Frobenius), a `(r, key)` index in place of the linear `AtlasFind`, `AtlasSeparationBy` by group counting with the slot-6 decision per pair (the slot-6 pairwise loop is kept verbatim in the validation as the reference, arm (L)); the four `ell <= 13` stores are byte-identical to the `3a40f68` binary's. `AtlasSeparation`'s counters are 64-bit. `AtlasOptions::n_limit` (a scratch atlas) and `progress` (one `ATLAS_CELL` line per cell). The results/ write guard is unchanged (slot-7 confirmation 14) |
| `HeckeAtlasCli.{h,cpp}` | E6 cont. (slot 7, WP-5) | `cohomology_app --atlas <ell> [--n-max N] [--collect-only]`, `--atlas --close <store>`, `--atlas --verify <store>`: the atlas build OUTSIDE the suite (the `ell = 691` build is of the order of an hour and is not a registered validation); `ATLAS_CELL` / `ATLAS_SUMMARY` / `ATLAS_TIME` / `ATLAS_STORE` lines; driven by `docs/matrix-engine/tools/atlas_build.sh` under a `git status` guard. `main.cpp` dispatches into it (rule 1) |
| `HeckeTangentPair.{h,cpp}` | E6 cont. (slot 7, WP-5) | `[E]` Stage 6 / `cl:bk`, `[X]` §6.2 E6 (`[XA M7]`): for each cuspidal system of a store, `dim_K m/m^2` and a greedy generating set of ANEMIC operators at the multiplicity-`>= 2` weight `n = w - 2 + ell^2 - 1`, run with `u_ell = off` (U_ell would split the anemic ideal); `theta`-twists inherit their base's; `deferred` beyond `n = 1000` (every cuspidal system at `ell = 691`). `hecke_tangent_pair/v1`, printed only -- no store, record or golden key. Measured: `ell = 11`, `Delta` at `n = 130`: `D = 4`, multiplicity 2, **embdim 0** (a split block); `ell = 13`, `n = 178`: `D = 6`, multiplicity 3, embdim 1, generated by `T_2` |

Planned ([X] §1.1 v1.7, not yet written; `HeckeLocalAlgebra` landed in slot 2, WP-4, `HeckeGenerationCertificates` in slot 2, WP-5, `HeckeBoundary` in slot 2, WP-7, `HeckeSturmSweep` and `HeckeUEllBudget.h` in slot 3, WP-2 / WP-1; `HeckeConeAction` landed in slot 3, WP-6, **under `Hecke/`**, not here — [X] v1.7 §1.1): `HeckeModuleStructure`, `HeckeExtRecord`, `HeckeExtReport`, `HeckeExtDriver`,
`HeckeInvolution` (E3, slot 4), `HeckeContragredient`,
`HeckePairing`, `HeckeReduction`, `HeckeAtlas` (`HeckePresentation` landed in slot 6, WP-2).
*(Slot-6 close: `HeckeAtlas` landed here in slot 6, WP-8; `HeckeContragredient` and `HeckePairing` landed in slot 6, WP-1 / WP-3, **under `Hecke/`**; `HeckeExtRecord`, `HeckeExtReport`, `HeckeExtDriver` in slot 4 and `HeckeInvolution` under `Hecke/` in slot 4. Still planned: `HeckeModuleStructure`, `HeckeReduction`.)*

## `HeckeSturmBound` in one paragraph

`B'(k, ell)` is the **least prime bound** `B` for which the `F_ell`-*algebra* generated in
`End M_k(F_ell)` by the reductions `T_q mod ell`, `q ≤ B` prime, computed on the Miller
basis, has dimension `dim M_k` ([E] `def:bprime`). It exists iff
`ell ∤ [T^sat : T_Z(M_k)]`, and when it does, every `T_q` is a `Z`-polynomial in the
`T_{q'}`, `q' ≤ B'`, plus `ell` times an element of `T_Z(M_k)` ([E] `cl:bprime`) — which is
what turns the Sturm sweep of certificate (c) into a proof on the range of
[E] `cl:sturmmodl`. The weight is `k = n + 2`.

The unit is **pipeline-free**: `E_4 = 1 + 240 Σ σ_3`, `E_6 = 1 − 504 Σ σ_5` and
`Δ = q ∏ (1 − q^n)^24` are formed with every coefficient reduced mod `ell` as it appears
(no bignum, and no division by 1728, so `ell ∈ {2, 3}` are fine); the Miller monomials
`E_4^a E_6^b Δ^j` are unitriangular over `Z`, so the echelon basis needs no pivoting at any
`ell`; `T_q` is read off from `a_m(T_q f) = a_{qm}(f) + q^{k−1} a_{m/q}(f)` at precision
`L = q·dim M_k + 1`, asserted at every use; and the closure is
`hecke::algebra_closure` in `Hecke/SubspaceOps`, shared with E1a's `HeckeLocalAlgebra`
([XA m16]) so that this unit never reaches for `HeckeModuleDecomposition`.

It is the **first implementation of the resumable-state contract R3** ([X] §1.3.1):
`state()` returns a `SturmState` that is a deterministic function of `(k, ell, cap)`;
`resume(new_cap)` continues from the largest processed prime; a cap below it is refused with
`"resume: budget below reached state"`; a resume on a complete state is a strict no-op; and
`budget_history` records every run/resume. Resume-invariance holds because the budget decides
only *where* the ascending prime sweep stops.

## `HeckeLocalAlgebra` in one paragraph

Fix a block `V = H_m` of dimension `D` over `F_ell` that `hecke::CertifyLocality` has
certified local. `ComputeLocalAlgebra` runs [E] `alg:stage3`: (1) `A_S = F_ell[t_p]` inside
`End(V)` as an `F_ell`-subspace of `F_ell^{D^2}` (`hecke::algebra_closure`, shared with
`HeckeSturmBound`), kept as an RREF coordinate matrix so membership is one solve;
(2) the coefficient field `K = F_ell[ŵ]` and the generators `n_p = t_p − P_p(ŵ)` for
**every** `p`, the witness included — `ŵ` and the `P_p` are **consumed from the locality
verdict**, never re-derived differently, which is why the call sets
`force_matrix_path` (the `k = 1` analytic shortcut deliberately never forms `ŵ`) — then
`m_S = (n_p)_p` as an ideal of `A_S` (**no separate `π_w(w)` generator**), its powers,
`ν`, `soc A_S = Ann_{A_S}(m_S)`, the Gorenstein verdict `dim_{F_ell} soc A_S = f` and the
Hilbert function; (3) the tangent test **over `K`** — the `K`-span of `n̄_p` is the
`F_ell`-span of `{ŵ^i n̄_p : i < f}` — giving `embdim = dim_{F_ell}(m_S/m_S²)/f`, the
`active` table and the minimal `K`-spanning subset (greedy in ascending `p`).

**Certificate (a)** ([E] `cert:cyclic`) is `CyclicityCertificate`: `dim_{F_ell} V/m_S V = f`
makes `V` cyclic over `A_S`, hence `End_{A_S}(V) = A_S|_V = T_m` — *no further `T_q` can
change the module structure at `m`*. Its `verify()` **re-derives** everything it concludes
from ([X] §1.3 R17): it recomputes the closure from the stored `t_p` and compares the basis
**entrywise** (the RREF is canonical, so this is a bit test), recomputes each `n_p` and its
nilpotency, recomputes `m_S` as the ideal `(n_p)_p`, recomputes `dim m_S V` as a rank, and
recomputes both the least generator index and `A_S v = V`. A truncated basis, a decremented
dimension or a perturbed matrix therefore fails — `HeckeExtAlgebraValidation`'s ten-way
tamper arm asserts the exact failing clause of each.

`SameMaximalIdeal` is Stage 2's **cross-block evaluation test** ([XA M5]): blocks `B`, `B'`
carry the same `m` iff, with `B`'s own `c`, `π_w` and `{P_p}` and `N'` taken for the
**tested** block (`ell^{N'} ≥ dim B'`), `π_w(w'|_{B'})` and every `t_p|_{B'} − P_p(ŵ')` are
nilpotent. It needs no shared candidate sequence and no "up to Galois" clause.

**There is no resumable state here.** [X] §1.3.1 R3.3 lists five instances and the algebra
construction is not one of them: it is a fixed, unbudgeted `O(k · dim A_S · D³)` closure.
The budgeted item in E1a is the **commutant** (`CommutantState`, R3.3 item 3), which is
WP-5's.

## `HeckeGenerationCertificates` in one paragraph

Certificate **(b)** of [E] `cert:comm`, the second of the three routes to
`A_S = T_m`. `C := End_{A_S}(V)` always contains `A_S`, and `dim C = dim A_S`
proves `A_S = T_m`. The unit first decides **whether (b) can succeed at all**,
by the socle bound of [X] §2.2 Stage 4(b) step 1: with `t = dim_F(V/m_S V)` and
`s = dim_F soc V = dim_F ⋂_p ker n_p`, both in units of `f`,

```
dim C  >=  dim A_S + f*t*s - dim_F soc A_S,
dim C  == dim A_S  requires  f*t*s <= dim_F soc A_S,
```

so (b) is `not_needed` on a cyclic `V` (where (a) certifies and the bound is an
equality) and `excluded_by_socle_bound` — a **proof of failure**, recorded with
its four numbers — on every Gorenstein non-cyclic `V`, the `(10, 691)` shape
included (`t = s = 2`, `f = 1`, `dim soc A_S = 1`, bound `5`, and `dim C = 5`
exactly). (b) can beat (a) only on co-cyclic-like modules over a
non-Gorenstein `A_S`, e.g. `V = A_S^∨` for `A_S = F[x,y]/(x,y)²`.

Where the gate admits it, the system `X t_p = t_p X` is solved **not** on `D²`
unknowns but inside the explicit basis of `End_{F_ell[a]}(V)` for a fixed
cyclic-most `a ∈ A_S` — the first element of the fixed candidate list
(`b_1 … b_{dim A_S}`, then the prefix sums) whose minimal polynomial on `V` has
maximal degree — in its rational canonical form
`R^{-1} a R = ⊕_i companion(π_a^{s_i})`, giving
`u = deg π_a · Σ_{i,i'} min(s_i, s_{i'})` unknowns. The **naive** commutant
(`O(k D⁶)`) ships as `NaiveCommutantDim`, the validation oracle only: an
`over_budget` run is never a silent fall-back to it.

`CommutantState` is R3 instance 3. The `k·D²` rows are formed in the fixed order
"operators ascending `p`, then the commutator entry `(i, j)` row-major" and
streamed into a canonical incremental RREF, and `rows_done` is a **pure function
of the budget**, so a prefix cut is exactly what R3.1 item 3 asks for:
`run(b₁)` then `resume(b₂)` gives the same state, certificate and outcome,
entrywise, as a fresh `run(b₂)` — modulo `budget_history`, which must differ.
`resume` refuses a lower budget with `"resume: budget below reached state"` and
is a strict no-op on a complete state.

`verify()` re-derives everything it concludes from ([X] §1.3 R17): route (a)'s
embedded certificate re-derives `A_S`, the `n_p` and `m_S`; `soc V`, `soc A_S`
and the bound are recomputed; the candidate list is rebuilt and `a` re-chosen;
`R^{-1} a R` is compared **entrywise** with the companion blocks; the `u` basis
matrices are **rebuilt by the fixed formula** and checked to commute with `a`
and to be independent; and the system is re-formed and row-reduced, so `dim C`
is recomputed and never read. A partial (`over_budget`) certificate verifies
exactly the partial statement — that the stored reduction is the RREF of the
first `rows_done` rows — and nothing more (R3.1 item 5). What it deliberately
does **not** pin is the RCF basis: rescaling a cyclic block of `R` gives another
valid basis and still verifies, because `dim C` is basis-independent.

`generation_verdict(a, b, c)` is the route-combining map of [X] §1.3.1 R3.2 in
the default configuration: `cyclic` / `commutant` / `sturm(B')` with
`final = true`, otherwise `final = false` with a reason from R3.2's fixed list
(`closure_unreachable`, `budget_exhausted:commutant`, `sturm_unavailable`,
`certificate_failed:sturm`). Owner decision OQ8: such an ideal is **emitted**
with its reason and, where a budget was involved, its `resumable` object — never
withheld. Route (c)'s verdict is an input here; the Sturm **sweep** is E1b's.

## `HeckeSturmSweep` in one paragraph

**Certificate (c)** ([E] `cert:sturm`) on the pipeline's own blocks. Given a block `V = H_m`
that `CertifyLocality` accepted and its `LocalAlgebra` `A_S`, the sweep computes
`B'(k, ell)` with `HeckeSturmBound` (`k = n + 2`), asks an operator source for `T_q|_V` for
**every prime `q ≤ B'`**, and tests membership in `A_S` by one reduction against the RREF
coordinate basis (`hecke::RrefRowSpan::contains`). On the range of [E] `cl:sturmmodl` — a
non-Dickson maximal ideal of `X_4` — every `T_q` in `A_S` **proves** `A_S = T_m`; off that
range the identical computation is a heuristic and the certificate says so (`proof_range` is
an input the caller asserts, because identifying Dickson-type `m` needs the residue atlas of
E6). **The prime list is `q ≤ B'` and nothing else**: `ell` is swept iff `ell ≤ B'`, since
[E] `cl:sturmmodl` writes `T_ell` too as `F(T_{q' ≤ B'}) + ell·t'`. `S_sep(ell)` is E6's.

**Never `B(k) = ⌊k/12⌋ + 1`**: at `(n, ell) = (22, 5)` and `(26, 7)` the `{2, 3}`-algebra is
proper, and the validation reproduces exactly that on the pipeline's own `H^1` — one block
rejects `T_5` (resp. `T_7`), while the `B'`-sweep on the `{2,3,5}` (resp. `{2,3,5,7}`)
decomposition passes on every block.

**The two `sturm.*` modes** are `Hecke/HeckeSettings`' third and fourth registered
mathematical keys (`sturm = auto | off | cap=<B>`, `sturm_unavailable = nonfinal |
heuristic`; appended after `u_ell`, so the defaults still fingerprint to `0`). Under the
default `nonfinal` a `none_at_cap` bound sweeps **nothing** and the ideal is `final = false`
with reason `sturm_unavailable`; under `heuristic` the primes `≤ cap` are swept and the ideal
is `final = "heuristic"` with `sturm_primes` — a value the defaults never produce.

**Stage 1** ([X] §2.2) is the other half: `ComputeStageOneOperators(n, ell, u_ell)` returns
`S_0 = {2, 3, 5, 7} ∪ {ell iff the frozen RSS model of `HeckeUEllBudget.h` admits U_ell}`
together with the `UEllVerdict` — recorded, never inferred. `u_ell = off` removes `ell` even
when it is one of the four good primes, because at `ell ≤ 7` the good prime `ell` **is**
`U_ell`. Where the proof list contains `ell` and the budget refuses it, (c) is `unavailable`
with `u_ell_blocked`, and no partial sweep is published.

**R3**: the sweep's budget is `sturm.cap`. The embedded `SturmBound` object is kept alive, so
`resume(new_cap)` *continues* it (never a restart) and re-sweeps under the new `B'`; a lower
cap is refused with `"resume: cap below reached state"`; a resume on a complete state is a
strict no-op; `budget_history` lists every cap; and `run(3) + resume(36)` at `(22, 5)` equals
a fresh `run(36)` entrywise, state and certificate, modulo `budget_history`.

`verify()` re-derives ([X] §1.3 R17): the bound's own certificate, certificate (a), the prime
list **rebuilt from `B'` / the cap by an independent sieve**, the membership of every stored
`T_q|_V` against an `A_S` recomputed from the stored `t_p`, and `generators_certified`,
`failing_prime` and `outcome`. What it does *not* do is re-run the cohomology: the provenance
of `T_q|_V` is the pipeline's own exact-solve invariance certificate.


---

## `HeckeExtFinal.h` / `HeckeModuleStructure` / `HeckeExtRecord` + `HeckeExtReport` — the extension record (E2a, slot-4 WP-1)

**The one tri-state.** [X] §1.3.1 R3.2 ends "the tri-state is ONE type:
`HeckeGenerationCertificates` reuses `SturmFinal` until E2's record layer owns `Final`, to
which both units then migrate". `HeckeExtFinal.h` is that type — `Final {True, False,
Heuristic}`, `Finality {value, reason}`, `FinalMode`, the **fixed reason list** and the
predicates `IsFinalReason` / `IsResumableKind` — and `SturmFinal` / `SturmFinality` are now
aliases of it. The emitted JSON strings (`true` / `false` / `"heuristic"`) are unchanged, so
nothing any shipped certificate writes moved. The header is dependency-free and entirely
inline, which is why `check_heckeext_includes.sh` rule 3 admits it in `HeckeSturmBound`, on
the same grounds as `EngineCheck.h` and `Fnv.h`.

**Stage 5.** `HeckeModuleStructure` ([E] `alg:stage5`) takes one `LocalAlgebra` and produces
the Loewy and socle series in units of `f`, the Jordan type of each `n_p` in units of `f`,
and the structure maps `θ_j` per consecutive layer pair and per generator — rows the
bottom-layer basis, columns the top-layer basis, exactly [E] §7's convention — with their
per-generator and stacked ranks. Every subspace is a canonical `hecke::RrefRowSpan` and every
layer basis is the greedy selection, in that canonical order, of the rows of `m^j V`
independent modulo `m^{j+1} V`; the coordinates of `n_p b̄` come from one exact solve against
`[bottom layer | m^{j+2} V]`. `verify()` re-derives Stage 5 by calling the **same** builder on
certificate (a)'s stored data and comparing entrywise (R17). Clause (4) of `alg:stage5` — the
canonical submodules of D6 — needs E3 / E4 / E5 objects and is not in this unit; the record
writes `"canonical": {"status": "unavailable"}`.

**The record.** `hecke_ext/v1` is a **separate writer** from `Hecke/HeckeReport`'s
`hecke_report/v2` and slot 4 does not touch the latter ([X] §1.6: the golden harness hashes
its text, so a new key there is a golden re-baseline). `HeckeExtRecord` carries the top-level
metadata ([E] §7's own key names: `n_sym`, `ell`, `weight`, `sign`, `commit`, `model`,
`iota_sign`, `coefficient_ring`, `settings`, `resumable_items`), **one object per complex**
with its own `q_max` / `truncated` / `q_max_source` / `degrees` *derived from that complex's
own degree count* ([C] `sec:recordschema` item 5 — `boundary` and `pair` are
`{"status": "unavailable"}` until slot 5's integration), the `groups` object, and one
`IdealRecord` per maximal ideal with `local`, `witness_poly`, `coefficient_field`, the
`algebra` invariants, `certificate` / `certificate_routes` / `generators_certified`, the
`sturm` object ([E] §7's four keys exactly, with `proof_range` recorded as an **input** —
`proof_range_source`, never promoted by a result), `u_ell`, `tangent`, the stored `t_p|_V`,
`modules` (Stage 5) and `final` / `final_reason`.

`is_final(FinalInputs, MathSettings)` is [X] R3.2's map. Its **generation half calls E1a/E1b's
own `generation_verdict`** rather than re-implementing it, so the record's verdict and the
certificates' cannot drift apart; what E2a adds is `not_applicable` (`n = 0`, [E] Def. 2.8),
interruption, locality (a budget survivor is `budget_exhausted:locality`, never `not_local` —
budget mode never *proves* non-locality), a failed certificate clause, and the two
user-selected heuristic modes: `sturm_unavailable = heuristic` for a cap-hit sweep, and
`final_mode = sturm_heuristic` for a complete sweep off the proof range. In the DEFAULT
configuration `"heuristic"` is unreachable, which the validation asserts over the whole
9216-point input grid.

**The sidecar** (R3.1 item 6). `WriteHeckeExtReport(record, path)` is the only place that
knows the record stem: a resumable state whose JSON exceeds 64 KiB is written to
`<stem>.state.<kind>.json` beside the record and referenced by its **basename** and its
FNV-1a-64 (`Fnv.h`, the one byte-wise FNV); `BuildHeckeExtReportJson`, which has no stem,
inlines the state and flags `state_oversize` rather than emitting a path that names no file
(the slot-1 WP-3 rule). A second oversize state of the same kind gets `.<i>` — a documented
deviation from [X]'s literal `<record-stem>.state.<kind>.json`, since one record may carry one
commutant state per ideal.

**`commutant_budget` and `final_mode`** are `Hecke/HeckeSettings`' fifth and sixth registered
mathematical keys, appended after `sturm_unavailable` so the all-defaults fingerprint is still
`0`.  (Slot-5 WP-7 appends the seventh and eighth, `field_polynomial` and `split_field_report`
-- see `HeckeCanonicalField` below.) `kDefaultCommutantBudget = 2·10^10` lives in `HeckeSettings.h` (`Hecke/` may not include
`HeckeExt/`) and `CommutantRoute::kCommutantBudget` is a `constexpr` **alias** of it; a
`CommutantRoute` constructed with `budget = 0` reads the setting in force, so a resume with a
raised budget is a new recorded setting ([X] R3.1 item 4) and a lower one is refused.
`final_mode` is registered because `is_final` — [X] §1.5's own named consumer for it — ships
in this package.

**The campaign.** `CampaignSpec` parses every registered mathematical key beside its own
(validated by `HeckeSettings` and nowhere else), the driver installs them per job with
`ScopedMathSettings` (thread-local, propagated onto `run_concurrently()` tasks), the per-job
JSON gains the `settings` block, and a **non-zero** fingerprint appends `_s<16 hex>` to
`Job::id()` after any `_q` tail and ` s=<16 hex>` to the shape signature. At the defaults the
fingerprint is `0` and nothing is appended, so `n16_ell5_p2_r1` and every published file name
are byte-unchanged.

---

## `HeckeSignSplit` — the sign split of the Hecke module (slot 4, WP-3, phase E3)

`HeckeSignSplit.{h,cpp}` is the ext-layer half of E3. `Hecke/HeckeInvolution` (pipeline side)
builds `iota^*` of [C, `def:iota`](iv) — `iota^* := -iota^*_geo`, the v1.2.0 normalisation of
owner decision OQ3, so that "+" is **by definition** the Eisenstein side — and splits the stored
`C/Z/B/H` objects. This unit takes the *result* (the matrix of `iota^*` on `H^q` and the two
eigenbases, in the stored basis) together with the computed `A^{(q)}(T_p)`, and does the
Hecke-module half. It takes **matrices only**, no pipeline object, so it can be exercised
without constructing a cohomology; `MatrixT<uint32_t>` throughout ([X] §1.1).

* **[C, `obl:iota`](v)**: `[iota^*, A^{(q)}] = 0`, exactly, per operator. **Asserted at
  `ell >= 5`**, where [C, `prop:iotacommutes`](3c) proves it (`6 in R^x`); **recorded at
  `ell = 3`**, where the only route is (3b) and its hypothesis `tau^iota = tau` is a separate
  obligation — and the restriction `A^{(q)}_+-` is formed **only where the commutation holds
  exactly**. At `ell = 2` the whole split is `not_applicable` ([C, `prop:iotasplit`](2)) and
  nothing is formed.
* `A^{(q)}_+-` by an **exact solve** (`hecke::SolveColumnsExact`, which throws rather than
  returning a wrong answer, so the solve *is* the invariance certificate).
* the **per-sign** primary decomposition (`hecke::HeckeModuleDecomposition` on the restricted
  family) beside the unsplit one, and cyclicity per sign through slot-2 WP-4's generation
  certificate (a).
* the **Eisenstein line** of [E] D6: for a block `V = X^+_{4,m}` of the plus side,
  `soc(V) = Ann_V(m_S) = intersect_p ker n_p` and `m_S V = sum_p im n_p` are computed and
  asserted equal (that equality *is* "the bottom Loewy layer"), and the line is **defined** when
  that space has `F`-dimension 1 (`dim_{F_ell} / f == 1`), **`"undefined"`** otherwise — which is
  exactly what the unsplit `H^1` at `(10, 691)` gives, its socle being 2-dimensional because the
  Eisenstein and congruent cuspidal reductions are not separable mod `ell`.

`SignSplitCertificate::verify()` re-derives all eight items (it recomputes the eigenvalue
equations, the commutation verdicts and every restriction rather than reading them). Golden
collectors — `charpoly_iota^q`, `rank_iota^q`, `dim_Hplus^q`, `dim_Hminus^q`,
`eisenstein_line_defined` (tier A) and `iota^q` (tier B) — are **defined here and called only
from `HeckeInvolutionValidation`**: slot 4 emits no golden entry, and `iota^*` occurs in no
golden object ([C, `rmk:iotasign`]).

`HeckeInvolutionValidation` (**`standard`**, ~2.1 s, 2818 checks) is the acceptance. Its oracle
is `hecke::iota_oracle` — the involution `HeckeIotaSignValidation` builds from [C, `def:iota`]
alone, exported by this package **value-preservingly** (that validation's printed lines are
byte-identical before and after the export) — and the comparison is **entrywise** in every
degree on its 77-cell grid. The production object reproduces the reference sign line field for
field:

```
IOTA_SIGN_PRODUCTION eisenstein=plus (geometric=minus) cells=77 signed=74 route_a=74
  route_b=53 both=53 split_match=66/66 split_mirror=0/66 dickson=11 h0_anchor=-1 (geometric=+1)
```

Route (b) runs through the **production** restriction (`Hecke/SL2ZBoundaryPair`), not the
oracle's own cusp path, so the 53 agreeing cells are an independent confirmation of the sign.

---

## `HeckePredictions` and `GloverFactors` in one paragraph each (slot 4, WP-7 — E6)

**`bernoulli_mod`** is what retires the pinned `num(B_k/k)` table. For `0 <= w < ell - 1` the
Bernoulli number `B_w` is `ell`-integral — by von Staudt–Clausen `den(B_w) = prod_{(p-1) | w} p`,
and `(ell - 1) | w` is impossible in that range — so the standard recursion runs verbatim in
`F_ell`: every `B_j` that enters has `j <= w < ell - 1`, and the only division is by
`w + 1 <= ell - 1`. No bignum, no rationals, `O(w^2)`. At `w >= ell - 1` the unit **refuses** with
the named `BernoulliOutOfRange` rather than returning a residue that does not exist. The Kummer /
Harder predicate `ell | num(B_k/k)` is total on even `k >= 2` and every prime: `(ell - 1) | k`
gives FALSE by von Staudt–Clausen (`ell` sits in the denominator, so it cannot divide the
numerator); `k < ell - 1` is decided directly; and otherwise the **Kummer congruence**
`B_k/k == B_{k'}/k'` for `k' = k mod (ell - 1)` decides it, `B_k/k` being `ell`-integral exactly
because `(ell - 1)` does not divide `k`. Odd `k >= 3` is TRUE (`B_k = 0`) and `k = 1` is FALSE
(`num(B_1/1) = -1`); `k = 0` is the one inapplicable input.

**`GloverFactors(n, ell)`** returns the Jordan–Hölder factors of `Sym^n(F_ell^2)` over
`F_ell[GL_2(F_ell)]`. Multiplication by the invariant `theta = x^ell y - x y^ell` (the product of
the `ell + 1` lines of `P^1(F_ell)`, on which `g` acts by `det g`) is an injection
`Sym^{n-ell-1} (x) det >---> Sym^n` whose cokernel has dimension `ell + 1` and is the principal
series obtained by restricting a degree-`n` form to each line; the JH factors of
`Ind_B^G(chi_1^s chi_2^t)` in defining characteristic are `Sym^r (x) det^t` and
`Sym^{ell-1-r} (x) det^s` with `r = (s - t) mod (ell - 1)` — the trivial/Steinberg pair when
`r = 0`. With `(s, t) = (n, 0)` that is the whole recursion, `O(n / ell)` steps. The **warrant is
the check, not the derivation**: `HeckePredictionsValidation` arm (H) compares the factor list
with the Brauer character of the *real* `rho_n` on every `ell`-regular class of `GL_2(F_ell)` —
eigenvalues read off the characteristic polynomial of the matrix
`ComputeSymmetricPowerMatrixFM` returns, by Hessenberg reduction over `F_ell` and root extraction
over `F_{ell^2}`, with the Hessenberg polynomial itself cross-checked against
`det(x I - rho_n(g))` at every `x in F_ell` — on 3 848 `(class, n)` cells. The two Teichmüller
exponent multisets agree **formally** in `Z[x]/(x^{ell^2-1} - 1)`, which is strictly stronger than
the equality in `Z[zeta]` that Brauer-character theory needs, so no cyclotomic reduction appears
in the validation.


---

## `HeckeExtDriver` --- the Stage 1-5 orchestration (slot 4, WP-2 / phase E2b)

*Appended by slot-4 WP-2. Nothing above this line was edited.*

`HeckeExt/HeckeExtDriver.{h,cpp}` is what RUNS a cell. WP-1 shipped the record
(`hecke_ext/v1`) and every piece that fills it; this is the orchestration of [X] Sec. 2.2
Stages 1-5 that produces one from `(n_sym, ell)`:

| Stage | Unit | Timed as |
|---|---|---|
| setup | `CreateSL2ZFundamentalDomainTriangulation` + `GenerateSL2ZRelationsWithGroupData` + `EquivariantCohomologyOnK::compute_cohomology()` + `HeckeOperatorFamily` | `setup_s` |
| 1 | `ComputeStageOneOperators` (`S_0 = {2,3,5,7} u {ell iff UEllDecision admits}`) + `family.prepare(p, 1)` + `family.T_p(p)` | `stage1_s` |
| 2 | `hecke::HeckeModuleDecomposition::compute()` + `certify()`, once per bulk degree with `dim H^q > 0` | `stage2_s` |
| 3 | `ComputeLocalAlgebra` per block | `stage3_s` |
| 4 | `CyclicityCertificate` (a), `CommutantRoute` (b), `SturmSweep` (c), combined by `generation_verdict_with_sweep` | `stage4_s` (`sturm_bound_s`, `streamed_s` inside it) |
| 5 | `ComputeModuleStructure` | `stage5_s` |

**ONE cohomology, ONE family, ONE pipeline alive at a time.** The family's `pipeline_cache_`
never evicts, so only `S_0` goes through it. A Sturm prime `q` outside `S_0` is **streamed**:
`StreamedOperatorSupplier` (added to `HeckeSturmSweep`) builds
`HeckeCosetTables -> HeckeTauTable -> HeckeOperatorOnK` in a local scope --- the two table
constructions under `hecke::tau_construction_mutex()`, the `compute()` under
`ScopedThetaEqCertification(false)` + `ScopedMemoryLean(true)` --- keeps only
`A^{(q)}(T_q)` and destroys the pipeline before returning. `ReuseBeforeStreamingSupplier` is
[X] Sec. 2.2 Stage 4(c)'s "reuse before streaming" made explicit: the primes of `S_0` go to the
family, everything else to the streamed route, and neither is consulted for a prime the other
owns. The driver memoises the streamed `A^{(q)}(T_q)` per `(degree, q)`, so a cell with several
blocks builds each streamed pipeline once and keeps only its `d_q x d_q` result.

**Stage 3 consumes the block's own locality certificate.** Since E0b the decomposition's
refinement loop enforces [E] Cert. 5.2 itself and stores `witness_poly`, `w_hat_exponent`,
`character` and the witness on the `PrimaryBlock`, so re-running `hecke::CertifyLocality` inside
`ComputeLocalAlgebra` searches again for a witness that is already proved.
`ExtDriverOptions::consume_block_locality` (default **true**) rebuilds the verdict from the block
--- `w_hat = W^(ell^N)` by `hecke::FrobeniusPower` being the one thing a `PrimaryBlock` does not
store --- and hands it to `LocalAlgebraOptions::precomputed_locality`. Every check after Step 0 is
unchanged, `pi_w(w_hat) = 0` and the nilpotency of every `n_p` included, so an adopted verdict
that is not a certificate still fails loudly. Setting the flag to `false` restores the pre-E2b
route, which `HeckeExtRamanujanValidation` arm 5 keeps as the ORACLE: the two must agree on
`w_hat`, every `n_p`, every `A_S` basis element **entrywise** and on the whole record text.

**The `u_ell` resume** ([X] Sec. 1.3.1 R3.3 item 4). A refused `U_ell` leaves
`u_ell.resumable {kind: "u_ell", complete: false}` on every ideal with its `budget_history`.
`resume_u_ell(budget_mb)` re-runs Stage 1 with `U_ell` attached to the SAME family (its cache is
keyed by `(p, r)`) and then Stages 2-5. A budget at or below the one reached is refused with
`"resume: u_ell budget below reached state"`; a resume on an included verdict is a strict no-op.
A resume to `kDefaultUEllBudgetMb` installs `auto` at the shipped default --- which IS the default
setting --- so the resumed record is byte-identical to a fresh `u_ell = auto` run, fingerprint
included.

**`--bench ext <n> <ell> [p,q,...]`** (`Bench/BenchDriver.cpp`) drives it through the public API
and prints `pipeline_s`, `decomp_s`, `ext_s`, `sturm_bound_s`, `streamed_s`, the streamed count
and RSS, `ext_ratio = ext_s / (pipeline_s + decomp_s)` --- each phase's min over the repeats, the
per-repeat ratios printed beside them --- and the informational `ext_over_hecke1_ratio` against a
one-prime `hecke` run of the same cell. The default family is `2,3,5`.  Since slot-5 WP-7 it also
prints `canonical_field_s`, `own_field_s`, `canonical_field_ideals`, `canonical_field_ok` and
`canonical_field_r_max` --- the measurement E6's Stage-6 performance gate is stated over
(`canonical_field_s <= 0.10 * ext_s` and `<= 2 * own_field_s`), REPORTED AND NEVER ACTED ON: the
gate is E6 acceptance's, in slot 6, and WP-7 does not flip the `split_field_report` default on it.
Neither column enters the record and neither is hashed.

**Validation**: `HeckeExtRamanujanValidation` (standard) --- [E] `cl:ramanujan` at `(10, 691)`
with `U_691` inside the frozen budget, the streamed arm at `(22, 5)` and `(26, 7)`, the `u_ell`
resume arm, the fast-path oracle arm and the `n = 0` / odd-`n` edges.

## `hecke_ext/v2` and `HeckeExtComplexRecords` (slot 5, WP-1 — phase E2 *cont.*)

**Why the version bump.** `[OD]` slot-4 close-out 5 (2026-09-22) ruled the record-shape repair
APPROVED and ordered it **before the stage's first golden emission of an ext record**, on the
discipline `hecke_report/v1 → v2` set in slot 2: the change alters the **value** of an existing key
(`resumable_items`), so it is a version bump and not an in-place amendment. It is free now and a
re-baseline of every emitted record if deferred past the emission.

**Five things move.**

1. **`u_ell` is the CELL's.** In v1 the `UEllRecord` sat inside every `IdealRecord`, so a
   four-ideal cell emitted one decision four times and `resumable_items()` counted one refusal
   four times ([E] `sec:schema` v1.5.0 states the defect and books the repair). In v2 it sits on
   `HeckeExtRecord` with **one** `resumable`; `HeckeExtRecord::all_resumables()` joins
   `IdealRecord::all_resumables()`, and the count is the record's own incomplete items plus each
   ideal's. `HeckeExtRecordValidation` arm 11 is the arm that proves it — a four-ideal cell with
   `U_ell` refused emits the verdict once and reports `resumable_items = 1`.
2. **A per-ideal `groups` list** ([E] `sec:schema` v1.5.0's second booked repair). Stage 2 already
   ran the cross-block evaluation test (`[XA M5]`, `SameMaximalIdeal`) and the driver already
   counted its matches; v1 had no key that **reported** them. Each entry names the ambient group,
   the ideal the test was run from, the witness (`π_w` low → high, the exponent `N`, and the `N'`
   the test used on this block) and, on a non-match, the `EvaluationFailure` code; the entry for
   the block's own group is the identity match. `modules` keeps its "one entry per ambient group"
   shape, so nothing else moves.
3. **`complexes.boundary` and `complexes.pair` are computed objects.** Each carries its own
   `q_max` / `truncated` / `q_max_source` / `degrees`, derived from that complex's own degree
   count; the pair additionally carries `derived_from = [bulk, boundary]` ([C] `sec:recordschema`
   item 5, "a complex built from others"), one `les_maps` entry per recorded spot of
   [C] `eq:coneles`, and — since v2 — the two flags item 6(e) reserved, `phi_flat_identity` and
   `residual_flat_zero`, beside `action_record: "hecke_cone_action/v1"`. All three are diagnostics
   under rule A6: recorded, never branched on, and a `false` is conformance.
4. **The per-sign (E3) objects**: an `iota` array with one entry per computed degree (status,
   `dim_H`, `dim_plus` / `dim_minus`, `rank_iota`, `charpoly_iota`, the commutation counters and
   the Eisenstein line), the per-sign groups `H<q>_plus` / `H<q>_minus`, an ideal's own `sign`,
   and two further `HeckeExtRecord`s with `sign = "plus"` / `"minus"`.
5. **`canonical` per ambient group**, with the Eisenstein line's `CanonicalPosition`; and
   `groups` gains the fourth integer scalar of [E]'s closed list, `dim_H1_shriek`, emitted as
   `null` where the boundary complex that defines it is unavailable, so the list is a constant of
   the schema and not a function of the cell.

**`HeckeExtComplexRecords.{h,cpp}`** is the record layer's view of the E3 / E4b / E5 objects, and
the only place the two vocabularies meet. Every function is a **pure translation** from an
already-computed pipeline object to the record fragment that describes it; nothing there builds a
cone, a Hecke operator or an involution, so what a caller attaches is the caller's cost. **One**
computed `EquivariantMappingCone` carries everything the boundary and pair objects need
(`BoundaryComplexRecord`, `ConeComplexRecord`, `BoundaryGroups`, `ConeGroups`, `ConeLesSummary`,
`ConeDimH1Shriek`), so the record re-derives none of them from a second object; `ConeLesSummary`'s
`exact` is a conjunction over the **asserted** spots only, because an unasserted spot records a
cokernel bound and claims nothing. It includes pipeline headers in its header exactly as
`HeckeExt/HeckeBoundary.h` does — guard rule 1 is about the reverse direction — and no `Console/`,
`Cuda/`, `Engine/` or `BlockFlatMatrix.h`.

**Two new driver options, both OFF by default**, so a default cell run costs exactly what it cost
in slot 4 and neither can change a value the record already carried: `with_cone` builds
`Hecke/SL2ZBoundaryPair` and its cone on the driver's **own** cohomology and fills the boundary and
pair objects, the two groups, `les` and `dim_H1_shriek`; `with_sign_split` builds the production
`Hecke/HeckeInvolution` and, per degree, `HeckeExt/HeckeSignSplit`, and fills the `iota` objects,
the per-sign groups, the Eisenstein line's position and the two per-sign records. Routes (b) and
(c) are deliberately **not** run per sign — certificate (c) is a statement about the Sturm algebra
on the unsplit `M_k`, so a per-sign sweep would be a claim [E] `cert:sturm` does not make — and
they record `not_run`. `attach_cone_action(action)` takes a **computed** `HeckeConeAction` (the
driver never builds one: its standard-basis sweep is 88 % of its wall) and checks it is about the
driver's own cone. `resume_u_ell()` rebuilds both attachments with the record it rebuilds.

**[X] §6.2 E3 acceptance (c) is asserted, not recorded**: `run()` checks that the unsplit record's
dimensions are the sum of the two per-sign ones, degree by degree, and that
`dim H^q_+ + dim H^q_- = dim H^q` ([C] `prop:iotasplit`(1)). The per-**ideal** matching by
`SameMaximalIdeal` is the validation's, because it needs the local algebras.

**`verify()` gains codes 16–21**: an ideal's sign must be the record's; an `iota` entry must be
well formed (status enumerated, `dim_+ + dim_- = dim_H` on a split, no duplicate degree, a
`"defined"` Eisenstein line one-dimensional over `F`); an **unavailable** complex may carry no v2
content ([C] `sec:recordschema` item 6(c)); `dim_H1_shriek` needs its complex; an **included**
`U_ell` verdict has nothing to resume, and the reason must be one of the enumerated three; and a
claimed cross-block match must carry a witness, a non-match a failure code.

**Validation**: `HeckeExtRecordValidation` (`fast`, ~0.1 s, **556 checks**) — arms 11–16 on top of
slot 4's ten. Arm 15 runs the real `(10, 691)` cell through the driver with both flags on and finds
[E] `sec:tests`' own values: `dim H^1 = 3`, `dim H^1_c = 3`, `dim H^2_c = 0`, **`dim H^1_! = 2`**
([E] v1.4.0's correction — `H^1_! ≠ H^1` at a Kummer cell), `dim H^1_± = (2, 1)`, the Eisenstein
line **defined** on the plus side, `cyclic` on each sign and `not_cyclic` unsplit. Arm 16 is the
[X] §1.6 collector **inventory** WP-3 wires: the E3 / E5 names already exist in the units that own
the objects (`HeckeSignSplit::golden_objects()`, `EquivariantMappingCone::golden_objects()`,
`HeckeConeAction::golden_objects()`, `HeckeExt/HeckeBoundary::golden_objects()`, all shipped
*defined and never emitted* in slots 3–4), and v2 adds the record's own view of them
(`CollectIota` / `CollectBoundaryDims` / `CollectConeDims`, file-local in the validation TU). None
is reachable from `Golden/GoldenHarness.cpp`: slot-5 WP-1 emits no golden entry, and the slot's one
emission is WP-3's, strictly after this bump.

## `HeckeCanonicalField` in one paragraph (slot 5, WP-7 --- E6)

A decomposition block reports its residue data in its OWN field: the witness polynomial `pi_w`,
irreducible of degree `r = f` over `F_ell`, and one `P_p` per family prime with
`a_p = P_p(w-bar)` in `F_ell[X]/(pi_w)` ([E] `cl:resfield`).  That form is free --- it is what
Stage 2 already computed --- but it is not comparable across blocks or across runs, because every
block names its own field by its own `pi_w`.  Stage 6 re-embeds all of them into ONE field per
`(ell, r)`, fixed by `(ell, r)` alone: `C_{ell,r}` is the `CanonicalLess`-least monic irreducible
of degree `r` ([E] `def:canon`) and `F := F_ell[Y]/(C_{ell,r})`.

**Two facts about the ORDER are load-bearing.**  (1) `hecke::CanonicalLess` is "degree, then
`a.coeffs < b.coeffs`", and `std::vector`'s comparison starts at index 0, so the CONSTANT
coefficient is the MOST significant digit: the candidate odometer's rightmost digit turns
fastest, and `C_{2,3} = Y^3 + Y^2 + 1`, not the more familiar `Y^3 + Y + 1`.  The one prune is
the theorem `c_0 = 0 and r >= 2 => Y | f`, applied as a SUBTREE skip --- without it `(2, 30)`
would have to reject `2^29` candidates one at a time.  (2) In `alg:split` step 1 the fixed
candidate sequence (the elements of `F` in `CanonicalLess` order) STARTS AT `Y`: no `s` in
`F_ell` can ever split `pi_w`, because the roots are a single Frobenius orbit, `x |-> x + s`
commutes with Frobenius for `s` in `F_ell`, and both two-valued invariants the splitting
polynomial uses --- the quadratic character for odd `ell`, the absolute trace for `ell = 2` ---
are Frobenius-invariant, so the gcd is `1` or `f`.  That is a theorem, not a heuristic, and it is
worth `ell` rejected candidates per splitting step (0.47 s on one `(160, 691)` cell before the
skip, 2.0 ms after).

`CertifyCanonicalSplit` re-derives all nine clauses from `(ell, pi_w, primes, character)` alone
--- `C_{ell,r}` is RE-ENUMERATED, never read back --- and a validation tamper arm shows each
corruption failing exactly its own clause.  Mathematical refusals (a reducible `pi_w`, a degree
mismatch, an exhausted candidate sequence) are REPORTED in `failure`, never thrown, so a report
layer can fall back to the own-field form.

**Validation**: `HeckeCanonicalFieldValidation` (`fast`, ~0.5 s, 17 054 checks) --- seven arms,
every oracle independent of the unit.

---

## The record on the campaign (hecke-extensions slot 5, WP-2 / phase E2 cont.)

`hecke_ext/v2` is now campaign output. The seam is **`Campaign/CampaignExt.{h,cpp}`**, the only
`Campaign/` TU that includes a `HeckeExt/` header (`check_heckeext_includes.sh` rule 1 admits
`Campaign/` by name), and **no file of this layer changed**: the driver, the record and the writer
are exactly what slot-5 WP-1 shipped.

**What a `HeckeExt/` reader needs to know about it.**

* **The pass runs on the campaign's main thread, sequentially, before the job loop** — one
  `HeckeExtDriver` per `(n_sym, ell)` cell, attached to the cell's first job in plan order
  ([XA m14]). This is not a scheduling preference: `HeckeExtDriver::Impl`'s constructor builds a
  `SimplicialComplex` and an `SL2ZTessellationTables` **outside**
  `hecke::tau_construction_mutex()`, and that build drives the non-atomic global `PredicateStats`,
  so it may not run on a campaign job thread while another holds that lock — and holding the lock
  around `run()` would **deadlock**, because `HeckeSturmSweep`'s streamed route and
  `HeckeExt/HeckeBoundary` re-enter it and `std::mutex` is not recursive. *If this layer ever
  acquires internal concurrency, that constraint is the thing to revisit.*
* **`ExtDriverOptions` maps onto the spec keys**: `ext_groups` onto `with_cone`, `sign` onto
  `with_sign_split` plus which of `sign_record(bool)` the summary reports, `family` onto
  `good_primes` (the literal token `ell` is recorded and never resolved — U_ell's admission is the
  `u_ell` budget's, [OD] R1).
* **The record is written twice from one serialisation**: standalone to
  `<output>/ext/hecke_ext_ell<ell>_n<n>.json` through `WriteHeckeExtReport` — the only form with a
  *stem*, hence the only one that writes the R3.1 item-6 **state sidecars** — and verbatim inside
  the owner job's JSON under `"extensions"`, beside `"extensions_file"`.
* **The record-level `u_ell` resume** ([X] §1.3.1 R3.3 item 4) reads the persisted record, verifies
  every `state_ref` sidecar's FNV-1a-64, and then calls **this layer's own**
  `resume_u_ell(budget)` after reproducing Stage 1 under the persisted budget. A lower budget is
  refused with the driver's own message, an included verdict is a strict no-op, and a resume to the
  shipped default reproduces a fresh `u_ell = auto` record byte for byte. **It reads
  `record().u_ell.resumable`, the CELL's — never an ideal's**, which is what the `v1 → v2` shape
  repair made possible.
* **`resumable_items()` counts decisions, not entries**, and the campaign's `resumable_items`
  column inherits that: at `(10, 691)` under `u_ell = budget_mb=64` it is `1`.

**Two new validations**, neither of them inside this directory: `HeckeExtCampaignValidation`
(`Campaign/CampaignExtValidation.cpp`, `standard`, ~2 s, 131 checks) and
`HeckeExtDeterminismValidation` (`src/hacs/HeckeExtDeterminismValidation.cpp`,
`standard`, ~12 s) — the latter lives beside the layer and not in it because installing an engine
thread pool needs `src/Engine/`, which rule 2 forbids here. It hashes the **whole record
text**, the unsplit one and both per-sign ones, on `(10, 691)`, `(34, 5)` and `(24, 7)` under every
`ENGINE_THREADS × ENGINE_BATCH` arm, plus `(128, 5)` at threads `{1, 16}` and two `device = gpu`
arms: digest **`0xfb71e225e4185be7`** in all ten — **bare** since slot-6 WP-6 redacted the record's
`commit` key before hashing (at the slot-5 close it was `0x415f73f051702a71@d362c71`, quoted with its
SHA because it moved on every re-configure).

**No golden entry, no golden value.** The slot's one emission is WP-3's.

## The slot-7 close (hecke-extensions slot 7, WP-14)

**The record is `hecke_ext/v3`** since slot-7 stage 3 (`kExtSchema = kExtSchemaV3`), the default
record having gained nine keys in stage-2 items 3–7: `anemic_pool` and the cell-local
`sturm.proof_range` (item 3), `s_sep` (item 4), per-ideal `label` (item 5), per-ideal `split`
(item 6), `predictions_mode` / `predictions` and per-ideal `predicted` / `confirmed` /
`unpredicted` (item 7); plus the per-ideal `presentation` object under `presentation = on`
(stage 1 (2)). `verify()` codes 24–29. Specified first in [E] v1.8.0 `sec:schema`, "The stage-2
keys" (a)–(h); the architecture prose is `docs/architecture-detail.md`. `HeckeInvolution`'s
consumer tag `"hecke_ext/v2"` is deliberately unchanged (it names the family, not the schema key).

**`HeckeExtDeterminismValidation` hashes the record text with the top-level `settings` block
reduced to its fingerprint** (stage 1 (1), owner decision 2, the same reduction as golden), and its
digest was re-pinned once by the slot-7 close: **`0x03b4f07c14555fec`** (printed
`0x3b4f07c14555fec`, no zero padding), heavy `(128, 5)` **`0x498147a50c6c1e30`** — the moves, one
per golden event, are the ledger of `docs/implementation/slot7/WP-14_report.md` §2.

**Golden**: E7's objects were emitted add-only (stage 0, +651, 16 666 → 17 317) and the 62
`ext_json_*` lines re-baselined in seven explained commits; the slot's one value change is golden
cell `(24, 5)` ideal 1 `final true → false` (`sturm_heuristic`, the cell-local `proof_range`).
`docs/matrix-engine/rebaseline-2026-09-23-e7-addonly.md`,
`docs/matrix-engine/rebaseline-2026-09-23-hecke-ext-staged.md`.

## The slot-8 close (hecke-extensions slot 8, WP-14)

**No new unit in this layer, and the record did not move**: `hecke_ext/v3` is unchanged, its 62
`ext_json_*` golden lines and `HeckeExtDeterminismValidation`'s digest (`0x03b4f07c14555fec`,
heavy `0x498147a50c6c1e30`) are unchanged through slot 8
(`docs/implementation/slot8/golden_ledger.md`). What the layer gained:

* **`coefficient_ring` refusals** (slot-8 WP-3, [E] `sec:ring` (b)): every `HeckeExt/` construction
  site of `EquivariantCohomologyOnK` / `HeckeOperatorOnK` (`HeckeBoundary`, the driver, the Sturm
  sweep, `--atlas`) raises `NotImplementedInThisBuild` naming itself under a non-default ring; the
  record and campaign support over `Z/ell^k` are slot 9's, with C5. The key is registered at its
  default, so no record text moved (key-additivity since slot-7 stage 1).
* **The `--atlas` build's between-cells hook** (slot-8 WP-5): `AtlasOptions::between_cells` /
  `RunAtlasCli`'s `between_cells` callback, which `main.cpp` fills with the engine's program-cache
  release — the layer still never names the engine (`check_heckeext_includes.sh`).
* **`resume_u_ell(…, src)`** (slot-8 WP-11b): the campaign's resume passes `SettingSource::Spec`.
* The ext pass that drives this layer's cells runs `ext_jobs` cells at once since slot-8 WP-6
  (`Campaign/`, not here); every record is byte-identical to the serial pass.

## Slot 9, WP-13 — console phase C5b: the extension-layer bindings

Two new **console-free** units and one validation; the bindings themselves stay in the one
`Console/`-admitted TU, `RegisterHeckeExtensionVerbs.cpp` (its "C5b" section), so no include-guard
allow-list moved. (S24 CL-8 split that TU by area: `RegisterC5` stays in `RegisterHeckeExtensionVerbs.cpp`; the
bindings live in `RegisterHeckeExtensionVerbs{Sturm,Algebra,Records,Values,Complex,Gallery}.cpp`, map in
`RegisterHeckeExtensionVerbsDetail.h`; each file is an exact-path entry of both guards.)

* **`HeckeAlgebraInvariants.{h,cpp}`** — the local-algebra invariants of ANY commuting family
  over `F_ell`, with no locality certificate as input: `rad = ker Frob^e` (`ell^e >= D`), the
  number of maximal ideals by Berlekamp's count (`dim{x : x^ell - x in rad} - dim rad`), the
  powers of the radical, `rad/rad^2`, `soc = Ann(rad)`, the Hilbert function, the Loewy and socle
  series of an `A`-stable subspace, and the Jordan–Chevalley split `s = t^(ell^N)` (`N` a multiple
  of the lcm of the irreducible factor degrees with `ell^N >= D`), re-verified before it is
  returned. It reuses `Hecke/SubspaceOps`' strict-locality primitives (`algebra_closure`,
  `frobenius_matrix`, `frobenius_kernel`, `FrobeniusPower`). It is the route of the console verbs
  `radical`, `tangent_space`, `socle`, `nilpotency_index`, `is_local`, `loewy`, `socle_series`,
  `nilpotent_part`, `algebra_invariants`.
* **`HeckeExtBindingSupport.{h,cpp}`** — the pipeline plumbing of the complex bindings, each a
  composition of public constructors in the order an existing validation already uses: the bulk
  / boundary / `eta` / cone-action stack at one prime (`ConeActionAtPrime`,
  `BoundaryActionAtPrime`), the production involution and its per-degree sign split, the induced
  Dickson map `L_*`, and `AssembleIdentityCertificate` (items 1–5 at every cell; items 6 and 7 on
  request).
* **`HeckeExtAlgebraInvariantsValidation`** (`standard`, registered in
  `Registry/Slot9/WP13.cpp`): (A) the certificate-free route equals `HeckeLocalAlgebra` /
  `HeckeModuleStructure` entrywise on every certified-local ideal of six driver cells (the algebra
  basis, `rad == m_S`, `nu`, Hilbert, socle, embdim, `nilpotent_part(t_p) == n_p`, the Loewy and
  socle series); (B) constructed algebras incl. the **Galois-conjugate trap** (`F[A, B] = F_49 x
  F_49`, each generator alone a field); (C) Jordan–Chevalley on a known split, the refusals;
  (D) the binding-support builders at `(12, 5)` — cone action, `eta`, involution, sign split,
  `L_*`, and the identity certificate with all seven items assembled and passing.

No record, golden object or digest moved: nothing here is reached from the ext driver, the
campaign or the golden harness.

## The slot-9 close (hecke-extensions slot 9, WP-17)

**The record did not move**: `hecke_ext/v3` is unchanged, its 62 `ext_json_*` golden lines and
`HeckeExtDeterminismValidation`'s digest (`0x03b4f07c14555fec`, heavy `0x498147a50c6c1e30`) are
unchanged through slot 9 (`docs/implementation/slot9/golden_ledger.md`); the slot's one golden
event, R1 (+126, E8 item 4b), is `Hecke/`'s. What the layer gained:

* **C5b** (WP-13, above): `HeckeAlgebraInvariants`, `HeckeExtBindingSupport`,
  `HeckeExtAlgebraInvariantsValidation`, and the console's "C5b" section of
  `RegisterHeckeExtensionVerbs.cpp` (scripts 040–047).
* **The ring stays out**: `HeckeExtDriver` and the record are `F_ell`-only; the campaign's ext pass
  refuses a non-default `coefficient_ring` at spec validation (slot-9 WP-8, [E] `sec:ring` (b4)),
  while the job loop honours it (`Campaign/CampaignRing`, `hecke_ring_report/v1`). Every C5b verb
  that runs the pipeline refuses the ring (exit 5).
* **Measured, not moved**: at `(160, 691)` the cone (1.49 s) and the ι split (1.77 s) are 89 % of
  the cell (slot-9 WP-7's profile); no slot-9 lever reaches them — a slot-10 candidate. The parallel
  ext pass gained from WP-7's L2 (the τ lock narrowed in `Hecke/HeckeOperatorFamily`) and L3 (the
  engine's arena first-fit): `sweep_full`'s `ell = 10007` cells at `ext_jobs = 12` 53 s → 16 s,
  every record byte-identical (indicative).
* **`ext_u691`** (the 45 over-budget `ell = 691` cells, optional WP-16) was not run in slot 9.

## The slot-10 close (hecke-extensions slot 10, WP-17)

**The record did not move**: `hecke_ext/v3`, its 62 `ext_json_*` golden lines and
`HeckeExtDeterminismValidation`'s digest (`0x03b4f07c14555fec`, heavy `0x498147a50c6c1e30`) are
unchanged through slot 10, a zero-golden-event slot (`docs/implementation/slot10/golden_ledger.md`).
No `HeckeExt/` unit was added. What reached the layer's cells:

* **The cone / ι profile at `(160, 691)`** (slot-10 WP-7): the sign split did not scale with threads
  (serial `FlatMatrix` elimination in `Hecke/HeckeInvolution`); lever **I-1** runs its checks and
  splits as host tasks in the serial order, 1.80 → 0.79 s; the cone (1.5 s) is now the largest phase,
  70 % of it `ConeCertificate::verify` (lever C-1, slot 11). Lever **L4** (route 2's rows on the pool,
  `Hecke/HeckeOperatorOnK`) speeds `U_ell` cells (`(18, 10007)` `--campaign-cell` ≈ −22 %). Both act on
  single-cell latency — the sequential ext pass (`ext_jobs = 1`, `ext_first`'s default) — and are
  neutral on the parallel pass; 84 / 84 `ext/` records byte-identical (`commit` redacted).
* **T2 `(10, 691)`**: slot 9's miss was `ThreadPool`'s `.bss` placement, not the layer (WP-6,
  `alignas(64)`).
* **The ring stays out**: `HeckeExtDriver` and the record are `F_ell`-only; `ext_u691` and the
  `Z/ell²` survey (W2) were not run in slot 10.


## Slot 12, WP-5 — `HeckeHidaTower`: the first Hida weight towers

`HeckeHidaTower.{h,cpp}` (`hecke_ext::hida`) implements [C] v1.12.0-pre.2 `def:hidatower` on top of
`Hecke/HeckeOrdinaryPart` (WP-4) — `HeckeExt → Hecke` only, no `Console/`, `Engine/`, `Cuda/`. Per cell
`(n, ell)`, `ell ≥ 5`, `n ≥ 2` even: `K = m_max + e_max` from the torsion pre-pass (`cor:ordprecision`(2),
before the run), `H_K = H^1(Γ; Sym^n (Z/ell^K)²)` by the shipped module route
(`EquivariantCohomologyOnKT<uint64_t>` under `coefficient_ring = Z/ell^K`) with `U_ell`, `T_2`, `T_3`
(`H_matrix_module`); then for each target `e'` the two readings of the ordinary part — on
`H_{e'} = H_K/ell^{e'}` (`cor:ordprecision`(1), torsion included) and on the free block `F'`
(`cor:ordprecision`(2)–(3)) — each with `P = U^{N_ord}` (`lem:ordexponent`), its rank over `F_ell` and the
characteristic polynomials of `T_2`, `T_3` on the free module `P M`: a basis of columns of `P` independent
mod `ell` (Nakayama), coordinates by an `ell`-adic digit solve (`P M ∩ ell M = ell P M`; every solve
re-verified by `ENGINE_CHECK`), Berkowitz's division-free algorithm over `Z/ell^{e'}`. Writes nothing and
emits no record: the tower data are `HIDA_TOWER` lines on the validations' stdout (not published, plan
§5 L4). Validations `HeckeHidaTowerValidation` (`standard`) and `HeckeHidaTowerSlowValidation` (`slow`),
`Registry/Slot12/WP5.cpp`; the prediction outcomes are in `docs/implementation/slot12/WP-5_report.md`.


## Slot 12, S12-FIN — the finality certificates (d0) "dickson" / (d1) "boundary"; `hecke_ext/v4`

`HeckeFinalityRoutes.{h,cpp}` (new; [E] v1.13.0 `cert:dickson`, `cert:boundary`; design
`docs/implementation/slot12/WP-12_finality_proposal.md` v2, owner-approved 2026-09-26). **(d0)**
`DicksonRoute`: Proposition D on an `X_1` block's stored `t_p` ([C] `prop:heckeH0`(1): every `T_q|_V`,
`q ∈ S \ {ell}`, a scalar, and exactly one class `a mod (ell − 1)` of a Dickson monomial of degree `n` with
`(1 + q) q^a = λ_q`). **(d1)** `BoundaryRoute` + `BoundaryFinalityCertificate`: Proposition F′ on an `X_4` block
(`rank(r^1 V) = dim V`, then `T_p|_V ∈ A_S` for `R(ell)`, the least prime of every class of `(Z/ell)^×`;
[C] `prop:bdrytranscription`, `lem:bdrytorus`(2), `thm:conewelldef`(a)); `verify()` re-derives both, membership
by the independent `HeckeAlgebraInvariants` closure. `HeckeExtDriver` runs them in Stage 4 **only where
(a)–(c) did not pass**, on the unsplit record, at `ell ≥ 5`, `n` even (`ell ∉ S` → `not_applicable:u_ell`);
(d1) builds the boundary pair alone (`ensure_pair_()`, shared with `with_cone`'s cone) and takes the
representatives from the Stage-1 family or `pool_operator_` (streamed beyond 31, from `ell = 11` on);
`boundary_certificate(i)` exposes the certificate object. **The record is `hecke_ext/v4`**
(`kExtSchema = kExtSchemaV4`): `certificate_routes.d0` / `.d1`, `boundary_reps`, certificate names
`dickson` / `boundary`, `FinalInputs::dickson_passed` / `boundary_passed` (`is_final`: a passed route is
`true` in every mode), `verify()` codes 30 / 31; `kCertificateNames` gains both names (a failed route falls
through to the existing reason). Golden: the 62 `ext_json_*` lines moved (text only; under the harness's
`u_ell = off` no route passes) — `docs/matrix-engine/rebaseline-2026-09-26-hecke-ext-v4-finality.md`;
`HeckeExtDeterminismValidation` `0xbd8508b97a1df740`, heavy `0x774b967f88fd5ad7` (`(128, 5)` ideals 2, 4
now `final = true`, `boundary`). In production every `sturm_heuristic` census row at `ell ≤ 13` is
discharged (`docs/implementation/slot12/FIN_report.md`); census edition 2
(`docs/matrix-engine/results/congruence_census/edition2/`, owner-signed 2026-09-26) carries these v4 finality
columns — 3285 / 3285 rows final, 36 `dickson` + 340 `boundary` (S12-census2). Cost: `finality_s` in `ExtDriverTimings`
(not printed on the `BENCH` line).

## Slot 13 — `hecke_ext/v5`, Hida part 2, C-F9, the heavy cells, the `mixed` case (WP-5 … WP-12)

* **`hecke_ext/v5`** (WP-7, owner Q3, FIN_audit G2; [E] v1.14.0-pre.1): `IdealRecord::boundary_data = {present,
  rank_r1, r1, t}`, present **iff** `certificate_routes.d1 == "passed"` — `r1 = r^1 V` (rows in the stored basis of
  `H^1(∂)`), `t` the `t_p|_V`, `p ∈ R(ell)`, index-aligned with `boundary_reps`, in the block basis of the stored `t_p`.
  `HeckeFinalityRoutes::BoundaryDataRederive` (clauses 1–7); the writer emits the object after `boundary_reps`;
  **`verify()` code 32** refuses presence off a passed (d1), absence on one, and any non-zero clause. `final` does
  not read it. Golden R2 (`6cc867e`: the 62 `ext_json_*` lines, schema only); ExtDet `0xaef77d786b4f29b4`, heavy
  `0x9ed8b83a2fb26622`. The ext record sets re-emitted under v5 and published as siblings (WP-8, `9526242`);
  `congruence_census.py --edition 2 --native-records`: `EDITION2_NATIVE rows=3285 identical=3285`.
* **`HeckeHidaTower` part 2** (WP-5; [C] `def:hidalattice`, (Hi7)–(Hi9)): `RunModuleCell(…, rep)` on `Sym^n` or
  Hida's `L_n`; `FreeImage` / `MatrixOnImage`; `ModuleDatum` / `ReadModuleDatum`; **`DecideIsomorphic`** — (Hi8)'s
  decision procedure (`Hom` by `kernel_module`, pre-checks, `det` on the grid `{0..h}^{dim Hom-bar}`, a verified
  certificate or "not isomorphic"; never "inconclusive"); `LatticePair` / `ReadDstar` (`D_*`).
  `HeckeHidaControlValidation` (`standard`, 1134). Golden R1 (`9d58b87`): 21 entries at nine `HIDA` points.
* **`HeckeCensusLattice`** (WP-9; [E] `pr:latticesplit`, `\CONJ`): per census cell the `F_ell` side (the driver,
  blocks matched to their record ideals) and the lattice mod `ell²` by the module route with the Fitting projector
  `e_m`; C-F9 ⇔ `X_q X_{q'} e_m ≡ 0 mod ell`. **Held on 654 / 654** non-reduced edition-2 rows; the Serre-weight
  mechanism not confirmed. `HeckeCensusLatticeValidation` (`standard`, 69; `HECKE_CENSUS_LATTICE_ALL` / `_CELLS` /
  `_TSV`).
* **`HeckeHidaHeavyCellProbe`** (WP-10; `slow`, interlocked `HECKE_HIDA_HEAVY=1`, cells `HECKE_HIDA_HEAVY_CELLS`):
  heavy cells under the `U_ell` lever; `(700, 691)`: (Hi11) held over `F_691` (13.65 GiB, 61 s) and `Z/691²`
  (33.45 GiB, 47 min). `(2420, 2411)` deferred to the cluster roadmap.
* **The `mixed` finality case** (WP-11; `docs/implementation/slot13/WP-11_mixed_finality_proposal.md` v2, refereed):
  Proposition M, (d2) "Dickson-mixed" — **specified only** (owner Q10), no driver or record change.
  `HeckeMixedFinalityValidation` (`slow`, scan hook `HECKE_MIXED_FINALITY_CELLS`) and `HeckeBocksteinCuspValidation`
  (`fast`, 1447: Claim (H4′) cell by cell).
* **Carries** (WP-12): `RegisterHeckeExtensionVerbs.cpp`'s `dickson_les` and `ext_record` refuse a non-prime `ell`
  (`TypeError`; A14 at two of three sites).

## Slot 14 — `HeckeRingHeckeAlgebra`, `HeckeRingLoewyValidation` (WP-6; close: WP-17)

* **`HeckeRingHeckeAlgebra.{h,cpp}`** (`hecke_ext::ringalg`, console-free; [E] v1.15.1-pre.2 `sec:ring`, paragraph "The Hecke algebra over `Z/ell^K`, read
  mod `ell`"): `H^1(Gamma; Sym^n Z/ell^K)` and `T_p` by the shipped module route (`uint32_t` while `ell^K < 2^32`, `uint64_t` above; `ell^K >= 2^64` refused by
  name, the digit route is slot 15), the `F_ell`-rational joint systems, the localisation `e_m`, the algebra `Z_ell[T_p]` read by `engine::elim::howell` /
  `length` only, and `ComputeRingLoewyCell` with its verdicts. The `stable` verdicts are **readings, not certificates** (audit Q-C). No record key, golden
  object or setting.
* **`HeckeRingLoewyValidation.{h,cpp}`** (`standard`, 123 checks, `Registry/Slot14/WP6.cpp`): arms (P), (a)–(e), (W); env `HECKE_RING_LOEWY_CELLS`; lines
  `LOEWY_RING*`; C-F9 failures reported as `LOEWY_RING_CF9_FINDING`, never gated. Cluster shard / collate driver:
  `docs/implementation/slot14/wpc14/loewy_collate.sh`.
* Slot 14 added no other `HeckeExt/` unit and no record change (`hecke_ext/v5` stands); the layer's dependency guard now allow-lists `Registry/Slot14/*.cpp`.

## Slot 15 — `HeckeHidaComponents`; the certified `stable`; the forms side (WP-5, WP-6, WP-20; docs: WP-17)

* **`HeckeHidaComponents.{h,cpp}`** (new, WP-6; `hecke_ext::hidacomp`, console-free): the bignum-free `B_k/k mod ell^K` (direct and
  Kummer–Mahler routes, `ell >= 5`, `ell^K < 2^64`) and the readings of [U] (U1)–(U3) at tame level 1 on the `Z/ell^K` module route;
  frozen oracle `HeckeHidaComponentsOracle.inc` (611 values). `HeckeHidaComponentsValidation` (`standard`, 157 checks,
  `Registry/Slot15/WP6.cpp`), env `HECKE_HIDA_COMPONENTS_CELLS="n:ell[:K],…"`. Citation: Washington 1st ed., p. 122 (Thm 7.1, Prop 7.6) and Thm 7.10 (verified, slot-16 WP-8).
* **`HeckeRingHeckeAlgebra`** (WP-5): the `stable` verdict is a certificate citing [C] `cor:stablecert` (`CertifyStable`), and
  `ell^K >= 2^64` runs on the digit route; `HeckeRingLoewyValidation` 123 → 181 checks (arms (C1)–(C3)); [E] 1.15.1-pre.6.
* **`HeckeFormsHeckeAlgebraValidation`** (WP-20): F9 gated torsion-corrected (`+ h`, `DicksonH0Lambda`), `M_k` faithfulness per
  block, the census columns `saturated` / `mult_one`; 163 → 172 checks (`HeckeBoundaryFinalityValidation` 13 441).
* No record, schema, golden or settings change in slot 15.

## Slot 19 — the L-value interface (WP-LV)

* **`LValue/LValueRoute.{h,cpp}`** (new; `hecke_ext::lvalue`, console-free): [E] `def:lvroute` — `LValueRoute` with `kind` ∈ {linear,
  bilinear, congruence}, `Domain` (the model is a parameter, `ModelId`), `NormalisationRecord`, one evaluation per kind (the others refuse),
  `IndexStatement`; the declared stubs `BirchModularSymbolRoute` (linear) and `HidaPairingRoute` (bilinear) throw `NotImplementedInThisBuild`.
* **`LValue/EisensteinBernoulliRoute.{h,cpp}`**: the instance (E), kind congruence ([E] `def:lveis`, `cl:lveis`): iota^* on `H^1(Z/ℓ^K)`,
  the `m_Eis`-block of `H^-`, the saturation verdict, `Q_m = A_m/(T_p − e_p)`, the inequality over `p ≠ ℓ`, the module form on `H^+`.
  Torsion in `H^1` and the character model (`cl:lvtwist`) are refused by name.
* **`ZellK.h`** (S24 CL-G1, header-only, `hecke_ext::zk`): the ONE `u64` `Z/ℓ^K` kit of the LValue / Hida modules — `Ring`
  (= `birch::Ring` = `hidapair::Zq`), `M22` (= `birch::Mat2`), `FreeMat` (= `latmod::FreeMat` = `lvalue::FreeMat`) with its
  arithmetic, row-list `MatMul` / `MatPow` / `MatVec`, Berkowitz `Charpoly`, `FieldRank` mod `ℓ`. Kept separate by design: the
  `u128` `ZMod` and `HeckeHidaLambdaForms` kit, `HeckeLambdaTower`'s `u32` lazy kit (measured faster), and the oracles
  (`RankModEllJoint`, the Voronoi Bernoulli route, `RhoDirectMatrix`).
* **`LValue/Bernoulli.{h,cpp}`**: exact `B_k/k` (tangent numbers over NumberField's `nf::BigInt`) and Voronoi's congruence mod `ℓ^m`.
* `HeckeLValueValidation` (`standard`, `Registry/Slot19/WPLV.cpp`). No record, schema, golden or settings change.

## Slot 19 — the lattice comparison (WP-LC)

* **`HeckeLatticeModule.{h,cpp}`** (new; `hecke_ext::latmod`, console-free; [E] `sec:block258`, `chk:latcmp`).
  * `RunLatticeModule(n, ℓ, K, primes)` builds `H^1(Γ; Sym^n Z/ℓ^K)` on the shipped `Z/ℓ^K` module route, with the `T_p` and with ι^*
    (`def:iota`(iv); `ι² = 1` and `[ι, T_p] = 0` are ENGINE_CHECKed). Torsion summands are allowed.
  * `LocalizeBlock` takes the joint idempotent (Newton-lifted) and forms the block and its ι-sign parts by `cyclic_decomposition` of the embedded
    images, with the operators restricted and every column re-verified. `FreeMat` arithmetic and `SmithValuations` (by `invariant_factors(I, X)`)
    complete the module, together with `RankModEllJoint` and `ReduceBlock` (`H_M = H_K/ℓ^M`).
* `HeckeLatticeModuleValidation` (`slow`, `Registry/Slot19/WPLC.cpp`), with an env arm `HECKE_LATTICE_MODULE_FULL=1`. No record, schema,
  golden or settings change.
