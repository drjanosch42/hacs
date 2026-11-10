# Hecke/

> **Registry paths (S24 CL-10, 2026-11-01).** Registry fragments `Registry/Slot<S>/<pkg>.cpp` and `cmake/slot<S>/<pkg>.cmake` named in this file for
> slots S <= 23 are historical: they are folded into `Registry/Slot<S>.cpp` (the package function `RegisterSlot<S><pkg>()` is kept there, called by
> `RegisterSlot<S>()` in the old order) and `cmake/sources/<Module>.cmake`; the old READMEs are in
> `docs/implementation/registry-slot-readmes-9-25.md`, the convention in force in `src/hacs/Registry/README.md`.

Implementation of the integral Hecke action on `H^q(K,R;L)` for `GL(2)/SL(2,Z)`,
per the approved staged plan
`docs/implementation/hecke_action_implementation_plan.md` and its binding
contract `docs/latex/hecke_action.tex` (**[C]**; labels below refer to it).
Stages S0-S14 (plus FIX-negI) have landed and are independently validated;
this file is the finalized map of the shipped module (Stage S14). The
feature is complete and acceptance-gated for the primary regime `ell < 2^32`
(Stage S12's `HeckeEndToEndValidation`). Stage S13 (the `ell < 2^64` backend
extension) has also landed and is validated by
`HeckeExtendedRegimeValidation`; nothing else under `Hecke/` depends on it.

## Build layout (Stage S1 design choice)

The Hecke core sources (`ExactInt`, `Predicates`, `QuadPoint`, and future
non-validation Hecke sources) are compiled into a dedicated **static library
target `HeckeCore`** (see the top-level `CMakeLists.txt`), linked into
`cohomology_app` and available to future Hecke stages/tools. It is pure C++
(no CUDA) and does not depend on `ModArithmetic`. Validation translation
units (e.g. `HeckeExactIntSmokeValidation.cpp`) are compiled directly into
`cohomology_app`, alongside their registration in `ValidationRegistry.cpp`.
The `HECKE_PURE_INTEGER` compile definition is applied to `HeckeCore` as
`PUBLIC`, so it propagates to everything that links `HeckeCore` (including
the validation sources in `cohomology_app`).

**Stage S3/S4 placement:** `HeckeCosetTables.cpp`, `HeckeTauTable.cpp` follow
the same rule as S2 below: they consume `ModArithmetic` (`FlatMatrix`,
`ComputeSymmetricPowerMatrixFM` for the `rho_n(.)` blocks) and the model
complex, so they are compiled into `cohomology_app`, not into `HeckeCore`,
and link `HeckeCore` for the exact-integer layer. `HeckeTauTable`'s
constructor runs the MANDATORY exact certification of [C, obl:tau] and
throws on any failure (invariant I8: certification over geometry — the
table cannot exist uncertified).

**Stage S2 placement:** `SL2ZTessellationTables.cpp` is compiled into
`cohomology_app`, *not* into `HeckeCore`, because it consumes the
`SimplicialComplex` model and the relation/generator seed table
`SL2ZRelationSeeds_7VertexModel()` of `LocalSystemSL2Z.cpp` (the single
source of truth for `(relation, g_r)` pairs, invariant I3), both of which
depend on `ModArithmetic` — `HeckeCore` deliberately does not. It links
`HeckeCore` for the exact-integer layer (`Mat22`, `QuadPoint`).

## Files → contract sections

| File (this directory) | Stage | Contract section(s) |
|---|---|---|
| `ExactInt.h/.cpp` (present) | S1; **the tier switch since slot-3 WP-1** | [C, def:primitives], [C, lem:bounds], [C, thm:backend], [C, obl:backend]. WP-1 gives `thm:backend`'s third (`__int128`) tier its one real consumer and makes the escalation observable: `ExactTier` (`Auto` / `ForceWide` / `ForceNarrow`, `thread_local`), `ScopedExactTier`, `exact_tier_escalations()`, and `escalate_on_overflow(narrow, wide)` -- **narrow first, widened only after a `HeckeOverflow`**, so bit-identity wherever the narrow tier completes is a property of the control flow, not of a test |
| `Predicates.h/.cpp` (present) | S1 | [C, sec:fpfilter], [C, thm:fp64], [C, cor:fptotal], [C, cor:fp32], [C, obl:fp] |
| `QuadPoint.h/.cpp` (present) | S1; **two tiers since slot-3 WP-1** | [C, alg:tau] step 1 (quadratic-point geometry); [C, thm:backend]'s escalation ladder. `QuadPointT<I>` is templated on its coordinate type: `QuadPoint = QuadPointT<int64_t>` is the pre-WP-1 class byte for byte (same operations, same FP64-filtered `sign_dot2` predicates), `QuadPointW = QuadPointT<__int128>` is the escalation tier, pure exact integer. Step 1's point coordinates reach `~145 N^2` on the shipped model and `moebius` squares them, so the `int64` tier dies at `N >= ~4580` -- the defect of `docs/KNOWN_ISSUES.md` Sec. 3 |
| `HeckeExactIntSmokeValidation.h/.cpp` (present) | S1 | S1 Definition of Done (smoke: `Mat22` identities, overflow abort, FP-vs-exact agreement, `QuadPoint` spot checks) |
| `HeckeFpFilterValidation.cpp` | S1V | [C, sec:fpfilter] (oracle O13) |
| `SL2ZTessellationTables.h/.cpp` (present) | S2 (+ WP-4a) | [C, def:primitives] (P2),(P4); [C, sec:model]; [C, alg:tau] step 2. **WP-4a (additive):** the boundary subcomplex of `gl2_contract.tex` `sec:boundarydatum` (B1)-(B3) -- `boundary_simplices(q)`, the horocycle line coordinates (`boundary_line_position/vertex/edge/nearest`), `CuspKey` (P^1(Q) labels: `of`, `apply`, `reduce_to_infinity`) -- certified at construction by `validate_boundary()` |
| `HeckeTessellationSmokeValidation.h/.cpp` (present) | S2 | S2 Definition of Done (smoke: closure fixed point, canonical-key idempotence/rewriting invariance, (P2) word round trips + intermediate-entry bound, legacy relation bit-identity + I3 derivation) |
| `HeckeTessellationValidation.cpp` | S2V | [C, sec:model] |
| `HeckeCosetTables.h/.cpp` | S3 (+ WP-4a) | [C, prop:cosets], [C, prop:blockmatrices], [C, sec:example]. **WP-4a:** `twisted_conjugate(gamma, j)` -- the ONE (T1) split-and-conjugate step (`gamma = h s_j`, `g^{-1} h g` exact integral) shared by the bulk `HeckeTauTable::eval_tau_on_translate` (value-preserving refactor, golden-gated) and the boundary datum; `p_hecke()` / `r()` accessors |
| `HeckeCosetTablesSelfCheckValidation.h/.cpp` (present) | S3 | S3 Definition of Done (`self_check()` battery over `p_hecke ∈ {2,3,5,7,11}` × `r ∈ {1,2}` × `ell ∈ {5,691}` × `n_sym ∈ {0,2,10}`); distinct from the independent S3V validation below |
| `HeckeCosetTablesValidation.cpp` | S3V | [C, sec:example] (oracle O16) |
| `HeckeTauTable.h/.cpp` (present) | S4; **step 1 on two tiers since slot-3 WP-1** | [C, def:tau], [C, alg:tau], [C, lem:tauexists], [C, lem:bounds], [C, obl:tau], [C, obl:backend], [C, eq:xiexplicit]. `reduce_to_fundamental_domain`, `choose_vertex` and `compare_vertex_distances` are templated on the `QuadPointT` tier; `construct_vertex` runs step 1 through `hecke::escalate_on_overflow`, writing only its two out-parameters so the retry can observe no partial write. Nothing **stored** widens (the outputs are `gamma_red`, `||.|| ~ N`, and a vertex index), so `certify()`, `bound_B_` and every downstream object are unchanged. The narrow limit is `certify()` clause (e) itself -- `B = max(B_tab, 4 N^2) <= 2^31 - 1`, i.e. `ell <= 23170`. **Since slot-24 ELL-W** a table past it is certified in the **wide tier** (`wide_tier()`; clause (e) `2 N B <= 2^63 - 1`, i.e. `N <= 2^20 - 1` when `B = 4 N^2`; code 52 if that fails): storage stays `Mat22` (`B < 2^62`), the stored-class products / determinants go through `hecke::mul_tiered` / `det_tiered` (int64 first, `__int128` after a `HeckeOverflow`); under `HECKE_TAU_WIDE=0` or a `ForceNarrow` build it is refused with code 51 as before |
| `HeckeTauTableSelfCheckValidation.h/.cpp` (present) | S4 | S4 Definition of Done (constructor-certified tau across the DoD grid incl. the bad cases p = ell; 69-chain count for p = 2; int64-tier bound; FixedV6 policy hook; writes `docs/implementation/tau_table_p2.txt`) |
| `HeckeTauTableValidation.cpp` | S4V | [C, obl:tau] (independent checker-only re-implementation) |
| `HeckeIotaTauSymmetryValidation.h/.cpp` (present) | E3 / C-A1 | `gl2_contract.tex` chapter 9 (`sec:involution`): `obl:iota`(i) GATING (iota_K re-derived from the shipped exact coordinates, `lem:iotamodel`'s sign table entry for entry) and `obl:iota`(iv) -- RECORDED per that clause and, since the 2026-09-16 re-baseline of the canonical tau (iota_K-equivariant tie-break + tree-routed path templates, `docs/matrix-engine/rebaseline-2026-09-16-iota-symmetric-tau.md`), also GATING: `iota_X tau iota_X == tau` now holds on every table of the grid, so a single differing slot on the canonical tau is a failure. The legacy (pre-re-baseline) policy is run alongside and its negative verdict stays recorded-only, `rmk:iotatausym` |
| `HeckeIotaSignValidation.h/.cpp` (present) | E3 / C-A1 | `gl2_contract.tex` chapter 9 (v1.2.0): `obl:iota`(ii), (iii), (v). Builds iota^* as a cochain map from `def:iota` alone -- since v1.2.0 `iota^* := -iota^*_geo`, the geometric pullback `(iota^*_geo f)(sigma) = rho_n(iota) f(iota_K sigma)` negated (owner decision OQ3 of 2026-09-16, `docs/implementation/oq3_resolution.md`) -- and GATES: local-system automorphism entrywise on all 31 relations, `iota_C == -iota_C_geo`, `iota_C^2 = I`, `iota_C d = d iota_C`, `(iota^*)^2 = I` on `H^q_eq`, `[iota^*, A^{(1)}(T_p)] = 0`, the degree-0 anchor `iota^* = -1` on `H^0(Gamma; F_ell)` (geometric `+1`; this pins the global sign), the agreement of two independent Eisenstein routes (`T_p`-eigenline of `1+p^{n+1}` vs restriction to the cusp path with its own iota), and **the definition-consistency oracle `Eisenstein in H^1_{eq,+}` on every cell where a route is defined** (a `-` anywhere would mean the geometric sign is not uniform across cells, i.e. `def:iota`(iv) ill-posed). RECORDS the geometric sign (`minus`, the v1.1.1 finding of `docs/implementation/oq3_iota_sign_verification.md`, now a statement about `iota^*_geo`) and `dim H^1_+/-` against `(d_{n+2}, d_{n+2}-1)` (66/66 off-Dickson cells match; the Eichler-Shimura mod-`ell` prediction, `CONJ`). Verdict: **`IOTA_SIGN eisenstein=plus (geometric=minus) cells=77 signed=74 ... h0_anchor=-1 (geometric=+1) ... gating_failures=0`** |
| `HeckeBoundaryDatum.h/.cpp` (present) | E4a / WP-4a | `gl2_contract.tex` chapter `sec:boundarydatum`: `def:bdrygroup`, `prop:bdrycosets`(1)-(3) (the `B_N/Gamma_inf` left-coset table `boundary_cosets()`, the bijection `boundary_coset_of_bulk(j)` / `gamma_j(j)` with the bulk cosets), `def:taubdry` (T1bd)-(T3bd), `lem:bdrycomponent` (`component_map`, recomputed from `g` alone), `lem:taubdryexists`(1)-(2) (the construction: nearest line vertex, ties to the larger real part; edges as the unique line path; every `Gamma''`-related slot derived through the single public helper `tau_on_translate`, which routes through `HeckeCosetTables::twisted_conjugate`), `obl:taubdry`(i)-(iv) certified in the constructor (`certify()`, codes documented in the header; `certify_with_override` is the validation seam). `HeckeBoundaryTauTableT<Elt>` is ell-independent and built for every ell; `solve_regime() == "Z"`, `integral() == true` unconditionally. Degree-generic store (`num_degrees() = boundary_dim + 1`). `Theta^bd`, routes A/B, the `chi_b` oracles, the `beta_j` control and `r^q` are WP-4b |
| `HeckeBoundaryTauValidation.h/.cpp` (present) | E4a / WP-4a | `obl:taubdry`(i)-(iv), `prop:bdrycosets`(1)-(3), (B1)-(B3), re-derived independently (own cusp arithmetic, own boundary operator, own (T1bd) translation) on 138 tables; tamper rejection; ell/n/width independence; prints the p = 2 table and `BOUNDARY_TAU tables=138 checks=52686 failures=0` |
| `HeckeBoundaryOperator.h/.cpp` (present) | E4b / WP-4b | `gl2_contract.tex` chapter `sec:boundarydatum`: `def:bdryop` / `eq:bdrychain` (the NORMATIVE `Theta^bd_g` on the circle model `C^*_eq(K_bd, Rel_bd; L)` -- `LocalSystemSL2Z::CreateSL2ZBoundaryPathTriangulation` / `GenerateSL2ZBoundaryRelationsElt` -- assembled from the SAME `s_j`, `delta_j` and the WP-4a `tau_bd` in two bracketings asserted `memcmp`-identical, `A_bd^{(q)} = P_q Theta^bd z_i` per `alg:assembly`), `prop:bdryop`(1) (relation closure and `d Theta = Theta d` asserted in `compute()`), `obl:taubdry`(v) route A (`Theta_shapiro_flat` / `A_shapiro`: a Gamma_inf-level twisted datum on the infinity-line -- `tau_inf: T -> T^p`, `tau_0: T^p -> T`, nearest-vertex rule -- with the canonical left cosets `delta_inf, delta_t`, no `tau_bd`, no class map; asserted `== A` on cohomology at `nu = 1`) and route B (`A_groupform`: `prop:bdryop`(2) `sum_delta rho_n(delta)` on `V^T` at every `nu`, `eq:bdrygroupform` on `V_T` at `nu = 1`, transported through `class_representative` / `lift_class` / `P_q`; asserted `== A`), `prop:bdrychar`(1)-(3) support (`invariant_monomial_basis`, `chi`), `prop:bdrycosets`(4) / `prop:bdrychar`(2) as the exposed NEGATIVE control `beta_control_fiber_operator`; the model recording of owner confirmation 11 (`boundary_model = "circle"`, `boundary_is_group_cohomology = (ell != 2)`, `q_max = 1` derived, `truncated = false`) and a JSON record `hecke_boundary_operator/v1` with the certificate bundle |
| `HeckeBoundaryOperatorValidation.h/.cpp` (present) | E4b / WP-4b | `obl:taubdry`(v)-(vii), `prop:bdrychar`(1)-(3), `prop:bdrycosets`(4), `prop:circlemodel`(2)-(3), `lem:ell2unbounded` (negatively), `cl:chi`(2) recorded, `sec:oraclesnew`(A): route A / route B / an own route B `== A` entrywise; the `chi_b(p)` eigenvalues on an own `x^{n-b ell} Nm^b` basis and `1 + p^{n+1}` on `[y^n]` asserted; the beta control gives `chi_{b+1}` and breaks (vii); `r^q A^{(q)} = A_bd^{(q)} r^q` against `HeckeOperatorOnK` through `EquivariantCochainComplexMorphismOnK`, asserted at `ell >= 5`, recorded at `ell in {2, 3}`; 546 cells, prints `BOUNDARY_OP cells=546 checks=19669 failures=0` |
| `HeckeBoundaryDiagnostics.h/.cpp` (present) | E4b / slot-2 WP-7 | `gl2_contract.tex` `rmk:conestrict` and [X] §0.4 C-A3(iii)(5): the two RECORDED diagnostics, never branched on -- `cusp_support_walk(tau, tau_bd)` (**`eta_zero`**: every stored boundary slot of the bulk `HeckeTauTable::tau(j, q, sigma)` compared, in canonical `KuSimplexKey` form over `Z`, with `HeckeBoundaryTauTable::tau(j, q, sigma)`, plus whether the bulk chain stays on the boundary subcomplex at all) and `cochain_commutes(coh_K, op_K, op_bd, res)` (**`cochain_commutes`**: `r^q Theta_g = Theta_g^bd r^q` on the stored `C^q_eq(K)` basis through `apply_Theta` and the restriction morphism). Lives here and not in `HeckeExt/` because it consumes `HeckeOperatorOnK` (rule 2 of the ext guard) and so that WP-8's `HeckePairHomotopy` can compare its own `eta == 0` with the walk without crossing the guard. Consumed one-way by `HeckeExt/HeckeBoundary`; validated by `HeckeBoundaryParabolicValidation` (`../HeckeExt/`) |
| `HeckeIntegerSolve.h/.cpp` (present) | S4 / slot-2 WP-8 | [C, alg:tau] step 3's hand-rolled integer solver `solve_integer_system` (column-echelon with a tracked unimodular transform, forced forward substitution with divisibility checks, exact self-verification; overflow-checked, throws `HeckeOverflow` as a tier failure). Moved VERBATIM out of `HeckeTauTable.cpp` by WP-8 so that the eta fills use the ONE solver the bulk tau uses; value-preserving (golden tier A/B unchanged, `tau_table_p2.txt` byte-identical). The caller decides what an overflow means: `HeckeTauTable` falls through to its permitted mod-ell solve, `HeckePairHomotopy` aborts (OP2) |
| `HeckePairHomotopy.h/.cpp` (present) | E5-1 / slot-2 WP-8 | `gl2_contract.tex` chapter `sec:cone`: `def:pairhomotopy` (the pair homotopy `eta : C_*(X_bd) -> C_{*+1}(X)` with (T1) and `boundary eta + eta boundary = j tau_bd - tau j`, sign convention OP6), `lem:etaexists` (the construction over Z for EVERY ell: (V) a vertex slot is the [C, alg:tau] step-2 edge path from `tau(s_j v)` to `tau_bd(s_j v)` -- (P2) word + the certified path templates, family selected by `EtaVertexPolicy` (canonical tree-routed by default, legacy BFS as the second admissible policy); (E) an edge slot is the UNIQUE 2-chain with boundary `(j tau_bd - tau j)(s_j e) - eta(boundary s_j e)`, found by the finite-neighbourhood integer solve of step 3 through `HeckeIntegerSolve`; every Gamma''-related slot derived through the single public helper `eta_on_translate`, which routes through `HeckeCosetTables::twisted_conjugate` like tau and tau_bd), `lem:etaunique` (`unique_fill(q, z)` is what its `L` is computed with), `obl:eta`(i)-(iv) certified in the constructor (`certify()`, codes in the header; `certify_with_override` is the validation seam), `open:ell2` rider (i) (degrees built = every boundary degree `q` with `q + 1 <= dim X`, i.e. `[0, 1]`, `q_max = 1` derived, `truncated = false`). `HeckePairHomotopyT<Elt>` takes no `ell` and no `n`: built for every ell, `solve_regime() == "Z"`, `integral() == true` unconditionally -- there is NO mod-m path (OP2; a tier failure aborts). Records `eta_zero` (`obl:eta`(vi), never branched on), `defect(j, q, sigma)` = `(j tau_bd - tau j)(s_j sigma)`, `table_fnv64()`, `to_json()` (`hecke_pair_homotopy/v1`). Clause (v) and the derived `h = eta~` are the cone's (slot 3, `HeckeExt/HeckeConeAction`) |
| `HeckePairHomotopyValidation.h/.cpp` (present) | E5-1 / slot-2 WP-8 | `obl:eta`(i)-(iv), `lem:etaunique`, `open:ell2` rider (i), re-derived independently (own canonical accumulation, own boundary operator, own (T1) translation with the conjugate written out, own transporter parity, own recounts) on 152 tables -- `(p, nu) in {(2,1),(3,1),(5,1),(7,1),(11,1),(2,2),(3,2)} x ell in {2,3,5,691,134217757} x n in {0,2}` at u32 and `(2,1),(3,1),(2,2)` at `2^61-1`, `2^32+15` (u64), both policies -- plus 12 random Gamma-translates per slot; the ell/n/width independence; nine tampered chains per tamper cell rejected; `eta_zero` agreeing slot by slot with WP-7's cusp-support walk (false on all 152 tables); the two policies' tables differing by `boundary L - L boundary` for the unique degree-2 `L`; prints the `p = 2` table and `PAIR_HOMOTOPY tables=152 checks=113596 failures=0 ...` |
| `SL2ZBoundaryPair.h/.cpp` (present) | E5-2 / slot-3 WP-5 | `gl2_contract.tex` chapter `sec:cone`, `def:pair`: the PIPELINE-SIDE pair `(K, K_bd)` of the 7-vertex model with the restriction `r = j^*` -- the circle model (`CreateSL2ZBoundaryPathTriangulation` / `GenerateSL2ZBoundaryRelationsElt`, `prop:circlemodel`(4)'s `-I` self-relations included), its cohomology, `SimplicialComplexMorphism(&K_bd, &K)` and the identity-fibre `EquivariantCochainComplexMorphismOnK` (domain = boundary, codomain = bulk), computed -- lifted VERBATIM out of `HeckeExt/HeckeBoundary::build_boundary_side` (which now consumes it one-way; `HeckeBoundaryParabolicValidation`'s summary lines and collector digest `0xaa7d14c17b488c14` unchanged are the value-preservation proof) so that the cone and WP-6's cone action can build on it without crossing rule 1 of `check_heckeext_includes.sh`; carries the record labels `boundary_model = circle`, `boundary_is_group_cohomology = (ell != 2)`; `make_cone()` builds the computed `EquivariantMappingCone` (`../EquivariantMappingCone.h`, generic: `eq:conediff` in `H_c`-indexing on the flat layout of `alg:assembly`, `Z / B / H / P` by `engine::blk` (D5b), the three map families of `eq:coneles`, the per-complex degree fields of `open:ell2` derived, `ConeCertificate` = `obl:coneexact`(iv)-(v) + `Dcone^2 = 0` with a re-deriving `verify()`) |
| `HeckeConeValidation.h/.cpp` (present) | E5-2 / slot-3 WP-5 | `def:pair`, `eq:conediff`, `lem:conerelative`(2)-(5), `eq:coneles`, `eq:conedims`, `open:ell2` riders (i)-(iii), `obl:coneexact`(iv)-(v) and the `Dcone^2 = 0` half of (i), re-derived independently (an own dense `eq:conediff` assembly from the simplicial coboundaries and the vertex-list restriction == the stored `D_c` entrywise, own `D^2 = 0`, an own zig-zag connecting map == the stored `r^q` == the morphism's induced map == `HeckeExt/HeckeBoundary::r(q)`, own ranks for exactness at every spot with `im pi^1 = ker r^1` explicitly, own `dim S_k`) on 74 cells -- `ell in {2, 3, 5, 7, 691, 134217757, 2^31-1} x even n <= 16`, `(20, 5)`, `(10, 13)` and the ten Kummer cells of [E] Claim 1.8 -- `H^0_c = 0` at EVERY `ell` and `n`, the first `eq:conedims` identity at every `ell`, `H^2_eq = 0` asserted at `ell >= 5` / recorded at `ell in {2, 3}` (0 of 18 cells non-zero), `dim H^1_c - dim coker r^0 = dim H^1_!`, the measured `dim H^1_c = 3` at `(10, 691)` and `1 + 2 dim S_{n+2}` with `H^2_c = 0` at all ten Kummer cells, the record fields (a set `q_max` on the finite model rejected), the truncated branch of (iv) on a declared truncation of the cylinder rel. both boundary circles (E9's shape as a test double: `q_max_c = min(q_max_K, q_max_bd + 1)`, the `H_bd^{q_max}` spot recorded, every other spot exact), a three-way tamper arm, the golden collectors' digest (defined, never emitted) equal in-process and across `ENGINE_THREADS = 1 / 16` children and under `ENGINE_BACKEND=legacy`; prints `CONE cells=74 kummer=10 ... checks=5350 failures=0` (1.9 s). WP-6 extends this TU with the action arms |
| `HeckeConeAction.h/.cpp` (present) | E5-3 / slot-3 WP-6 | `gl2_contract.tex` chapter `sec:cone`: `def:coneaction` / `eq:derivedh` (the derived cochain homotopy `h := eta~`, the COSET TRANSPORT of slot-2 WP-8's `eta`-table -- `(h phi)(y) = sum_j rho_n(delta_j) phi(eta(s_j . y))`, i.e. the block `sum_j sum_t n_t rho_n(delta_j gamma_t)` in two bracketings asserted bit-identical, never chosen), `eq:conemap` (`Phi_g(c, c_bd) = (Theta_g c, Theta^bd_g c_bd + h c)`, block lower-triangular over the cone's flat layout), `lem:conesign` (the residual `(d h + h d + r Theta - Theta^bd r)` computed as a matrix in every degree and asserted identically zero ON THE STORED `C^q_eq` BASIS -- the contract's complex is `Ceq(K, Rel; L)` and `eq:cosettransport` uses `phi`'s `Gamma`-equivariance, so the FULL FLAT identity is RECORDED in `phi_flat_identity` / `residual_flat_zero`, never asserted), `lem:derivedhomotopy`, `thm:conewelldef`(a),(b),(d), `alg:assembly` on `Cone(r)` (`A_c^{(q)} = P_c Phi H_c`). Two certificates with re-deriving `verify()`: `ConeActionCertificate` (items 1-12; items 1-11 = `obl:eta`(v),(vi) + `obl:coneexact`(i)-(iii)). **Since slot-5 WP-4 item 8, `r^q A = A_bd r^q`, is ASSERTED AT EVERY `ell`** -- the `ell in {2,3}` carve-out of `rmk:conerestrictsmallell` is dropped ([OD] 2026-09-22 OI-3, Option A; [C] v1.7.0 writes in the proof that clause (i) implies the `r`-half), value-preservingly. **Item 12 (slot-5 WP-12a, [OD] 2026-09-22 Q2) is RECORD HONESTY** ([C] rule A6): the emitted `restriction_equivariance_asserted` must report the assertion item 8 actually performs, so it now reads `true` at every `ell`, and a certificate that says otherwise fails item 12 alone and `ConeRelationsCertificate` (`obl:coneexact`(vi): commutativity and the `nu = 2` recursion with the FULL `T_{p^2} = T'_{p^2} + p^n I`, `sec:primfull`). `apply_Phi_direct` is the matrix-free second route (`apply_route2` + `apply_direct` + an own `eq:derivedh` evaluation), asserted equal to the stored `Phi` on every standard basis vector; since slot-5 WP-6 it shares three `hecke::RhoMemoT` tables (one per coset table) per degree, which cuts the `n = 256` build 74.5 -> 42.6 s for ~3.2 MB, entrywise identical. `to_json()` = `hecke_cone_action/v1` (the per-complex degree fields, `h`'s SOURCE-complex fields per `sec:recordschema` item 5, the three table hashes, `eta_zero` / `cochain_commutes` / the solve regimes). Golden collectors DEFINED, never emitted. In `Hecke/`, not `HeckeExt/`: it consumes `HeckeOperatorOnK` / `EquivariantMappingCone` (rule 2) and is pipeline code E2 consumes (rule 1) |
| `RhoMemo.h` (present) | slot-5 WP-6 | Not a contract object: the memo type behind the matrix-free evaluation of [C, eq:heckechain], [C, eq:bdrychain] and [C, eq:derivedh]. `hecke::RhoMemoT<Block>` = `std::map<(j, gamma), Block>` + an O(1) `Identity` stamp (`{owner, coset table, ell, n_sym, m}`) that `ENGINE_CHECK`s the part of the value's dependency set outside the key. Consumed by `HeckeOperatorOnK::apply_route2`, `HeckeBoundaryOperator::apply_direct` and `HeckeConeAction::apply_h_direct` through memo-taking overloads; the one-argument overloads keep the pre-slot-5 fresh-table behaviour, so `HeckeThetaBlockValidation`, `HeckeOperatorSmokeValidation` and `HeckeBoundaryOperatorValidation` remain cold-path controls. `assemble_theta` / `assemble_h` keep call-local tables (their `fused` flag keys a different expression) |
| `HeckeConeActionValidation.h/.cpp` (present) | E5-3 / slot-3 WP-6 | `obl:eta`(v),(vi) and `obl:coneexact`(i)-(iii), (vi), (viii)(a),(b), re-derived independently (an OWN coset transport of the `eta`-table with its own `rho_n((delta_j gamma_t) mod ell)` products, an OWN dense assembly of the `lem:conesign` bracket from the simplicial coboundaries and the vertex-list restriction, own kernels, an own rank-one `xi`) on 63 cells / 212 operators -- `ell in {2,3,5,7,691,134217757,2^31-1} x even n <= 16` with `(p, nu) in {(2,1),(3,1)}`, a `(5,1)` / `(2,2)` sub-grid, and `U_ell` at `ell <= 7`. Includes the four-way independence arm `EtaVertexPolicy x TauBoundaryVertexPolicy` with the NON-VACUITY assertion that both table hashes move, the `(viii)(b)` negative control `h' = h + xi` (failing EXACTLY `obl:eta`(v) and changing `A_c^{(1)}` at `(10, 691)`), and a two-way tamper arm; prints `CONE_ACTION cells=63 operators=212 ell23_operators=74 checks=123634 failures=0` (11.9 s) |
| `HeckeInvolution.h/.cpp` (present) | E3 / slot-4 WP-3 | `gl2_contract.tex` chapter `sec:involution`: `def:iota`(i)-(iv) with the **v1.2.0 negation** `iota^* := -iota^*_geo` (owner decision OQ3 of 2026-09-16, `docs/implementation/oq3_resolution.md`; '+' is BY DEFINITION the Eisenstein side), `lem:iotamodel` (the vertex permutation `0<->3, 1<->2, 4<->5, 6` and the six minus signs), `prop:iotacommutes`(1),(2), `prop:iotasplit`(1),(2), `obl:iota`(i),(ii),(iii) ABORTING in `compute()`. The **PRODUCTION** object `sec:codemap` specified and slot 4 built: an `EquivariantCochainComplexMorphismOnK` through the general constructor whose `FiberBlockTable` holds exactly one `PerSimplexBlock` per simplex -- source simplex `|iota_K sigma|`, block `-eps_sigma rho_n(iota)`, the sign of `lem:iotamodel` NEGATED per `def:iota`(iv) and folded into the matrix -- so the object carries no data beyond the single `d_V x d_V` diagonal `rho_n(iota) = diag((-1)^{n-j})`, interned once by the pool. `iota_K` is PROPOSED by an exact `QuadPoint` match of `z -> -conj(z)` against the shipped `vertex_position` and then CERTIFIED against `lem:iotamodel` (invariant I8: geometry proposes, certification disposes); the orientation signs are the parity of the sort back into increasing vertex order, i.e. pure combinatorics. `(iii-a)`/`(iii-b)` are decided BLOCKWISE and exactly -- `iota_C^2 = id` reduces to `eps_{iota sigma} = eps_sigma` plus `rho^2 = I`, and `iota_C d = d iota_C` to `D[iota t][iota s] = eps_t eps_s D[t][s]` over **Z** -- so the cost is `O(N_q^2)` integer work and the verdict is a PROOF of the matrix identity, not a sample of it (the dense matrices are compared entrywise by the validation). Stores the `+-` bases of `C^q_eq`, `Z^q_eq`, `B^q_eq` (flat coordinates) and `H^q_eq` (stored basis) as the D5b image bases of `(id +- iota^*)/2`; at `ell = 2` `sign_split = not_applicable` with `prop:iotasplit`(2)'s `a`, `b` recorded instead. `to_json()` = `hecke_involution/v1`, carrying `iota_sign = eisenstein_in_H1_plus` and the per-complex degree fields. Clause (iv) is NOT re-run here (it is `HeckeIotaTauSymmetryValidation`'s gate on the canonical `tau`, and this object reads no `tau`); clause (v) is `HeckeExt/HeckeSignSplit`'s. In `Hecke/`, not `HeckeExt/`: it consumes `EquivariantCohomologyOnK` and the morphism class (rule 2) and is pipeline code E2 consumes (rule 1) |
| `HeckeWideTauValidation.h/.cpp` (present) | slot-24 ELL-W | [C, lem:bounds](5), [C, thm:backend], [C, obl:backend](ii),(iii) (the wide tier, [C] v1.23.0-pre.9 draft). **`HeckeWideTauValidation`** (`standard`, ~37 s, 52 checks): the tiered helpers (fault injection, `ForceNarrow`, overflowing result, extreme operands), forced-wide = narrow on the `U_ell` tables at `ell in {5, 691, 5003}` and `T_2`, `T_3` at `(691, 10)`, the boundary `ell = 23173` (wide; `ForceNarrow` and `HECKE_TAU_WIDE=0` refuse), `U_1009` / `U_30011` at `n = 10` against `tau(p)` from the q-expansion of Delta. **`HeckeWideTauSlowValidation`** (`slow`): forced-wide = narrow at `ell = 23167`, `T_30011` mod 13 at `n = 10` against Delta |
| `HeckeUEllReachValidation.h/.cpp` (present) | E1b-0 / slot-3 WP-1 | `docs/KNOWN_ISSUES.md` Sec. 3, [C, thm:backend], [C, lem:bounds](3),(5), [C, obl:backend]. **`HeckeUEllReachValidation`** (`standard`, ~11 s, 6160 checks): `QuadPointW` vs `QuadPoint` on 4000 seeded transforms of the shipped vertex positions (identical wherever `int64` completes -- point AND both predicate signs -- and correct past it); the `Gamma_0(N)\Gamma` coset tables self-checked at `ell = 3617` and at `(p, r) = (71, 2)`, `N = 5041` past the old limit (the slot-3 audit's F13 vacuity check: the coset construction was never the site); the **forced-wide arm** at `ell in {5, 691}` and `(71, 2)`, comparing `tau`'s chains, `certified_bound_B()` and `A^{(q)}(U_ell)` **entrywise** between `ExactTier::ForceWide` and the default; escalation accounting (`Auto` widens 0 times on 8 sub-threshold cells, so every golden cell runs the pre-WP-1 path); the **negative control** (`ForceNarrow` at `ell = 5003` still throws the named `HeckeOverflow`); and the `u_ell` setting's `run_concurrently` propagation (it lives here because rule 2 of the ext guard forbids `Engine/` under `HeckeExt/`). **`HeckeUEllReachSlow`** (`slow`): the forced-wide arm at `ell = 3617`; the certified `tau` table at `ell = 5003` and `10007`; `U_5003` at `n = 2` **end to end** with `HeckeBadPrimeValidation`'s oracle rows; the `ell = 43867` coset tables; and the new limit -- at `ell = 43867` the table builds and is refused by `certify()` code 51 |
| `../HeckeExt/HeckeUEllBudget.h` (present, header-only) | E1b-0 / slot-3 WP-1 | [X] Sec. 1.5 `u_ell`, Sec. 2.2 Stage 1, [XA B3]. The **frozen** RSS model `rss_mb = (ell+1)(c0 + c1 (n+1)^2)` with `c0 = 0.123152`, `c1 = 2.24985e-4` (`u_ell_rss/v2-2026-09-17`), re-fitted on 16 measured points; `PredictUEllRssMb`, `UEllDecision(n, ell, budget_mb)` and its `UEllSetting` overload, `MaxAdmittedN`, `AdmitsNothing`. Pure, allocation-free, no pipeline dependency. Consumed by `HeckeExtDriver` Stage 1 (E1b, slot-3 WP-2); validated by `HeckeUEllBudgetValidation` (`fast`) |
| `InducedLocalSystem.h/.cpp` | S5 | [C, def:inducedmodule], [C, prop:blockmatrices], [C, eq:MH], [C, def:inducedls], [C, prop:lhcompliant] |
| `HeckeInducedLocalSystemSmokeValidation.h/.cpp` (present) | S5 | S5 Definition of Done (builds `L_{Gamma_0(2)}`/`L_{Gamma^0(2)}` and runs `EquivariantCohomologyOnK` UNCHANGED on it, [C, prop:lhcompliant]); distinct from the independent S5V validation `HeckeShapiroValidation` below |
| `HeckeShapiroValidation.h/.cpp` (present) | S5V | [C, obl:shapiro], [C, prop:lhcompliant] (oracle O11, independently re-derived dimension check on `L_{Gamma_0(2)}`/`L_{Gamma_0(3)}`) |
| `HeckeMorphismGeneralizationValidation.h/.cpp` (present) | S6 | [C, rmk:nogo] resolution: legacy single-fiber-matrix ctor vs. an explicitly-built one-entry `FiberBlockTable` give identical `get_induced_H_eq_maps()` (the class itself, `EquivariantCochainComplexMorphismOnK.h/.cpp`, lives outside `Hecke/` — see "Files elsewhere" below) |
| `HeckeMorphismRegressionValidation.h/.cpp` (present) | S6V | [C, rmk:nogo] (INDEPENDENT bit-identity regression of `get_induced_H_eq_maps()` against reference values captured on the pre-S6 commit `41df5e2`) |
| `HeckeOperatorOnK.h/.cpp` (present) | S7 | [C, def:xi], [C, thm:xi], [C, eq:xiexplicit], [C, eq:heckechain], [C, alg:assembly], [C, prop:symconv](2), [C, obl:welldef], [C, obl:agreement]. **Matrix-engine Phase 3:** `Theta^{(q)}` is an `N_q x N_q` `engine::BlockMat` over the uniform `(N_q, d_V)` partition whose pattern is the `(sigma, sigma_k)` pairs the tau table names; the fused route and route 2 are two predefined programs over that pattern (steps Theta5/Theta6/Theta7/Theta10 as `blk::gemm_dense`, Theta8 as one `blk::solve_multi_rhs` for all RHS, Theta9's residual as one `kernels::gemm`), compared with `blk::equal` **as mathematical matrices** so differing pools cannot make the comparison a tautology. New accessors `Theta_block(q)` / `block_pool()`; `get_Theta_flat_matrices()` survives as a lazily materialised, cached `to_dense` **shim** (its consumers are `HeckeCrossChecks`, `HeckeOperatorFamily::theta_flat*`, the JSON writer and the golden harness). `apply_Theta` is one `gemm_dense` on the block; `apply_route2` stays **matrix-free** (gemv + axpy per term) per its independence contract. The pre-Phase-3 dense assembly is kept verbatim as the legacy arm (`ENGINE_BACKEND=legacy`, `HeckeOperatorOptions::ScopedLegacyHeckeOperatorPath`); `-DENGINE_SHADOW_LEGACY=ON` runs both and compares entrywise. Pinned by `HeckeThetaBlockValidation`  **Matrix-engine Phase 4 (Stage B):** `compute_block()` no longer issues those block kernels one at a time -- it **records** them on the tape (`engine::tape`, `ENGINE_API` Sec. 10.9) and replays a cached compiled program: `hecke.theta` per degree (the fused assembly as kind-GEMMs, CSE computing each distinct `omega_i . rho_k` once, a wavefront's products batched into one `BATCHED_GEMM`, block rows writing disjoint outputs), `hecke.route2` per block row (route 2 **streamed** into arena temps and `COMPARE`d against the fused blocks -- nothing of route 2 is stored), `hecke.resid.chunk` (the A-N3 residual, one program per 512 right-hand sides `COMPARE`d against `W^T`; S23 PERF-P4a) and `tape::solve_multi_rhs` for `Theta_eq`. The program-cache key is the **tau-pattern hash** of the concrete `(n, p, r, q, ell)`, so a changed support misses the cache rather than replaying a wrong pattern (`EngineTapeConsumerValidation` asserts exactly that). Since the Phase 4 closure the degrees fan out over `run_concurrently` in the default mode and stay sequential in the memory-lean mode; the `Theta_eq` certification is what `--lean` / `certify` turn off (`"theta_eq_certified": false` in the JSON), and `A^{(q)}` is unchanged by the mode. **Phase 5:** the recorded program runs unchanged on the CUDA backend under `ENGINE_DEVICE=gpu|hybrid` -- device kernels are the exact-integer twins of the CPU ones, so no value depends on the device (`EngineCrossDeviceValidation`); the default is and stays `cpu`. **Phase 6:** the 7/11/5 literal became `RequireModel(SL2Z7VertexModel(), ...)` and `num_degrees_` now comes from the descriptor (`HeckeModelDescriptor.h/.cpp`). **S23 P4R:** route 2 is streamed key by key (one d × d accumulator per route-2 block; the distinct keys' `ρ_n` evaluated in batches of 64, each once, and added with delayed reduction; no per-degree memo `rho2`), then compared entrywise with the fused blocks; `HECKE_RHO2_STREAM=0` / `ScopedRho2Streaming(false)` is the rollback to the memo and its `hecke.route2` programs; gated by `HeckeRho2StreamValidation` |
| `HeckeOperatorSmokeValidation.h/.cpp` (present) | S7 | S7 Definition of Done (smoke: `A^{(0)}(T_2) = (3)` at `(p,ell,n) = (2,5,0)`, `A^{(1)}(T_2) = (4)` at `(2,5,2)`, O12(a) two-route bit-identity, active runtime assertions (i)/(ii)/(iii), parseable `to_json()`) |
| `HeckeCrossChecks.h/.cpp` | S8 | [C, rmk:barformula], [C, prop:barformula], [C, rmk:nogo] (the `alpha`/`omega` fiber-matrix morphisms + the `beta` per-simplex block cochain map, and the A-F3 localizing assertions). **Matrix-engine Phase 3:** mathematics, check order and return codes unchanged; only how the linear algebra is issued. `beta_bullet`'s fibre data is one `engine::BlockMat` per degree on a shared content-addressed pool (density <= 1/m: the `m^2 - m` zero sub-blocks per block row are STRUCTURAL, and each distinct `rho_n(g h)` is interned once); `alpha`, `omega` and the independent `Xi_g` are block matrices on the same pool; the `C^q_eq(K,R;L')` basis reuses `EquivariantCohomologyOnK`'s own `BuildACqBlockMat` + `blk::rref`/`kernel_basis` instead of a second dense copy; the three A-F3 identities are evaluated as whole-matrix block GEMMs over all of `C^q` at once (`blk::equal`), with failure diagnostics re-scanning in the original per-basis-vector order so a failing run names the same first offender. `beta()`'s dense `FiberBlockTable` form is materialised only on first use |
| `BarCocycleH1.h/.cpp` (present) | S8 | [C, prop:barformula], [C, rmk:barformula] (independent, from-scratch `H^1(Gamma; Sym^n F_ell)` via the amalgam `<S> *_{<S^2>} <ST>` and the corrected coset-permutation Hecke formula). **Matrix-engine Phase 3:** the route stays INDEPENDENT (its own division-free charpoly, its own Krylov minpoly, nothing shared with `FpPoly`, the route it cross-checks per C16), but its hand-rolled RREF-with-RHS solver **was absorbed** into `engine::elim::solve_multi_rhs` -- the plan's conditional absorption, granted here because this solver has no Z-first/mod-ell policy to reproduce (that belongs to `HeckeTauTable`'s triangle fill, which is NOT migrated) and its pivot rule is literally D5b's. `solve_multi_rhs` throws where the old body returned false, so `MinPolyOfMatrix` catches and converts back (inconsistency is how the degree is found there). Pinned by the new `BarCocycleSolverValidation` against a frozen byte-for-byte copy of the pre-absorption body |
| `HeckeCrossCheckValidation.h/.cpp` (present) | S8 | S8 Definition of Done ([C, obl:agreement] both routes: `omega.beta.alpha == Theta_g` bit-identically on all of `C^q(K;V)`, and bar-route charpoly/minpoly match `A^{(1)}(T_p)`) |
| `HeckeOracleValidation.cpp` | S7V | [C, sec:example], oracle families O1-O8, O14 (INDEPENDENT primary oracle battery for `HeckeOperatorOnK`; O12 is `HeckeIndependenceValidation`'s job, not this file's, despite this row's stage tag historically reading "S7V/S8V") |
| `HeckeIndependenceValidation.h/.cpp` (present) | S8V | [C, thm:welldef], [C, obl:agreement]/[C, obl:indep] (oracle O12 in full: two-route bit-identity, custom coset reps, `TauVertexPolicy` independence, the reachable weak form of double-coset-rep independence, `HECKE_PURE_INTEGER` verified externally) |
| `HeckeOperatorFamily.h/.cpp` | S9 | [C, def:operators], [C, prop:diamond], [C, thm:relations], [C, cor:ulrecursion]. **Matrix-engine Phase 4 (Stage B):** `build_pipelines()` / `prepare(p, r)` construct the coset / tau tables **serially in key order** under the process-wide `hecke::tau_construction_mutex()` (the tau build is first-match-wins over a fixed order and drives the non-atomic global `PredicateStats` -- audit Sec. 6.5 hazards 1-3), then run each pipeline's `compute()` concurrently and insert the results in key order; `pipeline_cache_` is a `std::map` behind `cache_mutex_`. The lock moved out of that file's anonymous namespace into the header so the **process has one lock and not two** -- the Phase 6 campaign driver takes the same one around every `HeckeCosetTables` / `HeckeTauTable` / `SL2ZTessellationTables` construction. `theta_flat*` still returns `std::vector<Matrix>` and is served by `HeckeOperatorOnK`'s lazily materialised `to_dense` shim |
| `HeckeOperatorFamilySmokeValidation.h/.cpp` (present) | S9 | S9 Definition of Done (`T_p_pow(p,1)==T_p(p)`, `T_pp(p)` scalar check, full-recursion-vs-direct-double-coset `T_{p^2}` identity, primitive-vs-full witness `T_p_pow_direct = T_p_pow_primitive_direct + p^n I`) |
| `HeckeRelationsValidation.h/.cpp` (present) | S9V | [C, obl:algebra] (oracle O9, all `ell >= 5`) and [C, obl:integral] (oracle O10, bad-prime `p = ell`, gated per amendment A-F1: hard at `p=ell=5`, recorded-only at `p=ell in {2,3}`) |
| `FpPoly.h/.cpp` (present) | S10 | `F_ell[X]` arithmetic substrate: exact poly arithmetic, division-free Berkowitz charpoly, Krylov minpoly, certified factorization (coeff order low→high per A-N4) |
| `HeckeModuleDecomposition.h/.cpp` (present) | S10/S11 | Plan §3 ("Definition: the three matrix output levels", level (c)): single-operator primary decomposition with unconditional certificates (S10, `ComputeSingleOp()`); the multi-operator MaxSpec(𝕋) simultaneous refinement (S11, `ComputeMultiOp()`), incl. the honest `residue_degree` / `residue_degree_lower_bound` witness-search certification (A-N4) |
| `HeckeSettings.h/.cpp` (present) | E0b / slot 2 WP-1 | The **mathematical settings layer** ([X] §1.2 `MathSettings`, §1.5 "The settings layer (R1)", [OD] R1): the second family of switches beside the `ENGINE_*` knobs — the ones that are part of what a result *means*. Engine-owned: the defaults and the validation of values live here and the campaign spec / console only *set* them by key, never twice. Slot 2 registers exactly one key, `locality` (`budget=<N>`, default `N = 64` = the shipped `kResidueWitnessMaxAttempts`, or `strict`); **slot-3 WP-1 (E1b-0) appended the second key, `u_ell`** (`auto | off | budget_mb=<M>`, default `auto` with `M = 2048`; `auto` and `budget_mb=2048` decide identically but are distinct settings, because a pin is recorded, never inferred). Its consumer is `HeckeExtDriver`'s Stage 1 (E1b, slot-3 WP-2) — nothing in this build reads it, so it cannot move a value. `ScopedUEll` is its scoped override. Carries `source ∈ {default, spec, console}`, the canonical text of the **non-default** entries and its FNV-1a-64 fingerprint (`0` at the defaults — so a golden mismatch can never be explained by a setting), `to_json()`, and a `thread_local` scoped override (`ScopedMathSettings` / `ScopedLocality`, the `ScopedLegacyHeckeOperatorPath` pattern) registered with `engine::tape::register_thread_context()` so that every task thread `run_concurrently()` starts inherits the launching thread's settings. **`locality = strict` is IMPLEMENTED since slot 3 WP-3 (E0b-S)**: it runs the same bounded linear witness family budget mode runs and then, on a block that family did not decide, the terminating Berlekamp route of [X] §3.1 — so the two modes are byte-identical wherever budget mode certifies, and strict was never silently degraded to budget mode in the meantime (slot-2 owner confirmation 3). `hecke::NotImplementedInThisBuild` survives as the named build-capability error; no settings value raises it today. Depends on nothing of the pipeline (`<string>`, `Fnv.h`, `EngineRuntime.h`) |
| `SubspaceOps.h/.cpp` (present) | E0a | The exact `F_ell` subspace toolkit shared by the level-(c) machinery: `SolveColumnsExact` / `HStack` / `KernelAsMatrix` / `RankOf` / `NilpotencyExponent` / `SplitMix64` / `PrimaryRCFBasis`, moved **verbatim** out of `HeckeModuleDecomposition.cpp`'s anonymous namespace (slot-1 plan §2.3 item 5: E0b and E1a both need them, and the `HeckeExt/` guard forbids `Hecke/` → `HeckeExt/`), plus the E0a additions `VStack` / `IsNilpotent` / `FrobeniusPower`, the E0b-S additions `RrefPivots` / `AlgebraCoords` / `frobenius_matrix` / `frobenius_kernel` (slot 3 WP-3: strict locality mode's whole decision is `rad A_S = ker Frob^N` and `B = ker(Frob_Ā − id)` — the Frobenius is `F_ell`-linear on a commutative `F_ell`-algebra, and the trace form is *not* usable, since it vanishes on `F_ell[x]/(x^ell)`), and the E1s addition `algebra_closure` (WP-3: the canonical RREF coordinate basis of the unital `F_ell`-algebra generated by pairwise-commuting matrices, shared by `HeckeSturmBound` and E1a's `HeckeLocalAlgebra`). Pure refactor — gated by `GOLDEN_STRICT=1` tier B's `decomp_*_U` |
| `HeckeLocalityValidation.h/.cpp` (present) | E0a + **E0b** + **E0b-S** | [E] Cert. 5.2 / Claims 3.1, 3.3 — the certificate, read-only as `hecke::CertifyLocality` (E0a) and **inside the refinement loop** (E0b, slot 2 WP-1). Registers **two** validations: `HeckeLocalityValidation` (`standard`, ~6.4 s, 1356 checks — Claim 3.1's 48 non-local families, each now **split** by `compute()` into its two maximal ideals with both sub-blocks certified and `certify() == 0`, plus, at a forced budget of one candidate, the **bounded exit** where the frozen pre-fix predicate still accepts the unsplit block and the certificate still rejects it, and the R3 **resume** arm `run(1) + resume(64) == run(64)` modulo `budget_history`; local algebras; the Teichmüller trap; the `k=1` analytic path vs the matrix path vs what `compute()` stored; the `HeckeSettings` arm (defaults, parsing, fingerprint, the named strict-mode error, propagation onto `run_concurrently()` tasks); and the **grid arm** over every decomposition the golden harness builds, with the loop's own certificate compared against the read-only verdict and all four hashes re-checked against `GoldenReference.inc`) and `HeckeLocalityProbe` (`slow` **and** interlocked on `HECKE_LOCALITY_PROBE=1`; the `(160, 691)` probe, recorded not asserted). **Slot 3 WP-3 adds arm 7, strict mode** ([X] §3.1 steps 1–6, [E] Cert. `sec:localcert` "Strict mode"): the `(2, 30)` covering example built here from a *certified* degree-30 irreducible and the search-free `x = u^{(2^30−1)/31}`, `y = u^{…/7}`, `z = u^{…/3}` recipe, with `deg min(a) = 15`, `deg min(b) = 10`, `deg min(a+b) = 6` and `dim F_2[a,b] = 30` all re-derived (budget mode `local = false` / `residue_degree_lower_bound = 30`, strict `local = true` / `f = 30` / `phase_reached = residue_search`, ~0.1 s against the 10 s gate); the Berlekamp split path at a forced budget of one candidate (`dim B = 2`, the same final partition as the budget loop's, block bases entrywise); **grid identity** — the whole sweep re-run under `locality = strict`, printing `LOCALITY_GRID_STRICT … nonlinear_phase=0 hashes=944 failing=0`; interruption at a step boundary plus `resume_locality()` equal to an uninterrupted run modulo `budget_history`; the fingerprint; the deterministic lexicographic fallback; a five-way tamper arm on `certify()`'s codes 20 / 23 / 25 / 26 / 30; and a concurrent budget-vs-strict arm |
| `../HeckeExt/HeckeSturmBound.h/.cpp` (present) | E1s / WP-3 | **Not under `Hecke/`** — the first unit of the extension layer `src/hacs/HeckeExt/` ([X] §1.1). [E] Def. `def:bprime` / Claim `cl:bprime`: the computed Sturm bound `B'(k, ell)`, the least prime bound `B` for which the `F_ell`-**algebra** generated by `T_q mod ell`, `q <= B` prime, on the Miller basis of `M_k(F_ell)` has dimension `dim M_k`; cap `12·B(k)` by default ([X] §1.5 `sturm.cap`). Pipeline-free (`q`-expansions of `E_4`, `E_6`, `Delta = q prod (1-q^n)^24` reduced mod `ell` as they are formed — no bignum, no division by 1728, correct at `ell in {2,3}`), `SturmBoundCertificate` with a re-deriving `verify()`, and the **resumable state** of [X] §1.3.1 R3 (instance 1; this is R3's first implementation): `state()` / `resume(new_cap)` / `budget_history`, a cap below the reached prime refused, a resume on a complete state a strict no-op. The closure itself is `hecke::algebra_closure` in `Hecke/SubspaceOps` (shared with E1a's `HeckeLocalAlgebra`, so the Sturm unit stays pipeline-free — [XA m16]) |
| `../HeckeExt/HeckeSturmBoundValidation.h/.cpp` (present) | E1s / WP-3 | `HeckeSturmBoundValidation` (`fast`, 0.011 s, 1306 checks): `q`-expansions against **exact-integer** `sigma_3` / `sigma_5` and `tau(2..7) = -24, 252, -1472, 4830, -6048, -16744` over `ell in {5, 7, 131, 691, 134217757}`, plus the independent identity `E_4^3 - E_6^2 = 1728 Delta`; `dim M_k` against `#{(a,b) : 4a + 6b = k}`; the echelon Miller basis and its stability in the precision; `T_q` on `M_12(F_691)` with `tau(q) == sigma_11(q) mod 691` and a nonzero square-zero part ([E] `cl:ramanujan`); **20 pinned bounds** — `B' = 2` on all ten Kummer cells of [E] Claim 1.8 and at `(12,691)`, `(34,131)`, `(162,691)`, `(200,691)`; `5` at `(24,5)`, `(30,5)`, `(32,5)`, `(36,5)`, `(60,5)`; `7` at `(28,7)`; `3` at `(36,7)` — each with `verify() == 0` and two tampered variants failing; the **proper** subalgebra `<T_2, T_3>` of dimension `2 < 3` at `(24,5)` and `(28,7)` (the counterexample to the naive `B(k)` sweep); the R3 arm (forced `cap = 3` at `(24,5)` recorded not thrown, `resume(36)` reaching `B' = 5` entrywise equal to a fresh run, `resume(2)` refused, complete-state no-op, `budget_history == [3, 36]`); and both `sturm.unavailable` modes as the pure `(outcome, mode) -> (final, reason)` map |
| `../HeckeExt/HeckeExtArchitectureValidation.h/.cpp` (present) | E1s / WP-3 | `HeckeExtArchitectureValidation` (`fast`, 0.02 s): runs `docs/matrix-engine/tools/check_heckeext_includes.sh` over the checkout through `std::system`, the `MatrixRetirementValidation` pattern. Four rules: (1) no `#include` of `HeckeExt/` outside `HeckeExt/`, `Campaign/`, `Golden/`, `Bench/`, `main.cpp`, `*Validation.*`, `ValidationRegistry.cpp`; (2) no `Console/`, `Cuda/`, `Engine/` or `BlockFlatMatrix.h` inside `HeckeExt/`; (3) `HeckeSturmBound.*` limited to `SubspaceOps` / `FpPoly` / `Matrix` / `ModArithmetic` / `EngineCheck` / `Fnv`; (4) no `unordered_` in the layer |
| `../HeckeExt/HeckeLocalAlgebra.h/.cpp` (present) | **E1a / slot-2 WP-4** | **Not under `Hecke/`** — the extension layer's Stage-3 unit. [E] `sec:localalg` / `alg:stage3`: for a block `V = H_m` that `CertifyLocality` has certified local, the local algebra `A_S = F_ell[t_p] ⊆ End(V)` (through `hecke::algebra_closure`, as an RREF coordinate matrix so membership is one solve), the coefficient field `K = F_ell[ŵ]` and the generators `n_p = t_p − P_p(ŵ)` for **every** `p` (`ŵ` and the `P_p` **consumed from the locality verdict**, with `force_matrix_path` so the `k = 1` analytic shortcut's missing `ŵ` is formed the same way), the ideal `m_S = (n_p)_p` with **no separate `π_w(w)` generator**, its powers, `ν`, `soc A_S = Ann_{A_S}(m_S)`, the Gorenstein verdict, the Hilbert function, and the tangent test **over `K`** giving `embdim = dim_{F_ell}(m_S/m_S²)/f`, `active` and the minimal `K`-spanning subset. Plus **generation certificate (a)** ([E] `cert:cyclic`, [X] §1.3 `CyclicityCertificate`) whose `verify()` re-derives the closure, the `n_p`, `m_S`, `dim m_S V` and `A_S v = V` rather than reading any of them (R17), and `SameMaximalIdeal`, Stage 2's cross-block evaluation test ([XA M5], `N'` taken for the **tested** block). Certificate (b) / the socle bound / `CommutantState` are WP-5's; Stage 5 is E2's |
| `../HeckeExt/HeckeExtAlgebraValidation.h/.cpp` (present) | E1a / WP-4 + WP-5 | `HeckeExtAlgebraValidation` (`standard`, 0.10 s, 4930 checks): constructed monomial algebras `F_{ell^f}[x_1..x_v]/(monomial ideal)` whose every invariant is combinatorial — `F[ε]/ε²`, `F[x,y]/(x,y)²`, `F[x]/(x³)`, `F_{ell²}[ε]/ε²` (**embdim 1 by the `K`-span test**, witness direction kept for `t = α + ε`), `F_{ell²}[x,y]/(x,y)²`, a non-cyclic `A ⊕ A` and the degenerate field (`m_S = 0`, `ν = 1`) — each run plain and conjugated by a seeded non-centralising `U`, over `ell ∈ {5, 13, 691, 134217757}`; the [XA B2] negative control (`P(w)` in place of `P(ŵ)` gives `n = 0`); the [XA M5] cross-block arms; and a ten-way tamper arm asserting the **exact** failing clause of each corruption. **WP-5** adds certificate (b): the socle bound against an independent naive `dim C` and a closed form on every case; `excluded_by_socle_bound` on the two Gorenstein non-cyclic cases (including the `(10, 691)` shape `A ⊕ A/m`, where the bound 5 is exact); the positive case `V = A^∨` for `A = F[x,y]/(x,y)²`, where (b) passes and (a) fails; structured == naive == closed form; the R3 `CommutantState` arms (interrupt at **every** reachable row count, resume, entrywise-identical state and certificate, a lower budget refused, `budget_history`); the OQ8 `final = false` emission; and a sixteen-way tamper arm with a positive control |
| `../HeckeExt/HeckeSturmSweep.h/.cpp` (present) | **E1b / slot-3 WP-2** | **Not under `Hecke/`** — the extension layer's Stage-1 and Stage-4(c) unit. [E] `cert:sturm` on the range of `cl:sturmmodl`: given a certified-local block and its `LocalAlgebra`, sweep `T_q|_V` for **every prime `q ≤ B'(k, ell)`** (`HeckeSturmBound`, `k = n + 2`) and test membership in `A_S` by one reduction against its RREF coordinate basis. The prime list is `q ≤ B'` and nothing else — `ell` is swept iff `ell ≤ B'` — and the `S_sep(ell)` additions of [E] `cl:sep` are deliberately E6's. `SturmCertificate` re-derives the whole list, the membership flags, `generators_certified`, `failing_prime` and the outcome (R17), and is R3-resumable on `sturm.cap` (`state()` / `resume(cap)`, a lower cap refused, a no-op on a complete state, `budget_history`). Plus **Stage 1**: `ComputeStageOneOperators` returns `S_0 = {2,3,5,7} ∪ {ell iff the frozen `u_ell` model admits U_ell}` with its `UEllVerdict`. Unlike `HeckeSturmBound` it MAY consume the pipeline, and does (`HeckeOperatorFamily`, `SubspaceOps`); it includes no `Engine/`, `Cuda/` or `Console/` header |
| `../HeckeExt/HeckeExtSturmSweepValidation.h/.cpp` (present) | E1b / WP-2 | `HeckeExtSturmSweepValidation` (`standard`, ~2.3 s, 255 checks) — the REAL-BLOCK arms. Stage 1 (`S_0` at `(10, 691)`; the pinned refusal `UEllDecision(16, 43867, 2048)`; `u_ell = off` removing `ell` from `S_0` even at `ell ≤ 7`; the two new keys `sturm` / `sturm_unavailable` with their parsing, canonical text, registry order and fingerprint). `(10, 691)`: `A_S = F_691[ε]/ε²` (`D = 3`, `dim A_S = 2`, `ν = 2`, `f = 1`, Gorenstein), (a) fails, (b) `excluded_by_socle_bound` with bound 5 == the INDEPENDENT naive `dim C = 5`, (c) `sturm(B' = 2)` with `generators_certified = [2]` and `final = true`, plus the two arms deferred from slot-2 WP-5 (every `T_q|_V`, `q ≤ 11`, in `A_S`; the joint `X_1 ⊕ X_4` decomposition, run substantively at the Dickson-exceptional `(12, 5)` where `X_1 ≠ 0`, each per-group ideal matching exactly one joint block by `SameMaximalIdeal`). The **B(k) counterexamples** `(22, 5)` and `(26, 7)`: on the `{2,3}` decomposition one block REJECTS `T_5` (resp. `T_7`), on the `{2,3,5}` (resp. `{2,3,5,7}`) one every block passes. Both `sturm.unavailable` modes on the forced `cap = 3` at `(22, 5)` with the real `A^{(1)}(T_q)`, the R3 arms (`resume(36)` == a fresh run entrywise modulo `budget_history`, `resume(2)` refused, a no-op resume, `budget_history == [3, 36]`), `sturm = off`, a refused `U_7` inside the sweep at `(26, 7)` (`u_ell_blocked`, no partial sweep published), `not_applicable` at `n = 0`, a fifteen-way tamper arm, and the recorded ext-layer wall time at `n = 256` |
| `../HeckeExt/HeckeExtFinal.h` (present, header-only) | **E2a / slot-4 WP-1** | **Not under `Hecke/`.** THE ONE TRI-STATE of [X] §1.3.1 R3.2: `hecke_ext::Final {True, False, Heuristic}` with `ToString` / `ToJsonValue`, `FinalMode`, `Finality {value, reason}` and the **fixed reason list** (`not_local`, `sturm_unavailable`, `sturm_heuristic`, `closure_unreachable`, `certificate_failed:<code or certificate name>`, `budget_exhausted:<kind>`, `interrupted`, plus E2a's record-level `not_applicable` for `n = 0`), with `IsFinalReason` / `IsResumableKind` as the predicates a `verify()` and a validation run over every emitted string. `SturmFinal` / `SturmFinality` in `HeckeSturmBound.h` are now **aliases** of `Final` / `Finality`, so `HeckeSturmBound`, `HeckeGenerationCertificates`, `HeckeSturmSweep` and the record layer share one type; the JSON strings are unchanged. Dependency-free (`<cstdint>`, `<string>`, all inline), which is why `check_heckeext_includes.sh` rule 3's list admits it |
| `../HeckeExt/HeckeModuleStructure.h/.cpp` (present) | **E2a / slot-4 WP-1** | **Not under `Hecke/`** — the extension layer's **Stage 5** ([E] `alg:stage5`(1)–(3),(5), [X] §2.2 Stage 5). On a certified-local block and its `LocalAlgebra`: the Loewy filtration `m^{j+1}V = Σ_a n_a (m^j V)` and the socle filtration `soc^{j+1}V = {x : n_a x ∈ soc^j V}` with their layer dimensions **in units of `f`** ([E] `cl:frank`), the Jordan type of each `n_p` on `V` in units of `f`, and the **structure maps** `θ_j` per consecutive layer pair and per generator (rows = the bottom-layer basis, columns = the top-layer basis, [E] §7) with per-generator and stacked ranks; `cyclic` / `free` / `min_generators`. Every subspace is a canonical `hecke::RrefRowSpan`, so the layer bases are a function of the SET they span and an entrywise comparison is meaningful. `verify()` re-derives the whole of Stage 5 from certificate (a)'s stored data by calling the SAME builder (R17). [E] `alg:stage5`(4), the canonical submodules of D6, needs E3 / E4 / E5 and is not here |
| `../HeckeExt/HeckeExtRecord.h` + `HeckeExtReport.h/.cpp` (present) | **E2a / slot-4 WP-1** | **Not under `Hecke/`** — the extension record **`hecke_ext/v2`** (**`hecke_ext/v3` since slot-7 stage 3**; [E] `sec:schema`, [X] §1.4, [C] `sec:recordschema`; `v1` through slot 4, bumped by slot-5 WP-1). The data model (`HeckeExtRecord`, `IdealRecord`, `ComplexRecord`, `SturmRecord`, `UEllRecord`, `ResumableRecord`, `ModuleRecordEntry`, `FinalInputs`), the one `is_final(FinalInputs, MathSettings)` — whose GENERATION half calls E1a/E1b's own `generation_verdict`, so the two verdicts cannot drift apart, and which adds `not_applicable`, interruption, locality and the two heuristic modes — and a hand-rolled writer in `HeckeReport.cpp`'s style. **A SEPARATE WRITER from `Hecke/HeckeReport`**: the golden harness hashes `hecke_report/v2`'s text, so a new key there would be a golden re-baseline, and slot 4 makes none. The per-complex degree fields are derived from each complex's own degree count. **Since slot-5 WP-1** `boundary` and `pair` are computed objects with their own degree fields and `les_maps` (they were `{"status": "unavailable"}` through slot 4), the `u_ell` verdict is the **cell's** with a single `resumable`, and an ideal carries its `sign` and a `groups` list; `WriteHeckeExtReport(record, path)` sets the record stem and writes any `> 64 KiB` resumable state to `<stem>.state.<kind>[.<i>].json`, referenced by relative path and FNV-1a-64, while a build with no stem inlines it and flags `state_oversize` (R3.1 item 6). The cone keys `phi_flat_identity` / `residual_flat_zero`, RESERVED through slot 4, are **emitted** inside an available `pair` since slot-5 WP-1. `verify()` gained **code 22** (slot-5 WP-11a: a `q_max_source = "set"` on a *derived* complex is refused, [C] `sec:recordschema` item 5) and **code 23** with the record-level `IotaExactComparisonRecord` (slot-5 WP-12a, [OD] 2026-09-22 Q3: the record's half of [C] `obl:iota`(ii)'s rider) |
| `../HeckeExt/HeckeExtRecordValidation.h/.cpp` (present) | E2a / WP-1 | `HeckeExtRecordValidation` (`fast`, ~0.1 s, **582 checks**) — see the entry in the validation list below |
| `DicksonCount.h` (present) | E0a | The public `hecke::DicksonInvariantCount(n, ell)` / `IsDicksonExceptionalCount(n, ell)` (slot-1 plan §3 WP-2 scope 1). Unifies the three former file-local `IsDicksonExceptional` copies (`HeckeOracleValidation.cpp`, `HeckeEndToEndValidation.cpp`, `Golden/GoldenHarness.cpp`) onto one function; the `HeckeOracleValidation.cpp` copy had under-approximated the locus (first Dickson generator only) |
| `../HeckeExt/HeckeIdentityCertificate.h/.cpp` (present) | E0a | The E0a subset of [X §1.3]'s `IdentityCertificate` (slot-1 plan §3 WP-2 scope 2): Eichler-Shimura in its `dim H^0`-robust form (`dim H^1 = dim M_{n+2} + dim S_{n+2} + dim H^0`, holds at Dickson-exceptional loci too), Hecke commutativity `[A_p, A_p']=0`, `d_2=0` (`dim H^2=0`), and the `p^2` recursion checked against the **independent** double-coset route (`T_p_pow_direct`, not the recursion's own output). `MatrixT<uint32_t>` (the `HeckeExt/` placement decision — carried in `check_matrix_includes.sh`'s allow-list with that reason); does not depend on `HeckeCrossChecks` |
| `../HeckeExt/HeckeIdentityValidation.h/.cpp` (present) | E0a | `fast`. Ten real pipeline points (classical, two Dickson-exceptional loci at different `ell`, `n=0`, odd `n`, `ell<5`) plus a four-way tamper arm (`dim_H1`, one entry of `A_p'`, one entry of `A_{p^2}`, `dim_H2` — each makes exactly the corresponding item fail while leaving the other applicable items passing) |
| `HeckeDecompositionSmokeValidation.h/.cpp` (present) | S10 | S10 Definition of Done (200 fixed-seed constructed-truth reconstructions over `ell ∈ {5,13,101,134217757}`; the `(X²+X+1)²/F_5` companion case; FpPoly corner batteries incl. the inseparable `f′=0` path and the `ell=2` trace-map split) |
| `HeckeBlockFormValidation.cpp` | S10V | oracles O6 (`ell=691,n=10`, non-semisimplicity, amendment A-F2) and O8 (`n=22`, the weight-24 Galois-conjugate-pair split-vs-nonsplit branch) |
| `HeckeReport.h/.cpp` (present) | S11 | Plan §3 "Output formats": hand-rolled JSON writer (no third-party library) covering all three matrix levels -- metadata, level (b) `A^{(q)}(T_p)` for every computed `p`, level (c) blocks (incl. `eigen_system` for degree-1 blocks and `residue_degree` vs `residue_degree_lower_bound`, A-N4), an OPTIONAL size-gated level (a) `Theta^{(q)}_flat` |
| `HeckeSimultaneousSmokeValidation.h/.cpp` (present) | S11 | S11 Definition of Done: `(ell,n)=(13,10)`, family `{T_2,T_3}` -> `{dim 2 cuspidal (Delta system, isotypic multiplicity 2, recorded), dim 1 Eisenstein (eigen-system `T_2↦8, T_3↦10`, asserted)}`; `(ell,n)=(691,10)` -> a single `dim 3` block re-deriving the S10V O6 result jointly with `T_3`; JSON round-trip through an embedded, from-scratch mini JSON parser |
| `HeckeSimultaneousValidation.h/.cpp` (present) | S11V | Plan §3 level (c) (INDEPENDENT re-check of `ComputeMultiOp()`/`BuildHeckeReportJson()` reading only the emitted JSON, own from-scratch parser + F_ell[X] toolkit); covers `(ell,n) ∈ {(13,10), (691,10), (10007,22), (101,22)}`, the last two selected via a runtime Legendre-symbol computation |
| `HeckeEndToEndValidation.h/.cpp` (present) | S12 | [C, sec:obligations] (all `obl:*`), full oracle battery; this one file registers **three** validations: `HeckeEndToEndValidation` (the acceptance gate: full `p x ell x n x q` oracle matrix, aggregate failure count), `HeckeBadPrimeValidation` (O10 bad-prime settings `p = ell ∈ {2,5}`), `HeckeBenchmarkSmoke` (performance smoke at `(ell,n)=(134217757,100)`, no wall-time assertion) |

Files elsewhere in the repo touched by this work (not under `Hecke/`):
`LocalSystemSL2Z.h/.cpp` (S2, `SL2ZRelationDatum` / `GenerateSL2ZRelationsWithGroupData`),
`EquivariantCochainComplexMorphismOnK.h/.cpp` (S6, per-simplex-block generalization),
`src/hacs/ValidationRegistry.cpp` (every `*V` stage registers
its validation there), `docs/latex/monodromy_v5.tex` (S14, documents the
`rmk:nogo` correction).

## The three matrix output levels (plan §3)

Every level is produced from the SAME `HeckeCosetTables`/`HeckeTauTable` data
for a fixed `(p_hecke, ell, n_sym)`; none re-derives the coset/tau tables.

* **Level (a) — cochain-complex level.** `Theta^{(q)}_flat` (the fused
  operator `Theta_g = omega_bullet . Xi_g` on the full flat cochain space
  `C^q(K;V)`) and `Theta^{(q)}_eq` (its restriction to `C^q_eq` in the stored
  basis). Produced by `hecke::HeckeOperatorOnK::compute()`
  (`HeckeOperatorOnK.h/.cpp`, S7), retrieved via `get_Theta_flat_matrices()` /
  `get_Theta_eq_matrices()`; a second, independent evaluation route
  (`apply_route2`, direct bracketing `rho_n(delta_i) . rho_n(gamma_k)` rather
  than the fused `rho_n(s_i^{-1}) . rho_n(g gamma_k)`) is asserted
  bit-identical to the fused route at construction time ([C, obl:agreement],
  O12(a)).
* **Level (b) — cohomology level.** `A^{(q)}(T_p)`, the `d_q x d_q` matrix of
  `T_p` on `H^q(K,R;L)` in the stored `H_eq_bases` basis, column convention
  `T_p[z_i] = sum_j A^{(q)}_{ji} [z_j]` ([C, alg:assembly]). Produced by
  `HeckeOperatorOnK::get_H_matrices()` for a single prime, and by
  `hecke::HeckeOperatorFamily` (`HeckeOperatorFamily.h/.cpp`, S9) for the full
  algebra: `T_p`, `T_p_pow`/`T_n` (the standard FULL determinant-`p^r`
  operator, recursion `T_{p^{r+1}} = T_p T_{p^r} - p^{n+1} T_{p^{r-1}}`),
  `T_p_pow_direct` (an INDEPENDENT double-coset route used only for
  cross-checking, see the `T_{p^r}` normalization erratum in the plan's
  §0.2), and `T_pp` (the diamond scalar `p^n . I`).
* **Level (c) — block-diagonalized level.** The primary decomposition of
  `H^q` as a module over the commutative F_ell-subalgebra generated by one
  operator (`HeckeModuleDecomposition::ComputeSingleOp()`, S10) or a
  commuting family (`ComputeMultiOp()`, the `MaxSpec` simultaneous
  refinement, S11), each block carrying its minimal/characteristic
  polynomial per operator, residue degree, generalized-Jordan partition, and
  a certified change of basis (`HeckeModuleDecomposition.h/.cpp`). Exported
  in the plan §3 JSON schema (all three levels, level (a) optional and
  size-gated) by `hecke::BuildHeckeReportJson()` (`HeckeReport.h/.cpp`, S11).

## Running the validations

Every validation is a zero-argument `uint32_t SomethingValidation()`
registered by name in `ValidationRegistry::RegisterAllValidations()`
(`src/hacs/ValidationRegistry.cpp`) and runnable standalone,
without editing `main.cpp::main()`:

```bash
cmake . && make
./hacs --list            # print every registered validation name
./hacs <Name>             # run exactly one; exit code = its own return code (0 = pass)
./hacs --all              # run the fast + standard tiers; exit code = failure count
./hacs --all-full         # run every tier, slow included; exit code = failure count
./hacs                    # no args: legacy default behaviour (benchmark_local_system_and_morphisms), unchanged
```

`--all` and `--all-full` both exit 0 in this repository (the two pre-existing
non-Hecke defects -- the `BlockFlatMatrixValidation` hang and the
`SimplicialCohomologyValidation` crash -- were fixed on 2026-07-03; see
`docs/KNOWN_ISSUES.md`). Since matrix-engine Phase 0
each validation carries a tier and `--all` skips the `slow` tier, whose three
members are `HeckeBenchmarkSmoke`, `SymmetricPowerValidationSlow` and
`EngineDeviceTrapValidation`: **the acceptance gate therefore requires
`--all-full`**, or run the three names directly
(`./hacs HeckeEndToEndValidation`, `HeckeBadPrimeValidation`,
`HeckeBenchmarkSmoke`).

The full list of currently registered validation names (grep the source of
truth, `ValidationRegistry.cpp::RegisterAllValidations()`, to keep this in
sync):

Pre-existing (Stage S0 preconditions, not part of this feature):
`mod_arithmetic_validation`, `MatrixValidation`, `FlatMatrixValidation`,
`BlockFlatMatrixValidation`, `SimplicialCohomologyValidation`,
`LocalSystemCohomologyValidation`.

Hecke feature (Stages S1-S12; run each with `./hacs <Name>`):
`HeckeExactIntSmokeValidation`, `HeckeFpFilterValidation`,
`HeckeTessellationSmokeValidation`, `HeckeTessellationValidation`,
`HeckeCosetTablesSelfCheckValidation`, `HeckeCosetTablesValidation`,
`HeckeTauTableSelfCheckValidation`, `HeckeTauTableValidation`,
`HeckeInducedLocalSystemSmokeValidation`, `HeckeShapiroValidation`,
`HeckeMorphismGeneralizationValidation`, `HeckeMorphismRegressionValidation`,
`HeckeOperatorSmokeValidation`, `HeckeOracleValidation`,
`HeckeCrossCheckValidation`, `HeckeIndependenceValidation`,
`HeckeOperatorFamilySmokeValidation`, `HeckeRelationsValidation`,
`HeckeDecompositionSmokeValidation`, `HeckeBlockFormValidation`,
`HeckeSimultaneousSmokeValidation`, `HeckeSimultaneousValidation`,
`HeckeEndToEndValidation`, `HeckeBadPrimeValidation`, `HeckeBenchmarkSmoke`,
`HeckeExtendedRegimeValidation`, `HeckeThetaBlockValidation`.

The hecke-extensions programme's phase E0a adds two more (see the
`HeckeSettings` / `SubspaceOps` / `HeckeLocalityValidation` rows of the file
table above): `HeckeLocalityValidation` (`standard`, ~8.1 s, 1478 checks; the
locality certificate `hecke::CertifyLocality`, **since slot-2 WP-1 (E0b) the one
the refinement loop itself enforces** and **since slot-3 WP-3 (E0b-S) in strict
mode as well**, seven arms, and the two grid verdict lines
`LOCALITY_GRID ... nonlocal=0 witness_changes=0 loop_disagreements=0
hashes=944 failing=0` and `LOCALITY_GRID_STRICT ... nonlinear_phase=0
hashes=944 failing=0`) and `HeckeLocalityProbe` (`slow` **and** interlocked on
`HECKE_LOCALITY_PROBE=1`, so it runs in neither `--all` nor `--all-full` nor
`cross_config.sh`; it prints `LOCALITY_PROBE n=160 ell=691 ...` and is
RECORDED, not asserted). The golden harness gains the owner-gated `GoldenDump`
(`GOLDEN_DUMP=<dir>`), the "before"/"after" half of E0b's re-baseline
protocol, diffed by `docs/matrix-engine/tools/golden_explain.py`.

Slot-1 WP-2 adds one more: `HeckeIdentityValidation` (`fast`; see the
`HeckeExt/HeckeIdentityCertificate` row of the file table above) --
`hecke_ext::IdentityCertificate`'s E0a subset, real pipeline points plus a
four-way tamper arm.

The involution chapter of `gl2_contract.tex` (chapter 9, extension programme
E3 / amendment C-A1) adds two more:
`HeckeIotaTauSymmetryValidation` (`fast`, 0.5 s; `obl:iota`(i) gating, and
`obl:iota`(iv) gating on the canonical tau since the 2026-09-16 re-baseline,
recorded-only on the legacy policy) and `HeckeIotaSignValidation` (`standard`, 1.4 s;
`obl:iota`(ii), (iii), (v) gating -- since contract v1.2.0 including "the Eisenstein
class lies in `H^1_{eq,+}`", true by definition under `iota^* := -iota^*_geo`, and the
degree-0 anchor `iota^* = -1`; the geometric sign and the `+-` split recorded; prints one
`IOTA_SIGN eisenstein=plus (geometric=minus) ...` line).

The boundary-datum chapter of `gl2_contract.tex` (`sec:boundarydatum`, extension programme
E4a / slot-1 package WP-4a) adds `HeckeBoundaryTauValidation` (`standard`, 0.03 s;
`obl:taubdry`(i)-(iv) GATING over Z on every table of its grid, `ell in {2, 3}` and the u64
primes included, plus `prop:bdrycosets`(1)-(3) and the (B1)-(B3) subcomplex data; the
operator clauses (v)-(vii) are WP-4b's). See `HeckeBoundaryDatum.h` for the object.

Its second half (E4b / slot-1 package WP-4b) adds `HeckeBoundaryOperatorValidation` (`standard`,
5.4 s; `obl:taubdry`(v) GATING -- route A, the Shapiro transcription on the infinity-line, and
route B, the closed forms, both equal to the normative `eq:bdrychain` operator on cohomology --
(vi) GATING for `chi_b(p) = p^b + p^{n+1-b}` on `x^{n-b ell} Nm^b` and `1 + p^{n+1}` on `[y^n]`
at `n < ell`, with the general-`n` `H^1` multiset of `cl:chi`(2) RECORDED (330/330 cells agree),
(vii) GATING at `ell >= 5` and RECORDED at `ell in {2, 3}` (216/216 pairs hold); the `beta_j`
right-coset negative control gives `chi_{b+1}` and breaks (vii) on 170 cells; the circle model is
built at every `ell` and RECORDED, never refused). See `HeckeBoundaryOperator.h` for the object.

Slot-2 package WP-7 (phase E4b, the ext-layer consumer `HeckeExt/HeckeBoundary` on top of the two units
above; see `../HeckeExt/README.md`) adds `HeckeBoundaryParabolicValidation` (`standard`, ~4.7 s, 4966
checks on 75 cells): `H^1_! = ker r^1` as a `T_p`-stable submodule with its action, the joint `1 + p^{n+1}`
eigenspace, Harder's criterion ([E] `cl:harder`) asserted on all ten Kummer cells of [E] Claim 1.8 and on
every non-Kummer cell with `n < ell - 1`, the Harder split, the classical `dim H^1_! = 2 dim S_{n+2}`,
and the two diagnostics of `HeckeBoundaryDiagnostics` above -- RECORDED, gating nothing: on the canonical
tau `cochain_commutes = false` and `eta_zero = false` on every one of the 315 `(cell, p, nu)` pairs
(the bulk tau is not a map of pairs), with the one theorem `eta_zero => cochain_commutes` asserted and
the `p = 2` walk pinned to the canonical table's shape (10 of 12 vertex slots and 5 of 9 edge slots leave
the boundary; `rmk:conestrict`(1)'s `8 / 2` is the pre-re-baseline count). Prints one
`BOUNDARY_DIAG cells=315 cochain_commutes=0/315 eta_zero=0/315 ...` line and one `BOUNDARY_PARABOLIC ...` line.

Slot-2 package WP-8 (phase E5-1, the first third of the cone chapter `sec:cone`) adds
`HeckePairHomotopyValidation` (`standard`, 0.33 s, 113292 checks on 152 tables): the certified pair
homotopy `eta` of `HeckePairHomotopy` -- `boundary eta + eta boundary = j tau_bd - tau j` over Z, built
by `lem:etaexists` against the shipped bulk tau and WP-4a's `tau_bd` on ONE coset table -- with
`obl:eta`(i)-(iv) GATING on every table for EVERY ell (`ell in {2, 3}` and the u64 primes included; the
identity re-derived on every stored slot and on random Gamma-translates through an own (T1) translation
and an own boundary operator; the (T1) bookkeeping across every stored transporter; solve regime Z with
no mod-m path), the ell/n/width independence of the table, nine tampered chains per tamper cell rejected,
`eta_zero` RECORDED and asserted only to agree slot by slot with the cusp-support walk (false on all 152
tables: 5266 of 7504 slots are nonzero, so the eta-table is genuinely needed, as `rmk:conestrict`(1)
predicts), and `lem:etaunique` exhibited on the two vertex-chain policies (2342 differing slots, every
difference `boundary L - L boundary` for the unique degree-2 `L`). Every one of the 3216 edge fills was
solved on its initial neighbourhood (`max_rings = 0`) with coefficients in `{+-1}`. Clause (v) and the
cone are slot 3's. Prints the `p = 2` table and one `PAIR_HOMOTOPY tables=152 checks=113596 failures=0 ...` line.

Slot-3 package WP-5 (phase E5-2, the second third of `sec:cone`: the cone itself) adds
`HeckeConeValidation` (`standard`, 1.9 s, 5350 checks on 74 cells): the equivariant mapping cone
`Cone(r)` of the pair (7-vertex model, circle model) as the generic `EquivariantMappingCone`
(`eq:conediff` in `H_c`-indexing on the flat layout of `alg:assembly`, all-scalar `BlockMat`
differential, `Z / B / H / P` through `engine::blk` with the pipeline's D5b rules, the three map
families `pi_*^q`, `r^q`, `iota_*^q` of `eq:coneles` in stored bases, the per-complex degree fields
of `open:ell2` DERIVED -- `bulk 2 / boundary 1 / cone 2`, none truncated -- and `ConeCertificate`
with a thirteen-item re-deriving `verify()`), built on the pipeline-side `SL2ZBoundaryPair` (the
restriction lifted verbatim out of `HeckeExt/HeckeBoundary`, value-preserving). Asserted on every
cell: the certificate; the own dense `eq:conediff` == the stored `D_c` and own `D^2 = 0`; exactness at
EVERY spot of `eq:coneles` by own ranks (composites zero, `rank in + rank out = dim`, and
`im pi^1 = ker r^1` explicitly); the connecting map by an own zig-zag == `r^q` == the morphism's
induced map == `HeckeBoundary::r(q)`; `H^0_c = 0` at every `ell` and `n` (`lem:conerelative`(4)),
`dim H^1_c = dim H^0(bd) - dim H^0 + dim ker r^1` at every `ell` (it needs no `H^2_eq = 0`),
`dim H^2_c = dim H^1(bd) - rank r^1` and `H^2_eq = 0` at `ell >= 5` (recorded at `ell in {2, 3}`:
0 of 18 cells non-zero), `dim H^1_c - dim coker r^0 = dim H^1_!`; `dim H^1_c = 3` at `(10, 691)`
(`1 - 0 + 2`, App. E v1.4.2 item 1) and `dim H^1_c = 1 + 2 dim S_{n+2}`, `H^2_c = 0`, `rank r^1 = 1`
at all ten Kummer cells; the truncated branch of `obl:coneexact`(iv) on a declared truncation of the
cylinder rel. both boundary circles; a three-way tamper arm; and the golden collectors' digest
(`dim_Hc^q`, ranks, `Hc^q`, `Pc^q`, the maps -- DEFINED, never emitted) equal across two in-process
builds, `ENGINE_THREADS = 1 / 16` children and `ENGINE_BACKEND=legacy`. Prints the per-cell table
and one `CONE cells=74 kummer=10 ell23_cells=18 ell23_H2_nonzero=0 checks=5350 failures=0` line.
The action arms (`Phi_g`, `h = eta~`, `obl:eta`(v), `obl:coneexact`(i)-(iii), (vi), (viii)) are
slot-3 WP-6's, in the second validation name below.

`HeckeConeActionValidation` (`standard`, 11.9 s, 123634 checks on 63 cells / 212 operators): the
HECKE ACTION on the cone (`HeckeConeAction`). `h := eta~` is the coset transport of WP-8's
`eta`-table and is never chosen; `Phi_g(c, c_bd) = (Theta_g c, Theta^bd_g c_bd + h c)`; `A_c^{(q)}`
is `alg:assembly` on `Cone(r)`. Asserted on every operator: the certificate
(`ConeActionCertificate`, items 1-11); `obl:eta`(v), the stored `h^q` against an OWN coset
transport of the `eta`-table; `obl:coneexact`(i), the residual bracket of `lem:conesign` assembled
densely here and annihilating the stored `C^q_eq` basis in every degree (424/424), with
`d h + h d = Theta^bd r - r Theta` and `Dcone Phi = Phi Dcone` on the equivariant cone cochains --
**the full flat identity is recorded, not claimed**: the transport formula uses `phi`'s
`Gamma`-equivariance, and it holds flat in 29 of 424 degrees; (ii), `Dcone Phi z = 0` on the stored
cocycles and `A_c^{(q)} = P_c Phi H_c`; (iii), `pi A_c = A pi` and `iota A_bd = A_c iota` against
the pipeline's own `HeckeOperatorOnK` / `HeckeBoundaryOperator`, with `r A = A_bd r` asserted at
`ell >= 5` and recorded at `ell in {2, 3}` (424/424 hold); (vi), `[A_c(T_p), A_c(T_p')] = 0`
(298 pairs) and the `nu = 2` recursion `A_c(T_{p^2}) = A_c(T_p)^2 - p^{n+1} I` with the **full**
`T_{p^2} = T'_{p^2} + p^n I` (37 degrees), the bulk analogue cross-checked against
`HeckeOperatorFamily::T_p_pow_direct`, the independent double-coset route; the two bracketings of
`h` and the matrix-free `apply_Phi_direct` on every standard basis vector; `(viii)(a)`, the four
combinations of `EtaVertexPolicy x TauBoundaryVertexPolicy` giving **bit-identical** `A_c^{(q)}`
while both table hashes move (asserted -- the non-vacuity rule of the slot-3 audit); `(viii)(b)`,
the negative control `h' = h + xi` with `xi = v_0 u^T` of `prop:coneambiguity`(2), which fails
**exactly** `obl:eta`(v) and changes `A_c^{(1)}` at `(10, 691)`; a two-way tamper arm; and
`eta_zero => cochain_commutes`, both false on all 212 operators. Prints
`CONE_ACTION cells=63 operators=212 ell23_operators=74 checks=123634 failures=0`.

Slot-3 package WP-1 (phase E1b-0, the `U_ell` exact-integer reach) adds three:
`HeckeUEllReachValidation` (`standard`, ~11 s, 6160 checks), `HeckeUEllReachSlow` (`slow`) and
`HeckeUEllBudgetValidation` (`fast`, 575 checks) -- see their two file-table rows above. The
package closes `docs/KNOWN_ISSUES.md` Sec. 3 and **corrects its diagnosis**: the
`hecke: exact multiply overflow` at `p = ell >= ~4580` was never in the coset construction but in
`QuadPoint::moebius` inside [C, alg:tau] step 1's fundamental-domain reduction, and it was not a
failure of [C, lem:bounds] (whose certified `B = max(B_tab, 4 N^2)` is `1.0e8` at `ell = 5003`,
`2 B^2 = 2.0e16 << 2^63`, and `certify()` clause (e) passes). The repair is
[C, thm:backend]'s own `__int128` tier, attempted only after the `int64` tier throws, so it is
bit-identical wherever `int64` completes by construction -- `GoldenReference.inc` unchanged,
determinism digest `0xe446b3bd2d28bd6e`, `tau_table_p2.txt` byte-identical. The remaining limit,
`ell <= 23170`, is `lem:bounds`(5)'s own bound on the STORED transporters and would need a
storage widening to lift; `u_ell = auto` refuses every such cell on memory grounds first.
(Slot-24 ELL-W lifts it without a storage widening -- see the `HeckeTauTable` row and
`HeckeWideTauValidation`; `docs/implementation/slot24/ELLW_report.md`.)

Phase E1s (slot-1 package WP-3) opens the extension layer
`src/hacs/HeckeExt/` and adds two `fast` validations:
`HeckeSturmBoundValidation` (0.011 s, 1306 checks; the computed Sturm bound
`B'(k, ell)` of [E] `def:bprime` / `cl:bprime`, its certificate, and the first
implementation of the resumable-state contract [X] §1.3.1 R3) and
`HeckeExtArchitectureValidation` (0.02 s; the `HeckeExt/` one-way include
guard). The unit under `HeckeExt/HeckeSturmBound` is **pipeline-free**: it
builds no cohomology and no coset table, only `q`-expansions mod `ell` on the
Miller basis, which is what lets the bound be computed and certified before any
Hecke matrix exists. The slowest case on the pinned list is `(k, ell) =
(258, 691)` at **1.4 ms**, against [X] §6.2 E1 acceptance (d)'s 0.1 s target.

Phase E1a (slot-2 packages WP-4 and WP-5) adds the layer's Stage-3 and
Stage-4(b) units `HeckeExt/HeckeLocalAlgebra` and
`HeckeExt/HeckeGenerationCertificates`, and one `standard` validation,
`HeckeExtAlgebraValidation` (0.10 s, 4930 checks): the local algebra `A_S` of a
certified-local block with its invariants (`dim A_S`, `ŵ`, the `P_p`, the `n_p`,
`m_S` and its powers, `ν`, `soc A_S`, Gorenstein, the Hilbert function, the
`K`-tangent table and `embdim`) and **generation certificate (a)** of
[E] `cert:cyclic`. It runs on constructed truths only — monomial algebras whose
invariants are read off the monomial set — with the two `f = 2` traps of
[E] `cl:resfield` / [XA B2] pinned by their own arms, the [XA M5] cross-block
evaluation test, and a tamper arm in which each corruption fails exactly its own
`verify()` clause.

**WP-5** adds **generation certificate (b)** of [E] `cert:comm`
(`HeckeExt/HeckeGenerationCertificates`, [X] §2.2 Stage 4(b), §1.3, §1.3.1
R3.3 item 3): the **socle bound**
`dim C ≥ dim A_S + f·t·s − dim_F soc A_S` with `t = dim_F(V/m_S V)` and
`s = dim_F soc V = dim_F ⋂_p ker n_p`, whose gate `f·t·s ≤ dim_F soc A_S` makes
`excluded_by_socle_bound` a **proof** that (b) fails; the structured system
`X t_p = t_p X` inside the explicit `u`-dimensional basis of
`End_{F_ell[a]}(V)` for the fixed cyclic-most `a` in its rational canonical
form; the **naive** `O(k D⁶)` commutant as the validation oracle only; the
budgeted, resumable `CommutantState` (`kCommutantBudget = 2·10^10` modmul,
`over_budget` **never** a silent naive fall-back); and the route-combining
`generation_verdict` of [X] §1.3.1 R3.2, under which an ideal that cannot reach
closure is **emitted** `final = false` with `closure_unreachable` /
`budget_exhausted:commutant`, never withheld (OQ8). The real-block
(Sturm-membership, joint decomposition) arms are E1b.

Phase **E1b** (slot-3 package WP-2) closes them. `HeckeExt/HeckeSturmSweep` is
**generation certificate (c)**, [E] `cert:sturm`, on the pipeline's own blocks,
together with **Stage 1**'s `u_ell` budget check; one `standard` validation,
`HeckeExtSturmSweepValidation` (~2.3 s, 255 checks), carries the real-block
arms. What (c) proves is bounded by [E] `cl:sturmmodl`: on a non-Dickson maximal
ideal of `X_4`, every prime `q ≤ B'(k, ell)` with `T_q|_V ∈ A_S` gives
`A_S = T_m`; off that range the same computation is a heuristic and the
certificate records it as one (`proof_range` is an input the caller asserts,
because identifying Dickson-type `m` needs E6's residue atlas). The sweep
**never** uses `B(k) = ⌊k/12⌋ + 1`, and the validation reproduces exactly why on
the pipeline's own `H^1`: at `(n, ell) = (22, 5)` and `(26, 7)` the `{2, 3}`
decomposition has a block whose `A_S` does **not** contain `T_5` (resp. `T_7`),
so the naive certificate would have accepted a false record. The two `sturm.*`
settings are registered here as `Hecke/HeckeSettings`' third and fourth
mathematical keys (`sturm = auto | off | cap=<B>`,
`sturm_unavailable = nonfinal | heuristic`, appended after `u_ell`); under the
default `nonfinal` a `none_at_cap` bound sweeps nothing and the ideal is
`final = false` with reason `sturm_unavailable`, while `heuristic` sweeps the
primes `≤ cap` and emits `final = "heuristic"` — a value the defaults never
produce. The sweep is R3-resumable on `sturm.cap`, and where the proof list
contains `ell` and the `u_ell` budget refuses `U_ell`, (c) is `unavailable` with
`u_ell_blocked` and no partial sweep is published.

Phase **E2a** (slot-4 package WP-1) adds the extension record's own layer, and
one `fast` validation, `HeckeExtRecordValidation` (~0.1 s, **582 checks**).
`HeckeExt/HeckeExtFinal.h` is [X] §1.3.1 R3.2's **one** tri-state `Final` with
its fixed reason list, to which `SturmFinal` / `SturmFinality` are now aliases,
so `HeckeSturmBound`, `HeckeGenerationCertificates`, `HeckeSturmSweep` and the
record layer share a single type with a single reason vocabulary — and no
emitted JSON string moved. `HeckeExt/HeckeModuleStructure` is **Stage 5** ([E]
`alg:stage5`): the Loewy and socle series and the Jordan types in units of `f`,
and the structure maps per consecutive layer pair and per generator with their
ranks, all on canonical `RrefRowSpan` bases and all re-derived by a `verify()`
that calls the same builder. `HeckeExt/HeckeExtRecord.h` +
`HeckeExt/HeckeExtReport` are the `hecke_ext/v2` schema of [E] `sec:schema` and
its hand-rolled writer — **a separate writer from `HeckeReport`'s
`hecke_report/v2`, which slot 4 does not touch**, because the golden harness
hashes that text — with the per-complex degree fields of [C]
`sec:recordschema` item 5 derived from each complex's own degree count, the one
`is_final` (whose generation half *calls* `generation_verdict` rather than
re-implementing it), and the R3.1 item-6 sidecar, whose stem only the record
layer knows. `Hecke/HeckeSettings` gains its fifth and sixth mathematical keys,
`commutant_budget` (whose default `kDefaultCommutantBudget = 2·10^10` now lives
there, with `CommutantRoute::kCommutantBudget` a `constexpr` alias, so a resume
with a raised budget is a new recorded setting) and `final_mode` — both appended,
both at their defaults, so the all-defaults fingerprint is still `0`. The
campaign spec parses every registered mathematical key, the driver installs them
per job and records them in the per-job JSON, and a non-zero fingerprint appends
`_s<16 hex>` to `Job::id()` and ` s=<16 hex>` to the shape signature; at the
defaults `n16_ell5_p2_r1` and every published file name are byte-unchanged.

Added by the matrix-engine programme and relevant to this feature:
`ThetaEqSolveValidation` (Phase 1, the multi-RHS `Theta_eq` solve),
`HeckeThetaBlockValidation` (Phase 3, `HeckeOperatorOnK` on the block model --
engine arm vs the forced legacy arm entrywise on `Theta_flat` / `Theta_eq` /
`A^{(q)}` / the whole `to_json()` text, `Theta_block(q)` against the flat shim
and its confinement to the tau pattern, and
`apply_Theta(q, e_b) == Theta_flat[:,b] == apply_route2(q, e_b)`),
`BarCocycleSolverValidation` (Phase 3, the absorbed route-B solver against a
frozen copy of the pre-absorption body), plus the substrate-level
`ConsumerMigrationValidation` / `MatrixRetirementValidation` static gates and
`GoldenRegressionValidation`, whose frozen hashes cover `Theta^{(q)}_flat`,
`A^{(q)}` and the normalised report JSON.

Phases 4-6 added seven more that touch this feature directly:
`EngineTapeConsumerValidation` (the tape path vs the direct `engine::blk` path
entrywise on `HeckeOperatorOnK`'s and `EquivariantCohomologyOnK`'s stored
objects; program-cache replay == fresh recording == forced fresh compile; a
changed tau pattern misses the cache), `EngineDeterminismValidation` (the golden
grid's cells under `ENGINE_THREADS` x `ENGINE_BATCH`: one digest across all
eight arms), `EngineCrossDeviceValidation` and `CudaBitIdentityValidation` (the
same objects under `ENGINE_DEVICE = cpu | gpu | hybrid`, and every device kernel
against its CPU twin by `memcmp` -- **a Hecke matrix never depends on the
device**), `EngineDeviceTrapValidation` (`slow`; the device cost table, gated on
identity only), and `CampaignSmokeValidation` /
`CampaignCertificateValidation` (the `--campaign` sweep driver: batched ==
sequential == golden, and C3-C6 run unconditionally in every `certify` mode).
`./hacs --list` is the authoritative list with tiers (**158 registered
validations** (59 `fast` / 85 `standard` / 15 `slow` = 159 at the slot-14 close, WP-17; 99 `Hecke*` names; slot-14 names in `Registry/Slot14/WP<n>.cpp`; earlier: 148 registered, 89 `Hecke*`, at the hecke-extensions slot-12 close, WP-9 — `docs/validation-catalogue.md`
has the roster; the slot-9 names are registered in `Registry/Slot9/WP<n>.cpp`, the slot-10 names in
`Registry/Slot10/WP<n>.cpp`, the slot-11 names in `Registry/Slot11/WP<n>.cpp`, the slot-12 names in
`Registry/Slot12/WP<n>.cpp`; historically 141 / 82 at the slot-11 close, 130 / 75 at the
slot-10 close, 125 / 71 at the slot-9 close, 114 / 65 at the slot-8 close, 81 / 40 after slot-3 WP-5, 80 / 39 at the slot-2
close, WP-13, 77 / 36 at the slot-1 close).

(Every `*Validation`
above with a `*SmokeValidation`/`*SelfCheckValidation` sibling is the
implementation-stage's own Definition-of-Done battery, while its sibling
without `Smoke`/`SelfCheck` in the name — e.g. `HeckeTessellationValidation`
vs. `HeckeTessellationSmokeValidation` — is the INDEPENDENT `*V`-stage
validation; see the file table above for the exact pairing.)

To exercise the pure-integer configuration (disables the certified
floating-point fast path of the sign predicates; both settings must produce
bit-identical Hecke matrices per [C, obl:fp]):

```bash
cmake -DHECKE_PURE_INTEGER=ON . && make && ./hacs HeckeFpFilterValidation
cmake -DHECKE_PURE_INTEGER=OFF . && make && ./hacs HeckeFpFilterValidation
```

### Slot 22 — Hecke-side additions (`docs/implementation/slot22/`; registration `Registry/Slot22/README.md`)

| File | What |
|---|---|
| `HeckeLevelNCurveValidation.{h,cpp}` (LN23; `standard`) | the `p = 2, 3` level-`N` route, the curve object |
| `HeckeGammaH.{h,cpp}`, `HeckeTameFamily.{h,cpp}`, `HeckeTameFamilyValidation.{h,cpp}` (WAKE-1; `standard`) | the `Γ_H(N)` level and the tame family of HPWWE |
| `HeckeGammaHTower.{h,cpp}`, `HeckeNebentypusWild.{h,cpp}` (P10) | the real Hida tower; the wild character models (validations in `HeckeExt/`) |
| `HeckeLevelPair.{h,cpp}` (LES-1, [C] `sec:nebles`) | `CongruenceLevel`, `QuotientOf` (normality on the coset tables), `RestrictionFibreMap` (`alpha`, trace `omega`), `ConjugationFibreMap` (`R_sigma`), `TwistedMonomialMatrix` (`eq:MHchi`): `def:levelpair`, `eq:levelres`, `def:normalpair` |
| `HeckeLevelNormalityValidation.{h,cpp}` (LES-1; `standard`, `Registry/Slot22/LES1.cpp`) | arms N1-N5, L, F, NEG: normality classification, fibre identities of `obl:levelconeexact` (i), negative controls |

### Slot 23 — Hecke-side additions (`docs/implementation/slot23/`; registration `Registry/Slot23/README.md`)

| File | What |
|---|---|
| `HeckeLevelCone.{h,cpp}` (LES-2, [C] `sec:nebles`, Route A) | `BuildLevelModel` (the character model `L^chi_A`), `RestrictionBlockTable` (`alpha_*` as a morphism), `LevelPairCone` (the level cone on the unequal-fibre `EquivariantMappingCone`; `def:levelcone`, `lem:levelconeexact`), `ApplyFibre`, `ApplyCoboundary` |
| `HeckeLevelConeValidation.{h,cpp}` (LES-2; `standard`, `Registry/Slot23/LES2.cpp`) | arms I, I′, II, III, III′, R3, IV, H, VI, NEG: six-term sequence, strict `T_p` / same-level `U_p` squares, level-raising `U_p` identity (square on `H^1` recorded), `model=` tags, negative controls; (LES-3) the read-only hook `HECKE_LEVEL_CONE_CELLS` (listed cells, Route A = Route B) |
| `HeckeLevelInfRes.{h,cpp}` (LES-3, [C] `sec:nebles`, Route B) | `TreeCokernel` (the tree complex `M^S ⊕ M^U → M^{−I}` of `prop:bredon` (2) as one sparse elimination `M / (M^S + M^U + M^-)` over `Z/ell^K`, P10's `ChainCokernel`), `OnH1` / `Escapes` (fibre maps `alpha`, `omega`, `R_sigma` and Merel's operators on `H^1`), dense helpers (rank, Howell length, Jordan type), `RouteBSixTerm`, `FiniteGroupH1SL2` (oracle O3) |
| `HeckeLevelInfResValidation.{h,cpp}` (LES-3; `standard`, `Registry/Slot23/LES3.cpp`) | arms AB (Route A = Route B), Z/p (the `Γ_1`-type steps, Jordan / Hochschild–Serre), HIDA1 (the first Hida step), TOWER (`F_p` and `Z/p^2`), GAMMA (`Γ(ℓ)` vs O3), MEREL (squares on `H^1`), NEG |
| `HeckeThetaRssDietValidation.{h,cpp}` (PERF-P4a; `standard`, `Registry/Slot23/PERF.cpp`) | the `compute_block()` RSS diet (program 1's arena and ρ₂ released at their last readers, the chunked A-N3 residual `hecke.resid.chunk`, `W` released before the Θ_eq copy): (T) tape == legacy route on Θ_flat / Θ_eq / A^{(q)} / counters, (L) memory-lean == certified on A^{(q)} / Θ_block, (C) ≥ 2- and ≥ 3-chunk residual coverage; 16 arms, ≈ 21 s, ≈ 0.3 GB; line `P4A_RSS_DIET … verdict=OK` |
| `HeckeRho2StreamValidation.{h,cpp}` (P4R; `standard`, `Registry/Slot23/P4R.cpp`) | route 2 streamed key by key vs the materialised memo (`ScopedRho2Streaming(false)`): (X) captured route-2 blocks equal entrywise and equal to Θ_block, Θ_flat / A^{(q)} / O12(a) on == off, (V) counters non-vacuous, (N) negative control, (M) fused-target mutations throw O12(a) at exactly the mutated entry, both routes; 15 arms, 225 checks, ≈ 1 s, ≈ 0.13 GB; line `P4R_RHO2_STREAM … verdict=OK` |

## Conventions

* Every validation is a zero-argument `uint32_t SomethingValidation()`
  returning 0 on success (I9), registered by name in
  `ValidationRegistry::RegisterAllValidations()` and runnable standalone via
  `./hacs <Name>`.
* Two primes, two names: `ell` (coefficient characteristic) and `p_hecke`
  (Hecke prime); the legacy `p`/`prime`/`p_mod` identifier of the arithmetic
  layer is `ell` and is not renamed (I1).
* `HECKE_PURE_INTEGER` (CMake option, default `OFF`) disables the
  floating-point fast path of the certified sign predicates (S1); both
  settings must produce bit-identical Hecke matrices ([C, obl:fp]).
* **JSON export conventions (amendment A-N4; binding for every `to_json()`
  writer under `Hecke/`):** polynomial coefficient lists are always emitted
  **low → high degree** (constant term first); `residue_degree` and
  `residue_degree_lower_bound` are kept as distinct keys and never conflated.
  Matrices are emitted as arrays of **rows** (row-major), entries as canonical
  residues in `[0, ell)`. S7's `HeckeOperatorOnK::to_json()` additionally
  documents in-band: the flat cochain layout (simplex-major, fiber-minor,
  index `s*d_V + k`), the cohomology-matrix column convention
  `T_p[z_i] = sum_j A[j][i] [z_j]`, and the certificate bundle
  (`routes_bit_identical`, the runtime-assertion check counters, and the
  exact `Theta_eq` solve residual, identically `0`, per amendment A-N3).

## E5 cost — the cone's memory and certificate cost (hecke-extensions slot 4, WP-4)

*Appended by slot-4 WP-4. Reason: the slot-3 close recorded "`compute()`'s `verify()` dominates at
`n = 256`" as an open point without a measurement; this block records the measurement, the three
value-preserving reductions it motivated, and the measured verdict on [X] §6.2 E5 (d). Nothing
here changes a computed value: `A_c^{(q)}` is bit-identical, both validations' whole printed output
is byte-identical (modulo wall-clock fields) against a binary built from `9ae1d7a`, and all 403
tamper-arm failing-item-set lines are identical by direct diff.*

**Where the cost actually is.** Reference box, `Bench`, `ell = 134217757`, `p_hecke = 2`, one quiet
interleaved round against the same-round `--bench hecke n 134217757 2`:

* the **cone** (`EquivariantMappingCone`, `HECKE_CONE_LARGE=n`) spends about **half** its wall in
  `ConeCertificate::verify()` (at `n = 256`: 4.34 s build + 4.31 s verify) — `verify()` does not
  "dominate"; and its **peak RSS is set by `compute()`'s `Z / B / H / P` elimination loop, not by
  `verify()`** (a VmHWM trace puts the high-water mark inside that loop, with `verify()` running
  ~20 % *below* it). Reducing the certificate therefore cannot move E5 (d), which is an RSS gate.
* the **cone action** (`HeckeConeAction`, `HECKE_CONE_ACTION_LARGE=n`) spends only ~9 % of its wall
  in `ConeActionCertificate::verify()`. **88 % of `compute()` is the standard-basis sweep** that
  asserts `apply_Phi(q, e_b) == Phi_flat[:, b] == apply_Phi_direct(q, e_b)` — which is in
  `compute()` but *not* in the certificate, so slot 3's open point 2 named the wrong object.

**The three reductions (all verdict- and value-preserving; none of them samples, relaxes or moves a
clause — every clause of both certificates still runs unconditionally inside `compute()`).**

1. `columns_in_span` (the C2 / C3 / C5 span inclusions of `ConeCertificate`) is the rank identity
   `rank [B | Y] = rank B`, read off **one** RREF of the augmented matrix by testing that no pivot
   index lands in the `Y` block. That is the same test on the same elimination that
   `engine::elim::solve_multi_rhs` performs before back-substituting; WP-4 simply stops before the
   back-substitution, which only built an `ny x nb` coefficient matrix the caller discarded. A
   throwaway shadow arm ran both routes on every call of the whole `HeckeConeValidation` /
   `HeckeConeActionValidation` grid: **1408 calls, 2 false verdicts (so the negative branch is
   exercised), 0 disagreements.**
2. In `EquivariantMappingCone::compute()`'s `Z / B / H / P` loop the gathered `C_eq` multi-vector,
   `D C_eq`, its RREF copy and the kernel basis are released **at their last reader** instead of at
   the end of the iteration (pure scoping plus moving the `B_{q+1}` pivot-column extraction one
   step earlier; the arithmetic, its operands and its order are untouched, and the `BlockPool`
   interning order is unchanged). The same narrowing is applied to `verify()`'s per-degree
   scratch views.
3. The action's standard-basis sweep evaluates the **stored-matrix** side of the equality one
   column block of the identity at a time (`kColumnBlock = 64`): the left-hand sides of 64
   consecutive basis vectors are the 64 columns of `Phi^q I_block`, one product instead of 64.
   Every basis vector is still compared individually, and the failure message still names the first
   offending one. The matrix-free side stays strictly per vector — it is matrix-free by definition
   — but is handed a shared `RhoMemo`, a memo of the pure function
   `gamma |-> rho_n(delta_j gamma mod ell)` that `apply_h_direct` previously rebuilt on every call.
   The public one-argument `apply_h_direct` / `apply_Phi_direct` still build a fresh local table, so
   every external caller sees exactly the pre-WP-4 behaviour.

**Measured effect** (Bench, interleaved with a `9ae1d7a` binary on a quiet box, median of two
passes):

| n | `hecke n` RSS | cone VmHWM before → after | cone wall | action VmHWM before → after | action wall before → after |
|---|---|---|---|---|---|
| 64 | 27.2 MB | 49.1 → 46.1 MB | 0.199 → 0.194 s | 95.2 → 95.1 MB | 1.99 → 1.29 s |
| 128 | 74.0 MB | 160.5 → 153.8 MB | 1.27 → 1.23 s | 335.5 → 334.8 MB | 15.46 → 9.99 s |
| 160 | 108.8 MB | 274.5 → 250.2 MB | 2.35 → 2.32 s | 515.7 → 513.7 MB | 30.86 → 18.86 s |
| 200 | 164.5 MB | 395.6 → 358.8 MB | 4.39 → 4.24 s | 786.1 → 783.3 MB | 57.90 → 36.13 s |
| 256 | 262.0 MB | 620.0 → 573.2 MB | 8.89 → 8.65 s | 1255.4 → 1243.0 MB | 122.4 → 75.4 s |

`HeckeConeValidation` 1.9 s → 1.7 s and `HeckeConeActionValidation` 11.9 s → 10.0 s in `--all`.

**The verdict on [X] §6.2 E5 (d) ("RSS at `n = 256` within `1.5x` of `hecke 256`"): NOT MET — and,
as written, NOT DECIDABLE.** (d) pins neither the denominator's measurement protocol nor which E5
object the numerator is, and the two candidate denominators differ by 1.42x:

| denominator at `n = 256` | value | cone ratio before → after | cone + action after |
|---|---|---|---|
| `--bench hecke 256 134217757 2` (one run; the like-for-like analogue of one cone build) | 262.0 MB | 2.37x → **2.19x** | 4.74x |
| the same with `--repeat 5` (the published protocol; from the arithmetic, the one slot 3's "1.7x" used) | 365-421 MB, median 372.7 | 1.66x → **1.54x** | 3.34x |

so the same tree misses (d) by 46 % or by 4 % depending on a flag that changes no work — and the
`--repeat 5` denominator is itself reproducible only to +-10 % at `n = 256` and +-28 % at
`n = 160`, i.e. its own noise exceeds the 4 %. Under the like-for-like denominator (d) is missed at
**every** `n` measured (best: 1.69x at `n = 64`); under `--repeat 5` the cone reads under 1.5x at
`n = 64` and `n = 200` and over it at `n = 128`, `160` and `256`. With the cone action included —
also E5 — (d) is missed under both, by 3.3x-4.7x. Slot 5's E5 acceptance has to pin the protocol
and the object before it can judge (d); WP-4 does not change (d).

**Why the remaining cost is structural.** At `n = 256` the cone's floor — the bulk cohomology with
its intermediates retained, which the cone is required to read in every degree — is already
~0.85x `hecke 256` on its own, and the stored cone objects (`C_eq`, `Z`, `B` over the three cone
degrees) are ~0.5x more, before any transient. The remaining transient peak is inside
`engine::blk::factor_basis`'s `RREF([B | Z | I])` (whose identity block alone is
`flat_dim x flat_dim`) and inside the gathers that `BlockMat::multivector_view` performs because a
`from_flat` block object stores one pool block per row block. Both are `ModArithmetic/` concerns,
not cone concerns. The next lever is the one slot 3 named: **put the cone on the tape**
(`engine::tape`, the `run_cached` route the cohomology and Hecke consumers already use), which is
what would let the arena own those transients instead of the heap.

## E5 cost, part 2 — the cone on the tape, and (d) re-measured (hecke-extensions slot 5, WP-5)

*Appended by slot-5 WP-5. It supersedes the verdict of the slot-4 block above (which stands as the
historical record of the pre-tape tree) on two points of fact, both measured here: the criterion's
wording, and where the cone's high-water mark sits. Nothing here changes a computed value — both
validations' whole printed output is byte-identical (modulo the two wall-clock fields) against a
binary built from the slot-5 integration head `00d9df7`, both collector digests are unchanged
(`0x4ecd2e429ec3b49`, `0xedbff302058416f3`), and `GOLDEN_STRICT=1` is tier A 9446/0, tier B
4603/0.*

**The criterion.** [X] §6.2 E5 (d) was restated by the owner on 2026-09-22 (`[OD]` OI-1,
`docs/implementation/slot4_open_items_brief.md` §1.6): denominator
`--bench hecke 256 134217757 2 --repeat 1`, numerator **the cone object alone** (not cone +
action), bound **2.0x**. That is the protocol every figure below uses, and it is decidable — the
denominator reproduces to +-0.2 % where `--repeat 5` reproduced only to +-10 %.

**What changed.** `EquivariantMappingCone` records its per-degree linear algebra on
`engine::tape`, the route `EquivariantCohomologyOnK` already takes:

* the two block GEMMs of the `Z / B / H / P` loop (`DC = D_q C^q_eq` and `Z^q = C^q_eq Kr^T`) are
  recorded as `gemm_dense` programs and served from the program cache by shape signature
  (`EngineRuntime.h` `run_cached`);
* `H^q_c` / `P_c^q` go through **`engine::tape::factor_basis`**, the identity-augmentation-free
  panel driver, instead of `engine::blk::factor_basis`;
* `ConeCertificate`'s C2 / C3 / C5 span inclusions run the **same** elimination over **borrowed**
  pool windows of both operands (stage 1 of `tape::factor_basis` with `want_projector = false`),
  one 256-column panel of the right-hand side at a time, above a 4 MB size threshold below which
  the dense in-place route of slot-4 WP-4 stays cheaper.

**What the identity augmentation actually cost, measured** (not derived): instrumenting
`blk::factor_basis` at `n_sym = 256`, the `q = 1` call is `a = 3855, nr = 985, ng = 1028`, so the
augmented matrix is `3855 x 5868`, it has full row rank 3855, and the RREF it materialises is
`3855 x 5868 = 86.3 MiB` — of which the identity columns alone are `3855^2 x 4 B = 56.7 MiB`
(59.4 MB decimal, i.e. the ~59 MB the brief derived). The process VmHWM across that single `rref`
call rises **375.2 -> 575.2 MB**. `tape::factor_basis` eliminates `[Rl | G]` alone (`3855 x 2013`)
and rebuilds the projector from the rank profile; the same call now costs +67 MB.

**Where the high-water mark is now.** The slot-4 finding "`verify()` runs ~20 % *below* the cone's
VmHWM" **no longer holds**, because the mark moved: at `n = 256` `compute()`'s construction now
peaks at **450.2 MB (1.70x)** and `ConeCertificate::verify()` takes the cone to **511.4 MB
(1.94x)**. The certificate is now the binding item. (It still runs in full, unconditionally,
inside `compute()`; nothing was sampled, relaxed or moved.)

**Measured effect** (Bench, five interleaved rounds against a `00d9df7` binary, one process per
measurement, `--bench hecke n 134217757 2 --repeat 1` in the same round):

| n | `hecke n` RSS | cone VmHWM before -> after | before x | **after x** | cone wall before -> after |
|---|---|---|---|---|---|
| 64 | 26.9 MB | 47.7 -> 42.6 MB | 1.77x | **1.58x** | 0.200 -> 0.156 s |
| 128 | 75.1 MB | 155.7 -> 136.0 MB | 2.07x | **1.81x** | 1.326 -> 0.796 s |
| 160 | 109.1 MB | 251.2 -> 202.7 MB | 2.30x | **1.86x** | 2.736 -> 1.488 s |
| 200 | 162.9 MB | 360.2 -> 307.7 MB | 2.21x | **1.89x** | 4.631 -> 2.671 s |
| 256 | 262.6-275.5 MB | 572.8-574.5 -> 510.8-524.1 MB | 2.08-2.18x | **1.90-1.95x** | 10.3-11.4 -> 5.5-6.2 s |

**[X] §6.2 E5 acceptance (d), under the restated protocol, is MET: the cone object alone peaks at
510.8-524.1 MB against a same-round `--bench hecke 256 134217757 2 --repeat 1` of
262.6-275.5 MB, i.e. 1.90x-1.95x (five interleaved rounds, median 1.93x), against the 2.0x
bound.** It is met at every `n` on the grid, and the worst cross-pairing of the five rounds
(largest numerator with smallest denominator) is 1.996x. Reported and **not** gated, because the
restated numerator is the cone alone: cone **with** its Hecke action at `n = 256` is
1242.6 -> 1275.8 MB (4.73x -> 4.86x) — the action's own peak is far above the cone's and rose
slightly, since the tape's program cache and executor arenas are retained underneath it.

**The input floor, measured at `n = 256`** (risk R-1 of the package brief; slot 4 cross-checked it
only at `n = 160`). On slot-4 WP-4's own definition — the process VmHWM at the cone's first trace
mark, i.e. the bulk cohomology, the boundary pair and the cone's copies of `H` / `P` / `r_flat`,
before a single cone object is allocated — it is **91.2 MB at `n = 160`** (reproducing that report's
figure exactly, 0.84x) and **210.2 MB at `n = 256`, i.e. 0.80x**, so it grows slightly *slower*
than the denominator rather than at the ~0.85x that had been assumed. The bulk cohomology alone,
before the boundary pair, is 153.4 MB = 0.58x. Reducing it is a change to
`EquivariantCohomologyOnK`'s retention contract and is not E5's scope.

**What is still not on the tape**, with the number: the per-degree `RREF(D C^q_eq)` and the kernel
and image bases derived from it stay on `engine::elim` over a dense copy, because one elimination
serves both bases and splitting it would double the work. At `n = 256`, `q = 1` that step costs
+48 MB (`DC` 25.4 MB, its RREF copy 25.4 MB, the kernel basis 12.7 MB) and 0.35 s — well below the
450.2 MB construction peak, so it is not the binding item.

## E7a — the contragredient side (hecke-extensions slot 6, WP-1)

[X] §6.2 E7, first half; [C] `sec:contragredient` (`def:contragredient`, `prop:pairing`,
`lem:symdual`), owner decision OQ4. Everything is the base change of a `Z`-datum.

| File | What |
|---|---|
| `ExactInt.h/.cpp` (`HeckeCore`) | `Mat22::adj()` — `adj [[a,b],[c,d]] = [[d,-b],[-c,a]]`, overflow-checked, [C] `eq:adjrules`. The inline adjugates of `HeckeTauTable.cpp` / `BarCocycleH1.cpp` are deliberately **not** migrated (a golden-path refactor for no benefit; slot-6 plan §3.15). |
| `../LocalSystemUtils.h/.cpp` | `FibreRepresentation {SymN, ContragredientN}`, `ComputeContragredientPowerMatrixFM(A, n) = rho_n(adj A)^T` and the dispatcher `ComputeFibreMatrixFM(rep, A, n)` whose `SymN` arm **is** the `ComputeSymmetricPowerMatrixFM` call. |
| `../LocalSystemSL2Z.h/.cpp` | `GenerateSL2ZContragredientRelationsWithGroupData` — the Sym^n relation list (same rows, order, simplices, tags, `g_r`), only `M = rho_n^vee(g_r)`. |
| `HeckeOperatorOnK.h/.cpp` | **The one golden-path touch**: an optional constructor argument `FibreRepresentation rep = SymN`. For `SymN` every block is read from the same tables and every `rho_of_exact` / route-2 block is the same `ComputeSymmetricPowerMatrixFM` call, and no program-cache key changes — byte-identical by construction (and by the golden gate). For `ContragredientN` the fused route's `omega_i = rho^vee(s_i^{-1})` and one `rho^vee((g gamma) mod ell)` per tau rho-pool slot are rebuilt from the exact coefficient-free coset / tau data, the program-cache keys carry a representation tag, and the constructor refuses a local system whose 8 seed relations are not `rho^vee(g_r)`. |
| `RhoMemo.h` | `RhoMemoT::Identity` gains `representation` (0 = `rho_n`, 1 = `rho_n^vee`), so a memo shared between the two representations is refused structurally, even at a reused owner address. |
| `HeckeContragredient.h/.cpp` | `PhiNMatrix` (antidiagonal, `Phi_n(u_t) = (-1)^t t!(n-t)! u_{n-t}^vee`), `PhiNIsIsomorphism` (`n < ell`), `PhiNFiberBlockTable` (the S6 general morphism constructor), `PairingGram` (= `I`), `EvaluationPairing`, `HeckeIntertwines(A_D, P, A_M, factor)`, `ContragredientCell` (M_n and D_n on the 7-vertex model, both cohomologies, `P = H^*(Phi_n)`, and `hecke_pair(p)` building both operators on one coset / tau table), and `ContragredientGoldenObjects` — tier A `dim_H_Dn^q`, `charpoly_A_Dn(T_p)^q`, hashed as the harness hashes, **defined and never called from `Golden/`** (slot-6 owner ruling 3). |
| `HeckePairing.h/.cpp` | (slot 6, WP-3 / E7b) the cohomological pairings of [C, cl:pd](1): `BuildModelFundamentalChain` (the relative fundamental 2-chain `[K]`, signs found combinatorially from the degree-1 seed relations, `d[K]` = −(horocycle)), `CupEvaluationForm` (Alexander–Whitney cup product with the fibre Gram of [C, prop:pairing](2)), `RestrictionFlat`, `PoincareDualityPairing` (the cone-level pairing `H^t_c(M_n) × H^{2−t}(D_n)`, `P_t((c, c_bd), y) = <c∪y, [K]> + <c_bd∪ry, d[K]>`, and the circle pairing `H^s(∂; M_n) × H^{1−s}(∂; D_n)`, with the D_n circle model), `PairingCertificate` (descent on the stored Z/B bases, Grams recomputed, record honesty — perfectness is RECORDED, never an item), `PairingAdjoint` (`A_c^T G = G A(D_n)`), `PairingGoldenObjects` (tier A `pairing_perfect^t`, `pd_adjoint^t`, tier B `pairing_matrix^t`, **defined, never emitted**). Reads the cone read-only. |
| `HeckeDualityValidation.h/.cpp` | `HeckeDualityValidation` (`standard`, ~6 s) — arms A1–A9 (WP-1) and A10–A15 (WP-3: descent and a second evaluation route, perfectness, **the PD gate for `X_3`** = adjointness for `p ∈ {2,3,5,7}` with `p^n`-twisted negative controls, socle ↔ Loewy = `HeckeIdentityCertificate` item 7 with a tamper arm, the ι-pairing sign RECORDED), see the header. |

**The exponent is 0, from [C] `lem:symdual`.** `Theta_g` carries one fibre matrix per term
(`n_k rho(s_i^{-1}) rho(g gamma_k) = n_k rho(delta_i gamma_k)`), the `D_n` operator carries
`rho^vee` of the same integral matrices, and `Phi_n rho_n(h) = rho^vee(h) Phi_n` for every
`h in Mat_2(Z)` with no determinant factor (`rho^vee(h) = rho_n(adj h)^T = det(h)^n rho_n(h^{-1})^T`,
so [C]'s `D_n` is the abstract dual twisted by `det^n`). Hence `Theta^D (1 (x) Phi) = (1 (x) Phi) Theta^M`
on cochains and `A(D_n)(T_p)^q P = P A(M_n)(T_p)^q` on cohomology, for **every** `n` (at `n >= ell`
`Phi_n (x) F_ell` is not an isomorphism and `P` is recorded with its rank). The `p^{+n}` / `p^{-n}`
readings are negative controls asserted to fail on every non-vacuous `(cell, p, q)`. [E] `cl:glover`'s
convention hazard concerns the abstract contragredient and is identity 6's (WP-9), not this package's.

### hecke-extensions slot 6, WP-4 — E5 cross-route 1: the strict-pair `τ^s`

The first sentence of `[C, obl:coneexact]`(vii), i.e. the strict-pair route of
`[C, prop:coneroutes]`(1) (owner decision OP3). Additive; the three shipped `TauVertexPolicy`
values, the default and every shipped table are untouched (`tau_table_p2.txt` byte-identical,
golden unchanged).

| File | What |
|---|---|
| `HeckeTauTable.{h,cpp}` | `TauVertexPolicy::BoundaryStrict` and the constructor taking a `HeckeBoundaryTauTable` (the plain constructor refuses the value). `[C, prop:strictpair]`: step (B) pre-assigns every boundary slot `j τ_∂(s_i σ)`; (V)/(E) as the default policy; (F) the unique fill **over `Z` only** — int64, then the widened `__int128` tier after a `HeckeOverflow` (`[OD]` OP2, through `escalate_on_overflow`), never mod `ℓ`. Its own `certify()` run: clauses (a)–(f) plus the strictness clause (g) (codes 70–72). Record fields `solve_regime()`, `num_fills_int64()` / `num_fills_widened()` / `num_fills_mod_ell()`, `num_strict_boundary_slots()`. |
| `HeckeIntegerSolve.{h,cpp}` | `solve_integer_system_wide()`: the same elimination on `__int128` working entries. The int64 solver is unchanged. |
| `HeckeConeAction.{h,cpp}` | `HeckeConeStrictPairRoute`: from a computed normative action it builds `τ^s`, `η^s` on `(τ^s, τ_∂)` (asserted **identically zero**), `Θ^s` and `Φ^s = (Θ^s, Θ^∂)` on the same cone, and compares `A_c^{(q)}` entrywise in every degree. It **aborts** (throws) on a disagreement or a build failure at `6 ∈ R^×`, and **records** both at `ℓ ∈ {2, 3}`. |
| `HeckeOperatorOnK.h` | read-only accessors `cohomology_of_L()`, `cosets()`, `tau()`. |
| `HeckeConeActionValidation` | arm **A14** on every one of the 212 operators, with its own check log and its own `strict pair` / `CONE_STRICT_PAIR` lines, so every pre-existing printed line is unchanged (`docs/matrix-engine/tools/cone_output_diff.sh`, created by this package). |

Measured on the grid: `A_c^s == A_c` on 212 / 212 operators (636 degrees), asserted on 138 and
recorded on 74 (`ℓ ∈ {2, 3}`) with **0 recorded disagreements** and 0 build failures; `η^s ≡ 0` and
`r Θ^s = Θ^∂ r` (flat) on all 212; `τ^s ≠ τ` on all 212. Solve regime `Z`: 4650 int64 fills, 0
widened, 0 mod-`ℓ`. The widened tier is exercised by forcing it (`ScopedExactTier`) on three cells: 21
fills, and the table is slot-for-slot identical to the int64 one.

### hecke-extensions slot 6, WP-2 — E2 (a): `HeckePresentation` (OQ11), without its settings key

Nothing under `Hecke/` changes. The unit and its validation live in `HeckeExt/` (see
`HeckeExt/README.md`): `HeckeExt/HeckePresentation.{h,cpp}` — generators and minimal relations of a
certified-local block over `T_m`, [X] §2.2 Stage 5 — and `HeckePresentationValidation` (`fast`,
~1.9 s, 995 checks). It is reached through `ExtDriverOptions::with_presentation` (default off) and
**not** through a `MathSettings` key; the key, the record key and the arm "record hash unchanged by
turning it on" are slot 7's changed-line re-baseline (slot-6 plan §2.3 item 2, §4.4).

### hecke-extensions slot 6, WP-5 — E5 cross-route 2 (route 3), and the `A_c(T_p)^q` relabel

The second sentence of `[C, obl:coneexact]`(vii), i.e. route 3 of `[C, prop:coneroutes]`(2) (owner
decision OP3; no hypothesis on `ℓ`). Additive; no default path, printed line or golden value moves.

| File | What |
|---|---|
| `../EquivariantCohomologyOnK.{h,cpp}` | The **relative sub-complex constructor** `EquivariantCohomologyOnK(lssc, vanishing_simplices)`: the listed `K`-simplices of each degree are forced to zero by one relation `f(σ) = 0·f(σ)` each, appended to a relation list only this object reads, so both arms (block and legacy) compute `C/Z/B/H/P` of `ker r = Hom_{Z[Γ]}(C(X, X_∂), V)` with their unchanged code. On every other object `effective_relations()` is the local system's own list (the same reference). |
| `HeckeConeAction.{h,cpp}` | `HeckeConeRelativeRoute`: from a computed normative action it builds `ext_0` (from the stored `r_flat`), `E = ext_0 ∘ h`, `Θ' = Θ + dE + Ed` (`[C, eq:route3]`), `ker r`, `A'` on `H^•(ker r)` and the correspondence `M: H^q(ker r) → H^q_c`, `c ↦ (c, 0)`, and compares `M A' M^{-1}` with `A_c^{(q)}` entrywise in every degree. It **aborts at every `ℓ`** on a disagreement and on each of its clauses R3.1–R3.6 (`r ext_0 = I` and `ext_0` equivariant; `dim C_rel`; `Θ'` a cochain map with `r Θ' = Θ^∂ r` on the nose and the same bulk `A^{(q)}`; `Θ'` preserves `ker r`; `M` invertible; the comparison). **The relabel**: `golden_objects()` labels the raw `A_c^q` tier `'B'` ([X] §1.6 / OP7). |
| `../Golden/GoldenHarness.cpp` | The `ExtTierFor` override (which mapped `A_c^…` from the unit's `'A'` to `'B'`) is **removed**; its comment, which attributed the grid digest `0xedbff302058416f3` to the cone action, was false. The published tier is unchanged, so `GoldenReference.inc` is unchanged. |
| `HeckeConeActionValidation` | arm **A15** on every one of the 212 operators, with its own check log and its own `route 3 (...)` / `CONE_ROUTE3` lines; the relative sub-complex's legacy arm is compared with its block arm on three cells. |

Measured on the grid: `M A' M^{-1} == A_c` on 212 / 212 operators (636 degrees), asserted at every
`ℓ`; `Σ dim H^q(ker r) = [0, 415, 156]` (`= Σ dim H^q_c`); `Θ' ≠ Θ` on `C_eq` on all 212; the
uncorrected `Θ` moves an `H(ker r)` cocycle off `ker r` on 131 / 212. `HeckeConeAction::golden_digest()`
(printed nowhere) moved with the relabel, e.g. at `(10, 691)`, `p = 2`, `ν = 1`: `0xb298009e1c80c9fb →
0xba258de7a334d5e0`. Nothing is re-pinned.

### Slot 7, WP-1 — E7 acceptance (b): `cl:pd` coverage on the E2 case list; the E7 golden objects emitted

[X] §6.2 E7 acceptance (b) on the E2 case list at `ℓ ≥ 5` (the 31 ext-subgrid cells of [X] §1.6
and the 23 M1 cells), beyond the 20 E7b cells. `[C, cl:pd]` stays OPEN: these are oracle
checks, and a failure would be a finding. Additive; no default path or printed line of an existing
arm moves.

| File | What |
|---|---|
| `HeckeDualityValidation.{h,cpp}` | arm **A16** (coverage): the certificate, `dim H^t_c(M_n) = dim H^{2−t}(D_n)`, perfectness of the cone and circle pairings, and the PD gate for `p ∈ {2,3,5,7}` with the `p^n`-twisted negative controls, on the **34** uncovered cells with `n ≤ 34` (29 subgrid + `(8,7)`, `(12,11)`, `(34,5)`, `(26,7)`, `(32,131)`): 102/102 cone and 68/68 circle pairs perfect, 272 gate pairs failing 0, 106/106 controls rejected. `HeckeDualityValidation` goes from ~6 s to ~21 s. |
| `HeckeDualityValidation.{h,cpp}` | **`HeckeDualityCoverageSlowValidation`** (`slow`, ~28 s): A16 on `(58,7)`, `(58,5)`, `(82,7)`. `(160,691)` and `(198,691)` (129 s + 241 s, indicative) are **left out for cost** and run only under `HECKE_DUALITY_COVERAGE_LARGE=1` (read-only). They passed there: 40 gate pairs failing 0, 15/15 cone pairs perfect. |
| `../Golden/GoldenHarness.cpp` | `RunE7Cell32`: the E7 collectors (`ContragredientGoldenObjects`, `PairingGoldenObjects`) emitted on the 31 subgrid cells, `p ∈ {2,3,5}`, point suffix ` e7`: **651 added, 0 changed, 0 removed**. See `docs/matrix-engine/rebaseline-2026-09-23-e7-addonly.md`. |

### Slot 8, WP-3 — E8 item 3: the pipeline over `Z/ell^k`, the `ρ_n` repair, `coefficient_ring` with its refusals

[X] §6.2 E8 item 3 under `ENGINE_API.md` §13 and [E] v1.9.0-pre.1 `sec:ring`. **The `F_ell` path is
untouched**: the composite route is selected by the ring in force (`coefficient_ring`, read once at
construction) and runs only on a local system built over `Z/ell^k`; golden, both digests and the
`ρ_n` prime route are byte-identical.

| File | What |
|---|---|
| `HeckeSettings.{h,cpp}` | `coefficient_ring` registered at the registry's **tail** (11 keys; four edits), grammar `F_ell \| Z/ell^<k>` (`2 ≤ k ≤ 63`); `Z` and `Z/<m>` refused **inside `set()`** by `NotImplementedInThisBuild`; `ScopedCoefficientRing`; the refusal guard `RequireFieldCoefficientRing(who, input_k)` / `RefuseCoefficientRing`. |
| `CoefficientRing.{h,cpp}` (new) | `CoefficientRingContext` (the honouring context; refuses `ell ∈ {2,3}`, `ell^k ≥ 2^64`, `ell^k` beyond `Elt`), `GenerateSL2ZRelationsRing` (the model over `Z/ell^k`, the same rows as the field list), `RhoOfExactRing`, `ExactKthRoot`. |
| `../../src/ModArithmetic/FlatMatrix.{h,cpp}` | E8-P5: `ComputeSymmetricPowerMatrixRing` (unit fast paths, the `diag(ell,1)` split, the first-row-zero case; Pascal by additions, no `mod_nCr`) and the division-free oracle `ComputeSymmetricPowerMatrixRingOracle`. `ComputeSymmetricPowerMatrixFM` is not touched. |
| `../../src/ModArithmetic/BlockMat.h`, `BlockKernels.cpp` | §13.7's decision: `BlockMat` / `BlockBuilder` **`ModCtx` overloads** (`zeros`, `from_flat`, `from_dense`, the builder); `blk::gemm/add/scale` results carry `A.ctx()`; the `Elt p` overloads still declare a field. |
| `../../src/ModArithmetic/DenseElim.{h,cpp}` | `cyclic_decomposition` / `cyclic_coordinates`: [E] `sec:ring` (c)'s M3 basis rule with its coordinates. |
| `../EquivariantCohomologyOnK.{h,cpp}` | the **module route** (`C` = the Howell kernel of `A_Cq`, asserted free; `X_q` = `d^q` in `C` coordinates; `Z = ker X_q`; `H = Z/B` by the M3 rule; `r_K(d^q)`, `ladic_type_certified`, `ladic_type^q`, the `ladic_level` state); every field accessor throws a named `logic_error` over `Z/ell^k`. |
| `HeckeOperatorOnK.{h,cpp}` | the module route: `Θ` fused **and** route 2 from the exact coset / tau data through the composite `ρ_n` (O12(a) over the ring; `U_ell` by the split), (i)/(ii)/(iii), `A^{(q)}` on the cyclic summands, `free_charpoly_H1()` (E8-P4, certified levels only). |
| the closure's members | one constructor-entry guard each; the closure table is in `docs/implementation/slot8/WP-3_report.md`. |
| `HeckeReductionPipelineValidation.{h,cpp}` | `HeckeReductionValidation`'s arms (K), (B), (P), (C), (T), (U), (X). |

### Slot 8, WP-4 — E8 item 4a: the reductions, the Bockstein, `higher_torsion_count`; the E8 golden objects (candidate)

[X] §6.2 E8 item 4 (bulk half) with finding F3, [E] v1.9.0-pre.1 `sec:ring` (d)–(e). Every map is
a map between **stored bases** of objects WP-3's module route and the `F_ell` route already
computed; the certificates are **map equalities** (`A(T_p) ∘ red = red ∘ A(T_p)`,
`β ∘ A(T_p) = A(T_p) ∘ β`), never "the matrices reduce to each other" (false: the Tor cokernel).

| File | What |
|---|---|
| `HeckeReduction.{h,cpp}` (new) | `ReduceHToField` (`red_H^q`: `P_q (b_j mod ell)`, the (c5) matrix to the D5b basis), `ReduceHBetweenLevels` (`Z/ell^{k+1} → Z/ell^k` via `H_class_coordinates`), `BocksteinH` (lift a D5b class into `C^q_eq(Z/ell²)` through the free Howell basis, `X_q c = ell·w`, `β(x) = [w]`), `RankOverField`, `BocksteinCounts` (`bockstein_rank^0`, `higher_torsion_count := dim H^0(F_ell) − bockstein_rank^0`), and the four certificates. |
| `HeckeReductionCoverageSlowValidation.{h,cpp}` (new) | `HeckeReductionBulkArms` — called from `HeckeReductionValidation` (`standard`): (R1)–(R5), (B1)–(B3) at `(10,691)` (`K = 2`) and `(6,5)`, `(16,5)`, `(30,5)`, `(60,5)` (`K = 2, 3`), `p ∈ {2,3,5}` (`U_5` validated, not frozen); the pre-stated `higher_torsion_count` 0/0/0/1/1 and `#{e_i ≥ 2}` wherever certified; the **D6 arm** at `(10,691)` (`charpoly_A(T_2)^1` over `Z/691²` = `(x − 2049)(x + 24)²`, eigenlines recorded). **`HeckeReductionCoverageSlowValidation`** (`slow`): (R1)–(R4), (B1)–(B3) at `Z/ell²` on the E2 case list at `ell ≥ 5` (52 cells); `(160,691)`, `(198,691)` only under the read-only `HECKE_REDUCTION_COVERAGE_LARGE=1`. |
| `../Golden/GoldenHarness.cpp` | `kGridE8` + `RunE8Point`, looped from `ComputeAll()` **only**, after the S6 loop: the seven new points `E8 ring=Z/<ell>^<k> n=<n> ell=<ell>`, 124 entries (103 A, 21 B), the ring installed around the E8 constructions only. See `docs/matrix-engine/rebaseline-2026-09-24-e8-addonly.md`. |

### Slot 9, WP-4 — E8 item 4b: the boundary and the cone over `Z/ell^k` (route 1)

[X] §6.2 E8 item 4's boundary-and-cone clause, [E] v1.10.0-pre.1 `sec:ring` (f), slot-9 decision 3
(A) with (iii) (A): **route 1 only** — the strict-pair and relative cross-routes and the relative
sub-complex constructor keep refusing; `obl:coneexact`(vii)'s composite half is open. **The `F_ell`
cone is untouched** (`cone_output_diff.sh` against `bd6d1bb`): every honouring member is selected by
its inputs' ring (the ring in force at construction), and the tape route stays the `F_ell` path.

| File | What |
|---|---|
| `../EquivariantMappingCone.h`, `../EquivariantMappingConeModule.cpp` (new) | `ModuleConeT<Elt>` — the cone's **dense module route** (§13.6 only; u32 / U64_Montgomery / U64_Wide): `Cc^q_eq` = the two Howell bases concatenated (asserted to be the block sum's Howell form), `X_q` = `D_c` in coordinates, `H^q_c` by (c2)–(c4), the eight maps of `eq:coneles` as (c5) matrices, `ConeModuleCertificate` (items 1, 10, 11, 12 in lengths; **item 14 appended**: `obl:coneexact`(iv)'s non-field half — composites by solves, each `ker` generator exhibited as an image by `solve_multi_rhs_module`, the witness re-derived); `RestrictionFlatRing`; `EquivariantMappingCone`'s composite branch (`module_cone()`; field accessors throw a named `logic_error`). |
| `../EquivariantCochainComplexMorphismOnK.{h,cpp}` | the constant-fibre constructor honours a ring shared by domain and codomain (mismatch = named `logic_error`); `pullback_matrix_module`, `get_induced_H_module_maps` ((c5)); the `FiberBlockTable` constructor keeps refusing. |
| `SL2ZBoundaryPair.{h,cpp}` | honours a bulk over the ring in force: the circle model over `Z/ell^k` (`GenerateSL2ZBoundaryRelationsRing`, `CoefficientRing.{h,cpp}`) and the composite restriction; refuses `ell ∈ {2,3}` (default `SmallEllPolicy`). |
| `HeckeBoundaryOperator.{h,cpp}` | the module route: `Θ^∂` both bracketings through the composite `ρ_n`, (i)–(iv), `T_p^∂` on the cyclic summands (`Theta_module`, `H_matrix_module`); routes A/B stay field-only. |
| `HeckeConeAction.{h,cpp}` | `ModuleConeActionT<Elt>` (route 1): `h` both bracketings, `Φ`, `A_c` on the cyclic summands, items 2, 4–9 asserted over the ring; `HeckeConeAction`'s composite branch (`module_action()`); the two cross-routes refuse a composite normative action. |
| `HeckeReduction.{h,cpp}` | `ReduceConeHcToField` (`cone_red_Hc^q`), `ReduceConeHcBetweenLevels`, `BocksteinConeHc` (`β^1_c`, validated, not an object), `ReductionIntertwines`; the boundary reuses the bulk functions on its `EquivariantCohomologyOnK`. |
| `HeckeReductionConeValidation.{h,cpp}` (new) | `HeckeReductionConeArms` (from `HeckeReductionValidation`): (4b-B/C/H/R/K) at the E8 points but `(60, 5)` plus the U64_Montgomery and U64_Wide cells, and (X4b), the 21 closure rows moved or added by 4b; `HeckeReductionConeCoverageArms` (from the `slow` sibling): the E8 point `(60, 5)` at `K = 2, 3`, `p ∈ {2, 3}` with every arm (moved there by slot-9 WP-4t, unchanged in content), then the E2 case list at `ell ≥ 5`, `K = 2`, `p = 2`. WP-4t also dropped the rows implied by a constructor throw or by the cone certificate (`docs/implementation/slot9/WP-4t_report.md`). |

### Slot 10, WP-3 — `obl:coneexact`(vii) over `Z/ell^k`: the cross-routes over the ring

Slot-10 decision 3 (A), plan M-3: the strict-pair route and route 3 each give `A_c` **entrywise
equal** to route 1's over `Z/ell^k` at `ell ≥ 5`, `ell^k < 2^32` — at every E8 point. The two
cross-routes stay `uint32_t` classes (not templated): above `2^32` they refuse by the ring context's
named `uint32_t` regime refusal, and only route 1 (`ModuleConeActionT<uint64_t>`) runs there. **The
`F_ell` cone is untouched** (`cone_output_diff.sh` against `bd6d1bb`: 0 hunks, both digests SAME).

| File | What |
|---|---|
| `HeckeConeAction.{h,cpp}` | `HeckeConeStrictPairRoute` / `HeckeConeRelativeRoute` honour a composite normative action under its ring (`CrossRouteRingGuard`; `RequireRingRegime`), every other combination keeps slot 8's refusal. Strict pair over the ring: `τ^s`, `η^s` integral as over `F_ell`, `Θ^s` by `HeckeOperatorOnK`'s module route, `Φ^s` = route 1 on `(τ^s, τ_∂, η^s = 0)`, `A_c` compared entrywise; a validation seam (a supplied strict table, re-certified). Route 3 over the ring: `ext_0` from `ModuleConeT::r_flat`, `E = ext_0 h`, `Θ' = Θ + dE + Ed`, `ker r` by the relative constructor's module route, R3.1–R3.6 with the explicit inverse `N: [(c, c_∂)] ↦ [c + d ext_0 c_∂]` (`M N = I`, `N M = I` in (c5) form — no inverse over the ring), `M A' N == A_c`; `*_module` accessors, field accessors throw a named `logic_error`. |
| `../EquivariantCohomologyOnK.{h,cpp}` | the relative sub-complex constructor honours `Z/ell^k` at `ell ≥ 5` (the module route reads the zero relations; `C^q_rel` is free), refuses by name at `ell ∈ {2, 3}`. |
| `HeckeConeCrossRouteRingValidation.{h,cpp}` (new; `standard`, `Registry/Slot10/WP3.cpp`) | (R) the seven E8 points × `T_p` (`p ∈ {2,3}`, and `5` at `(10, 691)`) × every degree: routes 1, 2, 3 equal; (N) a perturbed `τ^s` is refused by the route (its certificate) and, past the certificate, by route 1's item 4; (W) `(10, 134217757)` over `Z/ell²`: route 1 runs, both cross-routes refuse by the `uint32_t` regime; (X) the three flipped members both ways. |
| `HeckeReductionConeValidation.cpp`, `HeckeReductionPipelineValidation.cpp` | three closure rows changed in place (the two cross-routes and the relative constructor now honour); check counts unchanged. |

### Slot 11 — the Hecke-side additions (WP-5, WP-6, WP-11, WP-18c; `docs/implementation/slot11/`)

| File | What |
|---|---|
| `HeckeOperatorOnK.{h,cpp}` | accepts `FibreRepresentation::QuotientL` (WP-5): `d_V = ell + 1`, `n_sym` read from the tau table (the `Sym^n` tables), its own program-cache tag `kQuotientLKeyTag`; the `SymN` and `ContragredientN` paths byte-identical (golden 0 / 0, four digests, `bench_identity` 45 / 0) |
| `LadicCell.{h,cpp}` (new, WP-18c) | the digit pipeline route at `ell^k ≥ 2^64`, `ell ≥ 5`: the per-cell certified precision bound `K = k + e_max`, the cohomology over `Z/ell^K` in digit planes (`engine::ladic`), `Θ_g` both routes, `A^{(q)}`, the free charpoly to precision `k`, the ordinary idempotent at `p = ell` ([C] `def:ordinary`) |
| `HeckeLadicCellValidation.{h,cpp}` (new; `standard`, `Registry/Slot11/WP18c.cpp`) | 891 checks: oracle (i) 2 × 32 vs the shipped `u64` route; (ii) M-0's `(x − 2049)(x + 24)²` at `(10, 691)`, `k ∈ {7, 12, 20, 40}` (beyond `2^127`) and O5 at `691^7`; (iii) UCT; (iv) cross-width records byte-identical; (v) truncation; (vi) native `u128`; (P) / (P′) the bound at `e_max = 1, 2, 2` with cross-`K` and cross-route `χ(T_2)`; (H) / (H′) the idempotent at `(10, 691)` and `(30, 5)` |
| `CoefficientRing.{h,cpp}` (WP-12, F-2) | `IsPrimeU64` (deterministic Miller–Rabin); `ExactKthRoot` on it; `k < 2` defined (a UBSan find); `HeckeSettings.cpp`'s `coefficient_ring` parse cap `63 → 4096` (WP-18c) |
| `../Integral/Model/StrictLift.{h,cpp}` (WP-11, M-3) | the lifts for `diag(1, p²)`, `diag(1, p³)`, `pI`; `HeckeStrictLiftRelationsValidation` (`standard`, 3619): X7 in full on `H^q(C_I ⊗ R)`, `q ≤ 5` |
| `../HeckeExt/HeckeDicksonTheta*`, `../HeckeExt/HeckeCensusSplit*` (WP-5, WP-6, WP-7) | θ ([C] Part VII) and the census split — see `docs/architecture-detail.md` |

### Slot 12 — the certified ordinary projector, the towers, the forms side (WP-1 … WP-14; `docs/implementation/slot12/`)

At the close (WP-9): seven names added (148 = 57 `fast` / 78 `standard` / 13 `slow`), no golden event, the four
digests unchanged; [C] v1.12.0 (`sec:hida`, refereed and owner-signed 2026-09-26; `open:ordprecision`
answered, tag `PROVED`). Beyond the rows below:
`HeckeThetaCensusRingRefusalValidation` (WP-1, `fast`, F-1's named ring guards on the θ / census line),
`../HeckeExt/HeckeFormsHeckeAlgebra` (WP-7 / WP-12: the forms-side `T_m/ell` on `S_k` and `M_k`;
`HeckeFormsHeckeAlgebraValidation`, `standard`) and `HeckeBoundaryFinalityValidation` (WP-14, `standard`,
`Registry/Slot12/WP14.cpp`: Lemma B′ on the shipped boundary operators, the finality certificate (d1) evidence;
**landed by S12-FIN** (owner-approved 2026-09-26): `../HeckeExt/HeckeFinalityRoutes` — (d0) "dickson" from [C]
`prop:heckeH0`(1), (d1) "boundary" from [C] `prop:bdrytranscription` / `lem:bdrytorus` — in `HeckeExtDriver`, the
record `hecke_ext/v4`, [E] v1.13.0 `cert:dickson` / `cert:boundary`; the validation compares the driver's emitted
routes with its independent computation; `docs/matrix-engine/rebaseline-2026-09-26-hecke-ext-v4-finality.md`).

| File | What |
|---|---|
| `HeckeOrdinaryPart.{h,cpp}` (new, WP-4) | [C] v1.12.0 `sec:hida`: `TorsionExponentPrepass` (`prop:torsionprepass`: `e_max`, the lengths `λ_e`, the torsion multiset from `J_n = [ρ_n(S) − 1; ρ_n(T) − 1]` alone, every prime, `n ≥ 1`), `CertifiedWorkingPrecision` (`cor:ordprecision`(2): `K = e' + e_max`, `ell ≥ 5` only), `OrdinaryExponent` (`lem:ordexponent`'s `N_ord(g, K)` as factors), `RingMat` / `OrdinaryProjector` (`U^N` on `⊕ Z/ell^{a_i}`) |
| `LadicCell.{h,cpp}` | opt-in `Request::precision_source = Prepass` (default `KMode` unchanged, byte for byte); `CertifiedWordKMode` exposes step 1 |
| `HeckeOrdinaryPrecisionValidation.{h,cpp}` (new; `standard`, `Registry/Slot12/WP4.cpp`) | 829 checks: (Hi1), (Hi2) of `rmk:hidacheckable` against WP-18c, with negative controls |
| `../HeckeExt/HeckeHidaTower.{h,cpp}` (new, WP-5) | the first Hida weight towers ([C] `def:hidatower`) on `HeckeOrdinaryPart`: per cell `H_K`, `K = 2 + e_max`, by the module route; the ordinary part read on `H_{e'}` and on the free block (`cor:ordprecision`(1)–(4)), `χ(T_p)` on it (Nakayama basis, `ell`-adic coordinate solve, Berkowitz); `HeckeHidaTowerValidation` (`standard`) / `HeckeHidaTowerSlowValidation` (`slow`), `Registry/Slot12/WP5.cpp`: (Hi2)–(Hi6), incl. `(10, 2411)` |
| `../HeckeExt/HeckeCensusCellList.{h,cpp}` (new, WP-6) | the census cell list (`<n> <ell> [<u_ell_budget_mb>]`, refusals by line number) and `CensusEdition1Grid()`: `HECKE_CENSUS_SPLIT_CELLS=<file>` (`HeckeCensusSplitValidation` arm (D)) and `HECKE_THETA_IMAGE_CELLS=<file>` (`HeckeDicksonThetaHeckeValidation` (P5)), read-only; census edition 2 staged, not published; `HeckeCensusCellListValidation` (`fast`), `Registry/Slot12/WP6.cpp` |

### Slot 13, WP-4 — the `U_ell` memory lever (default-on since slot-14 Q8; `docs/implementation/slot13/WP-4_report.md`)

| File | What |
|---|---|
| `HeckeLowRank.{h}` (new) | `hecke::lowrank`: `LowRankT` (`U V`), `factor_low_rank` (exact min-valuation elimination over `F_ell` / `Z/ell^k`; the term count `≤ min(k, n + 1)` — the rank claim — `ENGINE_CHECK`ed per block), `sum_of_low_rank` (one GEMM over stacked factors) |
| `HeckeOperatorOnK.{h,cpp}` | (**retired by S24 CL-R1**: `ScopedUEllLowRank` and the other slot-13/14 `Scoped*` are in `legacy/HeckeOperatorLeversLegacy.h`, ON builds only; `HECKE_U_ELL_LOW_RANK=0` refused in OFF builds; the bench flags are gone) `HeckeOperatorOptions::ScopedUEllLowRank` (thread-local, **`Auto` by default since slot-14 Q8** — in effect iff `ell ∣ N` and `SymN`; opt-out `HECKE_U_ELL_LOW_RANK=0` / `ScopedUEllLowRank(false)` / `--no-u-ell-low-rank`; not a setting; refused when forced on unless `ell ∣ N` and `SymN`): the fused blocks assembled on the host from the factored pool (no product temps), route 2 on factored `ρ_n((δ_i γ_k) mod ell)` compared block by block; the module route keeps factored memos. Default path byte-for-byte unchanged |
| `HeckeTauTable.{h,cpp}` | a table **constructed** under the lever with `ell ∣ N` keeps `rho_pool()` factored (`rho_pool_low_rank(slot)`, placeholders on which `RhoRefT::get()` throws); certification (f) compares every term with `U V` of its slot, recomputed from its own exact `γ_k` |
| `HeckeUEllLowRankValidation.{h,cpp}` (new; `standard`, `Registry/Slot13/WP4.cpp`) | 344 checks (230 + WP-14's arm (E), audit A1: the `!tau_factored` branch and the two loud refusals; + WP-16's arm (F): the lever on the legacy arm, in-process; + arm (G), `c0566e2`: E7's contragredient operator over a factored τ pool on the legacy arm at `ell = 5`): the lever's `A^{(q)}`, `Θ_flat`, `Θ_eq`, certificates equal to the default at `ell ∈ {5, 7, 11, 13, 691}` (+ `T_{ell²}`), the refusal on `T_2`, `Z/ell²` and `Z/5³` module route; `(128, 691)` peak RSS printed |

### Slot 13, WP-5 / WP-6 — the lattice fibre and `Γ_0(ell)` on the model (`docs/implementation/slot13/WP-5_report.md`, `WP-6_report.md`)

| File | What |
|---|---|
| `../LocalSystemUtils.{h,cpp}` | `FibreRepresentation::HidaLatticeH`: `ρ^H_n(A) = ρ_n(A^T)^T` (Hida's lattice `L_n`, [C] `rmk:stabconvention`(3), `def:hidalattice` variant (H)) |
| `CoefficientRing.{h,cpp}` | `RhoHidaOfExactRing`, `GenerateSL2ZRelationsRingHida` (the ring relation list with `ρ^H_n`, same rows / tags) |
| `HeckeOperatorOnK.{h,cpp}` | the **module route** honours `HidaLatticeH` (seed relations checked against `ρ^H_n`); the field route and the lever refuse it by name |
| `../EquivariantCohomologyOnK.cpp` | the module route's **Nakayama-basis fallback**: a free `C^q_eq` whose Howell form has a non-unit pivot (on `L_n`, never on `Sym^n`) takes the Howell rows independent mod `ell`, checked free by length — value-preserving on every object computable before ([C] v1.13.0 `def:hidalattice`'s amended sentence; audit A3) |
| `HeckeOrdinaryPart.{h,cpp}` | `PrepassRoute::{Auto, Native, Ladic}`: the pre-pass beyond `2^64` by `engine::ladic` (`W = uint64_t`); `ell ≥ 2^31` there refused by name |
| `HeckeLevelEll.{h,cpp}` (new, WP-6) | `hecke::level` — [C] v1.13.0 `sec:levelshapiro`: `CosetLevel` / `FullLevel` / `Gamma0Level` (the shipped coset table at `N = p^ν`), `CorrespondenceSpec`, `VerifyTransversal`, `BuildLevelCorrespondence` (`eq:levelcorr`, one routine for `U_ell^{(ell)}`, `dg_ell`, `T_p^{(ell)}`, `T_ell`, `dg_1`, `ω`; exact `Mat22` products, det checked), `BuildUellLevelEllFused` (`eq:Ulevellmodel`), `BlockCochainMap` (no `D_N × D_N` matrix; det-`ell` blocks factored), `InducedRelationsRing` (over `Z/ell^k`), `HFrame` / `InducedOnH` / `HMat`. Pipeline-free and value-free; general-`N` plug points |
| `HeckeHidaLevelEllValidation.{h,cpp}` (new; `standard`, `Registry/Slot13/WP6.cpp`) | 2906 checks, ≈ 42 s: `eq:Ulevellmodel` = `eq:levelcorr` entrywise; both against an independent group-cocycle route (Shapiro on `⟨S, T⟩`); `prop:oldspace`, `prop:levelorbit`(3), `lem:levelelltorsion`(1), `prop:stabilisation` mod `ell^{1,2,3}`, (Hi10) on 92 cells |

### Slot 14 — WP-6 / WP-10 / G0 / WP-2 (audit R2; `docs/implementation/slot14/`)

| File | What |
|---|---|
| `../HeckeExt/HeckeRingHeckeAlgebra.{h,cpp}`, `HeckeRingLoewyValidation.{h,cpp}` (WP-6; `standard`, `Registry/Slot14/WP6.cpp`) | the `Z/ell^K` Hecke-algebra route read mod `ell` ([E] v1.15.1-pre.2 `sec:ring`, paragraph "The Hecke algebra over `Z/ell^K`, read mod `ell`"; the `stable` verdicts are readings, not certificates); 123 checks; env `HECKE_RING_LOEWY_CELLS`; no record key, golden object or setting |
| `HeckeCosetTables.{h,cpp}`, `HeckeTauTable.{h,cpp}`, `HeckeOperatorOnK.{h,cpp}` (WP-10) | coset streaming (`ScopedCosetStreaming`, opt-out `HECKE_COSET_STREAM=0`) and the serial phase on the pool (`ScopedParallelSetup`, `HECKE_PARALLEL_SETUP=0`, above `items · work ≥ 2^22`); both **on by default** (Q8); no fingerprint, no value |
| `HeckeLowRank.h` (G0) | `rank_one_rho_factors`: the rank-one closed form over `F_ell` (opt-out `HECKE_RANK1_CLOSED_FORM=0`, `ScopedRankOneClosedForm`), and `HeckeOperatorOptions::for_each_index`, the host fan-out budget per consumer pool |
| `HeckeUEllStreamValidation.{h,cpp}` (`slow`, `Registry/Slot14/WP10.cpp`; **S24 CL-R1: the `.cpp` is `legacy/HeckeUEllStreamValidation.cpp`, `[SKIP]` in OFF builds**), `ValidationHarnessCatchValidation` (`fast`, `Registry/Slot14/WP2.cpp`) | 2262 checks (arms A–I); the engineered-throw test of the harness catch path |

### Slot 14, WP-12 — level `N` on the model, first build (`docs/implementation/slot14/WP-12_report.md`)

| File | What |
|---|---|
| `HeckeLevelN.{h,cpp}` (new, WP-12) | `hecke::level` — [C] v1.14.0 `sec:levelN`: `Gamma0LevelN` (a `CosetLevel` implementing `def:cosetrepsN` (1)–(5) for **every** `N`: component indices at each `q^ν`, the mixed radix, canonical pairs by CRT, representatives by cases (i) / (ii) / (iii), the class map and the digitwise right action; at `N = p^ν` it **is** the shipped `def:cosetreps` table, `lem:cosetrepsN`(3)), `Gamma0SubTransversalN` (`def:cosetrepsN`(6), the pinned `r_t` of `T_p^{(N)}` / `U_p^{(N)}`), `InducedRelationsLevel` (`L_A` for any `CosetLevel` over any `ModCtx`, `def:levelNmodel`), `MonomialAction` / `OrbitWiseFixedSpace` / `DenseFixedSpace` (`prop:levelorbit`(1): `M^{⟨S⟩}`, `M^{⟨ST⟩}` orbit by orbit, one `d × d` idempotent per fixed point, none on free orbits), `CokernelFromFixedSpaces` (`prop:levelorbit`(2)), `IndexGamma0` / `Nu2` / `Nu3` / `CuspCount` / `DimCuspForms`. The operators are `HeckeLevelEll`'s `BuildLevelCorrespondence` (`eq:levelcorr`, the level-1 τ-table, no new τ-table). Pipeline-free and value-free. **Not built** (slot 15): the Heilbronn action `H_{N,p}`, the sparse cokernel elimination, composite-`N` degeneracies |
| `HeckeLevelNValidation.{h,cpp}` (new; `standard`, `Registry/Slot14/WP12.cpp`) | 1665 checks (1574 before the 2026-09-29 review repairs), ≈ 4 s, ≈ 0.2 GB: (T) `lem:cosetrepsN` for `N ≤ 60` and eleven larger `N` (incl. mixed classes; prime powers = the shipped table); (D) (LN1) and `prop:levelorbit`(3); (O) (LN2) orbit-wise = dense; (X) cochain maps into `C_eq` (every even-`n` `F_P` cell and both `Z/5²` cells); (H) commutativity and independence of the pinned `r_t`; (G) an independent group-cocycle route at composite `N` (dims + charpolys); (K) classical charpolys; (E) the plug-point regression at `N = ell` (relations, cochain maps and `H^1` matrices entrywise = slot-13 WP-6's); (R3) Mazur's Eisenstein ideal at weight 2 on the 26 scan cells `N ≤ 211` (`dim W_{N,p}`, Loewy length `e + 1`, Merel's criterion); `HECKE_LEVEL_N_R3_CELLS` (read-only) replaces the R3 list |

### Slot 15, WP-7 / WP-8 / WP-9 / WP-17 — level `N` at scale (`docs/implementation/slot15/`)

| File | What |
|---|---|
| `HeckeLevelNHeilbronn.{h,cpp}` (new, WP-7) | [C] `sec:levelN` `lem:merelpath`, `prop:heilbronn`, `rmk:heilbronnconv`: Merel's set `Mer_p`, `HeilbronnOperator` (`H_{N,p}` on the induced module, `T_p` at `p ∤ N`, `U_p` at `p \| N`), `CokernelPresentation` (the dense `M/Rel_N`, the natural RREF), `OnCokernel`, `RelEscapes` |
| `HeckeLevelNHeilbronnValidation.{h,cpp}` (new; `standard`, `Registry/Slot15/WP7.cpp`) | 636 checks, ≈ 2 s: (M) `Mer_p`, (N1) level 1, (Z) `Rel_N` preserved / the convention traps, (L) (LN3) on both halves (Heilbronn = τ-route) on 39 cells, 133 operators, (K) classical charpolys, (E) the `N = ell` plug point |
| `HeckeLevelNSparse.{h,cpp}` (new, WP-8; WP-17) | [C] `rmk:levelcost`(4): `OrbitWiseFixedGenerators`, `SparseCokernel` (sparse Gauss–Jordan, smallest-column pivots, entrywise = the dense RREF), `OnCokernelSparse`, the fibre maps `α^0`, `ω` on the presentation (`LevelChangeMap`, `OnCokernelLevelChange`; [C] 1.15.0-pre.3 scope sentence), `CokernelCostModel` / `ChooseCokernelRoute` / `CokernelHeckeMatrices` (`kAuto` = the cost model; `p ∈ {2, 3}` refused by name on every route, WP-17), `ReadEisensteinR3`; `D = m (n + 1)` checked to fit `uint32` (WP-17); slot-18 WP-5: the generators are consumed sorted by their largest column, descending (value-free, the RREF is unique; rollback `HECKE_LEVEL_N_GEN_ORDER=as_built`), `PresentationDigest` |
| `HeckeLevelNSparseValidation.{h,cpp}` (new; `standard`, `Registry/Slot15/WP8.cpp`) | 1229 checks, ≈ 8 s, ≈ 0.1 GB: (S) sparse = dense on 47 cells (`T_q`, `q ≤ 13` at `D ≤ 300`), (F) level change (ranks and joint invariants across routes, WP-17), (D) degeneracy maps `α_d` on the model, the old space, (R) R3 on the sparse route, (C) the ladder and cost model; env `HECKE_LEVEL_N_SPARSE_CELLS="N:n:p,…"` |
| `HeckeLevelNOrderValidation.{h,cpp}` (slot-18 WP-5; `cmake/slot18/WP5.cmake`, no new name) | arms (W18o) (the generator order is value-free, 12 cells + 4 Hecke cells) and (W18g) (`Gamma1LevelN::contains_wide`) of `HeckeLevelNSparseValidation` |
| `HeckeLevelNR3.{h,cpp}` (new, WP-9; WP-17) | `ReadEisensteinR3Scan` (the sparse presentation with engine dense kernels; `W` returned in the quotient coordinates), `R3HeckePrimes`, `MerelCriterionLN5` (product form, WP-17); the `GeneralisedKernel` guard throws on exhaustion (WP-17) |
| `HeckeLevelNValidation.cpp` (WP-9, WP-17) | 1883 checks (slot-16 WP-2: + (R3m) the model route vs the dense presentation, basis-free, at `(11,5) (31,5) (61,5)`); 1871 at slot 15: + (R3s) the frozen cells on the sparse route, + (R3x) `W` as a subspace across routes and the scan reader vs `ReadEisensteinR3` at `(3001, 5)`, `(4159, 7)`; hook `HECKE_LEVEL_N_R3_CELLS` on the sparse route (bare `N:p` readings, `_ROUTE=model`, `_QMAX`) |
