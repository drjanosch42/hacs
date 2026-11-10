import GL2.Model.KeyTie

/-!
# MP-L1, row L0 (slot 22): orbifold triangulations of `[SL₂(ℤ)\Y]` as finite data; Moerdijk–Pronk's rules as decidable
predicates

[GT] `gt_part_MP.tex`, rows `def:mp-datum`, `def:mp-predicates`, backing [C] `def:mptri`, `def:mprules`, `def:mpell`
(and, with `GL2.MP.Models`, [C] `def:mpRZ` and `tab:mprules`).

**The datum** (`MPDatum`; integer matrices `M22`, decided by the kernel):

* `stab o` -- all elements of the stabiliser `Γ_o` of the representative `ι(o)` of the vertex orbit `o` (a finite list);
* `pts p = (o, g)` -- the points of the lifted domain, `p = g · ι(o)`; `bd p` -- `p` lies on the boundary `∂Y`;
* `top σ` -- the maximal simplices `ι(σ)` of the lifted domain as lists of point indices (stored order).

A vertex of `Y` is `(o, g Γ_o)`: `samePt (o, g) (o', g')` iff `o = o'` and `g⁻¹ g' ∈ Γ_o`.  The **arrows**
`Ar(σ, σ') = {γ : ι(σ) ∩ γ ι(σ') ≠ ∅}` are, under (SC), the finite union `⋃ g_p Γ_o g_w⁻¹` over the vertices `p` of
`ι(σ)` and `w` of `ι(σ')` in one orbit `o` (`arr`; [C] `rmk:mpwhat` (1)).

**The predicates** (Booleans; `= true` is the claim):

* `wfB` -- well-formedness: determinants one, every `stab o` a group (contains `1`, closed under products and inverses),
  indices in range;
* `ioB` -- (IO) **invariant order**: no maximal simplex has two vertices in one orbit, so ordering by orbit index is a
  `Γ`-invariant total order on every simplex ([C] `def:mptri` (3)); `orderedTop` is that order;
* `scTopB` -- (SC) on the maximal simplices: distinct representatives are not `Γ`-translates of one another as vertex sets
  (with (IO) a transporter between equal vertex sets fixes them pointwise);
* `adaptedB none` -- **adapted**, MP's rule (ii) in the **subgroup form** ([C] `def:mprules` (1)): every face `τ` of every
  maximal simplex has a vertex `v` with `Γ_p ⊆ Γ_v` for all vertices `p` of `τ` (point stabilisers as subgroups of `Γ` after
  transport, `g_v⁻¹ g_p Γ_{o_p} g_p⁻¹ g_v ⊆ Γ_{o_v}`; since `Γ_τ = ⋂_{p ∈ τ} Γ_p`, this is "`Γ_τ' ⊆ Γ_v` for every face
  `τ'` of `τ`");
* `adaptedB (some ℓ)` -- **adapted for `𝔗^{(ℓ)}`** ([C] `def:mprules` (2), `def:mpell`): the same with the `ℓ`-Sylow
  subgroups `P_p = {h ∈ Γ_p : h^{|Γ_p|_ℓ} = 1}` (the stabilisers are cyclic);
* `goodStrongB` -- **good**, the strong form of the C++ and `mp_rules.py`: `T(v) = {γ : γ St(v) ∩ St(v) ≠ ∅}` lies in the
  stabiliser of one point of `St(v)` ([C] `def:mprules` (3) writes `T(v) ⊆ Γ_v`, which is stronger: `goodVtxAtB`);
* `collarB` -- **collar** ([C] `def:mprules` (4)): no maximal simplex meets both `∂Y` and a vertex whose stabiliser is not
  the kernel `{±I}` of the action;
* `quotV`, `quotE`, `quotF` -- the quotient counts `(V, E, F)` (orbits of vertices, edges, maximal simplices).

The orders-only comparison of `mp_rules.py` is `adaptedOrderB`; the subgroup form is the normative one ([C]
`rmk:mprulesread` (4)).
-/

namespace GL2.MP

open GL2.TauChk

/-- The identity matrix. -/
def M22.I : M22 := ⟨1, 0, 0, 1⟩

/-- `x^n`. -/
def M22.pow (x : M22) : ℕ → M22
  | 0 => M22.I
  | n + 1 => x * M22.pow x n

/-- **An MP descriptor**: vertex-orbit stabilisers (all elements), lifted points `(o, g)`, boundary flags, maximal
simplices of the lifted domain. -/
structure MPDatum where
  stab : List (List M22)
  pts : List (ℕ × M22)
  bd : List Bool
  top : List (List ℕ)
  deriving Repr

namespace MPDatum

variable (D : MPDatum)

/-- `Γ_o`. -/
def st (o : ℕ) : List M22 := D.stab.getD o []

/-- The orbit of point `p`. -/
def orb (p : ℕ) : ℕ := (D.pts.getD p (0, M22.I)).1

/-- The translate `g_p` of point `p` (`p = g_p ι(o_p)`). -/
def gp (p : ℕ) : M22 := (D.pts.getD p (0, M22.I)).2

/-- The point indices of maximal simplex `σ`. -/
def tv (σ : ℕ) : List ℕ := D.top.getD σ []

/-- `(o, g)` and `(o', g')` are the same vertex of `Y`. -/
def samePt (o : ℕ) (g : M22) (o' : ℕ) (g' : M22) : Bool := o == o' && (g.adj * g') ∈ D.st o

/-- `γ · p'` is the vertex `p`. -/
def maps (γ : M22) (p' p : ℕ) : Bool := D.samePt (D.orb p) (D.gp p) (D.orb p') (γ * D.gp p')

/-- The elements `g Γ_o g'⁻¹`. -/
def coset (g : M22) (o : ℕ) (g' : M22) : List M22 := (D.st o).map fun h => g * h * g'.adj

/-- **The arrows** `Ar(σ, σ') = {γ : ι(σ) ∩ γ ι(σ') ≠ ∅}` (deduplicated). -/
def arr (σ σ' : ℕ) : List M22 :=
  ((D.tv σ).flatMap fun p => (D.tv σ').flatMap fun w =>
    if D.orb p = D.orb w then D.coset (D.gp p) (D.orb p) (D.gp w) else []).dedup

/-! ### Well-formedness -/

/-- Every listed stabiliser is a group of determinant-one matrices. -/
def stabGroupB : Bool :=
  D.stab.all fun l => l.all (fun h => h.det == 1) && M22.I ∈ l && l.all (fun h => l.all fun h' => (h * h') ∈ l) &&
    l.all fun h => h.adj ∈ l

/-- **Well-formedness.** -/
def wfB : Bool :=
  D.stabGroupB && D.pts.all (fun x => x.1 < D.stab.length && x.2.det == 1) && D.bd.length == D.pts.length &&
    D.top.all (fun σ => !σ.isEmpty && σ.all fun p => p < D.pts.length)

/-! ### (IO), (SC) -/

/-- **(IO)**: no maximal simplex has two vertices in one orbit. -/
def ioB : Bool := D.top.all fun σ => (σ.map D.orb).Nodup

/-- The vertices of `σ` in the invariant order (by orbit index). -/
def orderedTop (σ : ℕ) : List ℕ := (D.tv σ).mergeSort fun p q => D.orb p ≤ D.orb q

/-- `γ ι(σ')` and `ι(σ)` have the same vertex set. -/
def sameSet (γ : M22) (σ' σ : ℕ) : Bool :=
  (D.tv σ').all (fun w => (D.tv σ).any fun p => D.maps γ w p) &&
    (D.tv σ).all fun p => (D.tv σ').any fun w => D.maps γ w p

/-- **(SC) on the maximal simplices**: distinct representatives are not translates of one another. -/
def scTopB : Bool :=
  (List.range D.top.length).all fun σ => (List.range D.top.length).all fun σ' =>
    σ == σ' || (D.arr σ σ').all fun γ => !D.sameSet γ σ' σ

/-! ### Moerdijk–Pronk's rules -/

/-- `|Γ|_ℓ`, the `ℓ`-part of `n` (for `ℓ` prime). -/
def lpart (ℓ n : ℕ) : ℕ := Nat.gcd n (ℓ ^ n)

/-- The isotropy datum: `Γ_o` (`none`) or its `ℓ`-Sylow subgroup (`some ℓ`). -/
def iso : Option ℕ → ℕ → List M22
  | none, o => D.st o
  | some ℓ, o => (D.st o).filter fun h => M22.pow h (lpart ℓ (D.st o).length) == M22.I

/-- `Γ_p ⊆ Γ_v` (resp. `P_p ⊆ P_v`), after transport: `g_v⁻¹ g_p h g_p⁻¹ g_v ∈ Γ_{o_v}`. -/
def stabLe (sel : Option ℕ) (p v : ℕ) : Bool :=
  (D.iso sel (D.orb p)).all fun h => (D.gp v).adj * D.gp p * h * (D.gp p).adj * D.gp v ∈ D.iso sel (D.orb v)

/-- **Adapted** (subgroup form; `none`: actual isotropy, `some ℓ`: the Sylow datum `𝔗^{(ℓ)}`). -/
def adaptedB (sel : Option ℕ) : Bool :=
  (List.range D.top.length).all fun σ => ((D.tv σ).sublists.filter (!·.isEmpty)).all fun τ =>
    τ.any fun v => τ.all fun p => D.stabLe sel p v

/-- The orders-only form of `mp_rules.py` (comparing `|Γ_p|` instead of subgroups). -/
def adaptedOrderB (sel : Option ℕ) : Bool :=
  (List.range D.top.length).all fun σ => ((D.tv σ).sublists.filter (!·.isEmpty)).all fun τ =>
    τ.any fun v => τ.all fun p => (D.iso sel (D.orb v)).length % (D.iso sel (D.orb p)).length == 0

/-- The closed star of `p`: the translates `g ι(σ)` containing `p`, as pairs `(g, σ)`. -/
def star (p : ℕ) : List (M22 × ℕ) :=
  (List.range D.top.length).flatMap fun σ => (D.tv σ).flatMap fun w =>
    if D.orb w = D.orb p then (D.coset (D.gp p) (D.orb p) (D.gp w)).map fun g => (g, σ) else []

/-- Insert `x` unless present (a linear-time de-duplication step). -/
def insertNew {α : Type} [DecidableEq α] (l : List α) (x : α) : List α := if x ∈ l then l else x :: l

/-- Insert the point `(o, g)` unless the same vertex is present. -/
def insertPt (l : List (ℕ × M22)) (x : ℕ × M22) : List (ℕ × M22) :=
  if l.any (fun y => D.samePt y.1 y.2 x.1 x.2) then l else x :: l

/-- The vertices of the closed star `St(v)` (one name per vertex), as pairs `(o, g)` naming `g ι(o)`. -/
def starPts (v : ℕ) : List (ℕ × M22) :=
  (D.star v).foldl (fun acc x => (D.tv x.2).foldl (fun acc w => D.insertPt acc (D.orb w, x.1 * D.gp w)) acc) []

/-- **`T(v)`** (without repetitions): the `γ` with `γ St(v) ∩ St(v) ≠ ∅`, i.e. `γ a = b` for vertices `a, b` of `St(v)`
(a finite intersection of closed subcomplexes is non-empty iff it has a vertex), `γ ∈ g_b Γ_o g_a⁻¹`; as the C++ `Good`
enumerates it. -/
def tSet (v : ℕ) : List M22 :=
  (D.starPts v).foldl (fun acc a => (D.starPts v).foldl (fun acc b =>
    if a.1 = b.1 then (D.st a.1).foldl (fun acc h => insertNew acc (b.2 * h * a.2.adj)) acc else acc) acc) []

/-- **Good, strong form, as the C++ `Good` / `good_strong` and `mp_rules.py` test it**, at `v`: `T(v)` lies in the
stabiliser `g_p Γ_o g_p⁻¹` of **one point `p = g_p ι(o)` of `St(v)`** (`goodAtB_iff_onePt`).  This is the form behind
[C] `tab:mprules`' "fails at" column.  It is **not** the wording `T(v) ⊆ Γ_v` of [C] 1.22.0-pre.5 `def:mprules` (3)
(that is `goodVtxAtB`, strictly stronger: finding `goodVtx_discrepancy`, reported to the contract editor; the repair of
[C] is the contract editor's, this file keeps the C++ form). -/
def goodAtB (v : ℕ) : Bool :=
  (D.starPts v).any fun p => (D.tSet v).all fun γ => p.2.adj * γ * p.2 ∈ D.st p.1

/-- **[C] 1.22.0-pre.5 `def:mprules` (3)'s literal wording** `T(v) ⊆ Γ_v` (the stabiliser of the vertex `v` itself;
`goodVtxAtB_iff_vtx`).  Strictly stronger than the C++ form `goodAtB` (finding `goodVtx_discrepancy`); kept only to
document the discrepancy. -/
def goodVtxAtB (v : ℕ) : Bool := (D.tSet v).all fun γ => (D.gp v).adj * γ * D.gp v ∈ D.st (D.orb v)

/-- **The C++ form of goodness, as a proposition**: some point `p = g_p ι(o)` of `St(v)` has `g_p⁻¹ T(v) g_p ⊆ Γ_o`. -/
theorem goodAtB_iff_onePt (v : ℕ) :
    D.goodAtB v = true ↔ ∃ p ∈ D.starPts v, ∀ γ ∈ D.tSet v, p.2.adj * γ * p.2 ∈ D.st p.1 := by
  simp [goodAtB]

/-- **[C]'s literal wording, as a proposition**: `g_v⁻¹ T(v) g_v ⊆ Γ_{o_v}`, i.e. `T(v) ⊆ Γ_v`. -/
theorem goodVtxAtB_iff_vtx (v : ℕ) :
    D.goodVtxAtB v = true ↔ ∀ γ ∈ D.tSet v, (D.gp v).adj * γ * D.gp v ∈ D.st (D.orb v) := by
  simp [goodVtxAtB]

/-- **Good** (strong form) at every vertex of the lifted domain. -/
def goodStrongB : Bool := (List.range D.pts.length).all D.goodAtB

/-- The closure of `{1}` under right multiplication by `T`, stopped once it exceeds `6` elements (`fuel` rounds). -/
def closeUnder (T : List M22) : ℕ → List M22 → List M22
  | 0, g => g
  | fuel + 1, g =>
    if 6 < g.length then g else
      let g' := g.foldl (fun acc x => T.foldl (fun acc t => insertNew acc (x * t)) acc) g
      if g'.length = g.length then g else closeUnder T fuel g'

/-- **Good, the finite-group criterion** ([C] `def:mprules` (3)), at `v`: the closure of `T(v)` stabilises within `6`
elements (a finite set closed under products by the generators: the generated group); more than `6` elements is read as
"infinite", the finite subgroups of `SL₂(ℤ)` being cyclic of order `≤ 6` ([C] `rmk:mprulesread` (3)). -/
def goodFinAtB (v : ℕ) : Bool := (closeUnder (D.tSet v) 8 [M22.I]).length ≤ 6

/-- **Good** (finite-group criterion) at every vertex. -/
def goodFinB : Bool := (List.range D.pts.length).all D.goodFinAtB

/-- **Collar**: no maximal simplex meets both `∂Y` and a non-central vertex (stabiliser of order `≠ 2`). -/
def collarB : Bool :=
  (List.range D.top.length).all fun σ =>
    !((D.tv σ).any (fun p => D.bd.getD p false) && (D.tv σ).any fun p => (D.st (D.orb p)).length != 2)

/-! ### Quotient counts -/

/-- Number of vertex orbits met by the maximal simplices. -/
def quotV : ℕ := ((D.top.flatMap id).map D.orb).dedup.length

/-- The edges of the maximal simplices, as point pairs. -/
def edges : List (ℕ × ℕ) :=
  (List.range D.top.length).flatMap fun σ => ((D.tv σ).sublists.filter (·.length == 2)).map fun e =>
    (e.getD 0 0, e.getD 1 0)

/-- `γ {p', q'} = {p, q}`, for `γ` running over the candidates `g_p Γ g_{p'}⁻¹ ∪ g_q Γ g_{p'}⁻¹`. -/
def edgeRel (e e' : ℕ × ℕ) : Bool :=
  let cand := (if D.orb e.1 = D.orb e'.1 then D.coset (D.gp e.1) (D.orb e.1) (D.gp e'.1) else []) ++
    (if D.orb e.2 = D.orb e'.1 then D.coset (D.gp e.2) (D.orb e.2) (D.gp e'.1) else [])
  cand.any fun γ => (D.maps γ e'.1 e.1 && D.maps γ e'.2 e.2) || (D.maps γ e'.1 e.2 && D.maps γ e'.2 e.1)

/-- Number of edge orbits. -/
def quotE : ℕ :=
  (D.edges.foldl (fun reps e => if reps.any (D.edgeRel e) then reps else reps ++ [e]) []).length

/-- Number of maximal-simplex orbits. -/
def quotF : ℕ := D.top.length

/-- `χ = V − E + F`. -/
def quotChi : ℤ := (D.quotV : ℤ) - D.quotE + D.quotF

/-- The number of strings of `S_1` (all, degenerate included): `∑_{σ, σ'} |Ar(σ, σ')|`. -/
def card1 : ℕ :=
  ((List.range D.top.length).flatMap fun σ => (List.range D.top.length).map fun σ' => (D.arr σ σ').length).sum

end MPDatum

/-- A predicate row as the C++ dump prints it: triangulation (`0 = M₇`, `1 = R_ℤ`, `2 = R_ℤ⁺`, `3 = R_good`), `ℓ`
(`0` = full isotropy), adapted (subgroup form), adapted (orders only), good, collar, `V, E, F, χ`. -/
structure PredRow where
  tri : ℕ
  ell : ℕ
  adapted : Bool
  adaptedOrder : Bool
  good : Bool
  collar : Bool
  V : ℕ
  E : ℕ
  F : ℕ
  chi : ℤ
  deriving DecidableEq, Repr

end GL2.MP
