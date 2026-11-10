import Mathlib

/-!
# L3-b(1): the `obl:tau` checker (L16-4)

[GT] `chk:tau`, mirroring [C] `def:tau` (T1)–(T3), `alg:tau` step 0 (the single (T1) helper),
`def:primitives` (P1), (P4) (the canonical key), and `obl:tau` (i)–(v), (vii).

**What is modelled.** A `TauData` is the C++ dump of `HECKE_DUMP_TAU` (L16-4's hook): the
7-vertex model's simplex lists, its orbit representatives and transporter sets `S(σ₁, σ₂)` with
their vertex correspondences, the coset data `N`, `g = diag(1, N)`, `s_1, …, s_m`, and the
`τ`-table, one stored chain `τ(s_i · ι(σ))` per slot `(i, q, σ)`.

* An oriented simplex `γ · ι(σ)` of `Kᵘ` is named by its **canonical key** (P4) `canon`: the
  lexicographically least `γ t`, `t ∈ S(σ, σ̂)`, `σ̂` the orbit representative, with the parity of
  `t`'s vertex correspondence as a sign.  A chain of `C_q(Kᵘ)` is a finite list of
  (key, coefficient) pairs, and its **value** `Chain.eval` is the coefficient function
  `Key → ℤ`; two chains are equal iff their values are equal functions.
* The (T1) helper `tauT1 γ q σ` is [C] `alg:tau` step 0: split `γ = h s_i` with
  `h ∈ Γ₀(N)` (P1) -- here by the test `N ∣ (γ · adj s_i)₂₁`, an independent re-derivation of the
  C++'s bottom-row class map -- and return `(g⁻¹ h g) · τ(s_i · ι(σ))` in canonical form.
* `TauSpec D` is the `obl:tau` predicate on generators:
  (i) structure, (ii) = (T3), (iii) = (T2), (iv) = (T1) bookkeeping, (v) the int32-tier bound
  `max(‖γ_t‖, 4N²) < 2^31`, (vii) the integral solve regime; plus the completeness of the slot set, the
  coset data and the closure of the simplex lists under faces.
* `tauCheck` is the Boolean checker; `tauCheck_sound` (`tauCheck D = true → TauSpec D`) and
  `tauCheck_complete` make `TauSpec D` a **decidable** `Prop` (`instDecidableTauSpec`).

**The trust boundary.** That canonical keys are a faithful coordinate system on `C_•(Kᵘ)` (i.e.
that the dumped transporter sets are the true sets `{t ∈ Γ : ι(σ₁) = t ι(σ₂)}` of the 7-vertex
model, and that `Kᵘ` is that model's `Γ`-tessellation) is the C++ model certificate
`SL2ZTessellationTables::validate_closure` and [C] `H4`; it is an input here, not re-proved.
Given it, `TauSpec D` is exactly (T1)–(T3) of [C] `def:tau` on the generators `s_i · ι(σ)`.
-/

namespace GL2.TauChk

/-- An integral `2 × 2` matrix `[[a, b], [c, d]]` (row-major, as `hecke::Mat22`). -/
structure M22 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  deriving DecidableEq, Repr

namespace M22

def mul (x y : M22) : M22 :=
  ⟨x.a * y.a + x.b * y.c, x.a * y.b + x.b * y.d, x.c * y.a + x.d * y.c, x.c * y.b + x.d * y.d⟩

instance : Mul M22 := ⟨mul⟩

def det (x : M22) : ℤ := x.a * x.d - x.b * x.c

/-- The adjugate; the inverse of a determinant-one matrix. -/
def adj (x : M22) : M22 := ⟨x.d, -x.b, -x.c, x.a⟩

/-- `hecke`'s `lex_less` on `(a, b, c, d)`. -/
def lexLt (x y : M22) : Bool :=
  if x.a ≠ y.a then x.a < y.a else if x.b ≠ y.b then x.b < y.b else
    if x.c ≠ y.c then x.c < y.c else x.d < y.d

def maxAbs (x : M22) : ℕ := max (max x.a.natAbs x.b.natAbs) (max x.c.natAbs x.d.natAbs)

end M22

/-- A stored term `coeff · γ · ι(σ)`. -/
structure Term where
  coeff : ℤ
  g : M22
  sigma : ℕ
  deriving DecidableEq, Repr

/-- A transporter `t ∈ S(s₁, s₂)` of dimension `q` with its vertex correspondence. -/
structure Tr where
  q : ℕ
  s1 : ℕ
  s2 : ℕ
  g : M22
  perm : List ℕ
  deriving DecidableEq, Repr

/-- The stored chain of slot `(i, q, σ)`. -/
structure Slot where
  i : ℕ
  q : ℕ
  sigma : ℕ
  terms : List Term
  deriving DecidableEq, Repr

/-- The dump. `simplices` lists `(q, index, vertices)`; `orbitRep` lists `(q, σ, σ̂)`. -/
structure TauData where
  N : ℕ
  m : ℕ
  g : M22
  s : List M22
  simplices : List (ℕ × ℕ × List ℕ)
  orbitRep : List (ℕ × ℕ × ℕ)
  trs : List Tr
  slots : List Slot
  regime : String
  fnv : ℕ
  deriving Repr

/-- A key `(q, σ̂, γ_canon)` of an oriented simplex of `Kᵘ`. -/
abbrev Key := ℕ × ℕ × M22

/-- A chain: (key, coefficient) pairs; its value is `Chain.eval`. -/
abbrev Chain := List (Key × ℤ)

/-- The coefficient function of a chain (`C_q(Kᵘ)` as finitely supported `Key → ℤ`). -/
def Chain.eval (l : Chain) (k : Key) : ℤ := ((l.filter (fun p => decide (p.1 = k))).map Prod.snd).sum

/-- The parity of a vertex-slot permutation restricted to slots `0..q` (`perm_sign`). -/
def permSign (perm : List ℕ) (q : ℕ) : ℤ :=
  let inv := ((List.range (q + 1)).map fun i => ((List.range (q + 1)).filter fun j =>
    decide (i < j) && decide (perm.getD j 0 < perm.getD i 0)).length).sum
  if inv % 2 = 0 then 1 else -1

section
variable (D : TauData)

def TauData.rep (q σ : ℕ) : ℕ :=
  match D.orbitRep.find? (fun r => r.1 = q ∧ r.2.1 = σ) with
  | some r => r.2.2
  | none => σ

/-- The transporter set `S(σ₁, σ₂)` in dimension `q`. -/
def TauData.trSet (q σ₁ σ₂ : ℕ) : List Tr := D.trs.filter fun t => t.q = q ∧ t.s1 = σ₁ ∧ t.s2 = σ₂

/-- (P4) The canonical key of `γ · ι(σ)` and its orientation sign. -/
def TauData.canon (γ : M22) (q σ : ℕ) : Key × ℤ :=
  let sh := D.rep q σ
  let pick := (D.trSet q σ sh).foldl (fun (acc : Option (M22 × ℤ)) t =>
    let cand := γ * t.g
    match acc with
    | none => some (cand, permSign t.perm q)
    | some (best, sg) => if M22.lexLt cand best then some (cand, permSign t.perm q) else some (best, sg))
    none
  match pick with
  | some (best, sg) => ((q, sh, best), sg)
  | none => ((q, σ, γ), 1)

/-- A list of terms of dimension `q`, scaled by `e`, in canonical form. -/
def TauData.canonTerms (q : ℕ) (e : ℤ) (ts : List Term) : Chain :=
  ts.map fun t => let k := D.canon t.g q t.sigma; (k.1, e * t.coeff * k.2)

def TauData.sAt (i : ℕ) : M22 := D.s.getD (i - 1) ⟨1, 0, 0, 1⟩

/-- (P1) the class `i` with `γ s_i⁻¹ ∈ Γ₀(N)` (first match; `0` if none). -/
def TauData.classOf (γ : M22) : ℕ :=
  ((List.range D.m).map (· + 1)).find? (fun i => (γ * (D.sAt i).adj).c % (D.N : ℤ) = 0) |>.getD 0

/-- `g⁻¹ h g` for `g = diag(1, N)`, `h = γ s_i⁻¹`. -/
def TauData.conj (γ : M22) : M22 :=
  let h := γ * (D.sAt (D.classOf γ)).adj
  ⟨h.a, (D.N : ℤ) * h.b, h.c / (D.N : ℤ), h.d⟩

def TauData.lookup (i q σ : ℕ) : List Term :=
  match D.slots.find? (fun s => s.i = i ∧ s.q = q ∧ s.sigma = σ) with
  | some s => s.terms
  | none => []

/-- **The single (T1) helper** ([C] `alg:tau` step 0): `τ(γ · ι(σ)) = (g⁻¹hg) · τ(s_i · ι(σ))`,
scaled by `e`.  If `γ` lies in no class (`classOf γ = 0`, a defective coset table) the helper
returns the poison chain on the key `(3, 0, 0)` of a non-existent degree, so no (T1)/(T2)
comparison involving it can succeed. -/
def TauData.tauT1 (e : ℤ) (γ : M22) (q σ : ℕ) : Chain :=
  if D.classOf γ = 0 then [((3, 0, ⟨0, 0, 0, 0⟩), 1)] else
  D.canonTerms q e ((D.lookup (D.classOf γ) q σ).map fun t => { t with g := D.conj γ * t.g })

def TauData.verts (q σ : ℕ) : List ℕ :=
  match D.simplices.find? (fun x => x.1 = q ∧ x.2.1 = σ) with
  | some x => x.2.2
  | none => []

/-- The face `j` (vertex `j` removed) of the `q`-simplex `σ`, as the simplex record it is, if any. -/
def TauData.faceRec? (q σ j : ℕ) : Option (ℕ × ℕ × List ℕ) :=
  D.simplices.find? (fun x => x.1 = q - 1 ∧ x.2.2 = (D.verts q σ).eraseIdx j)

/-- A simplex index no simplex of the dump carries (one past the largest). -/
def TauData.poisonIdx : ℕ := D.simplices.foldl (fun a x => max a (x.2.1 + 1)) 0

/-- The index of face `j` (vertex `j` removed) of the `q`-simplex `σ` (`face_simplex_index`).  A missing
face gives the **poison** index `poisonIdx` (L16-5 minor 6), not a valid simplex; `TauSpec.closed` rules
the case out on an accepted table. -/
def TauData.face (q σ j : ℕ) : ℕ :=
  match D.faceRec? q σ j with
  | some x => x.2.1
  | none => D.poisonIdx

/-- Closure under faces: every face `∂_j σ` (`j ≤ q`) of every simplex of dimension `q ≥ 1` is a simplex
of the dump. -/
def TauData.closedB : Bool :=
  D.simplices.all fun x => (List.range (x.1 + 1)).all fun j => x.1 == 0 || (D.faceRec? x.1 x.2.1 j).isSome

def sgnPow (j : ℕ) : ℤ := if j % 2 = 0 then 1 else -1

/-- `∂` of a chain of terms of dimension `q`: `∑_j (-1)^j γ · ι(∂_j σ)`. -/
def TauData.bdry (q : ℕ) (ts : List Term) : Chain :=
  ((List.range (q + 1)).map fun j =>
    D.canonTerms (q - 1) (sgnPow j) (ts.map fun t => { t with sigma := D.face q t.sigma j })).flatten

/-- `τ(∂(s_i · ι(σ)))`, every summand through the (T1) helper. -/
def TauData.tauBdry (i q σ : ℕ) : Chain :=
  ((List.range (q + 1)).map fun j => D.tauT1 (sgnPow j) (D.sAt i) (q - 1) (D.face q σ j)).flatten

def TauData.slotKeys : List (ℕ × ℕ × ℕ) := D.slots.map fun s => (s.i, s.q, s.sigma)

/-- The slots the table must have: `i ∈ [1, m]` times every simplex. -/
def TauData.wantKeys : List (ℕ × ℕ × ℕ) :=
  ((List.range D.m).map (· + 1)).flatMap fun i => D.simplices.map fun x => (i, x.1, x.2.1)

end

/-- **The `obl:tau` predicate on generators** ([C] `obl:tau`; (T1)–(T3) of `def:tau`). -/
structure TauSpec (D : TauData) : Prop where
  /-- the coset data: `g = diag(1, N)`, `m` representatives of determinant one, pairwise in
  distinct `Γ₀(N)`-cosets, and every `q`-simplex has a nonempty `S(σ, σ̂)`. -/
  cosets : D.g = ⟨1, 0, 0, D.N⟩ ∧ 0 < D.N ∧ D.s.length = D.m ∧ (∀ x ∈ D.s, x.det = 1) ∧
    (∀ i ∈ List.range D.m, ∀ j ∈ List.range D.m, i ≠ j →
      (D.sAt (i + 1) * (D.sAt (j + 1)).adj).c % (D.N : ℤ) ≠ 0) ∧
    (∀ x ∈ D.simplices, D.trSet x.1 x.2.1 (D.rep x.1 x.2.1) ≠ [])
  /-- closure (L16-5 minor 6; the C++ `validate_closure` on its side): every face `∂_j σ` of every
  simplex of positive dimension is a simplex of the dump, so `face` never returns its poison index. -/
  closed : D.closedB = true
  /-- completeness: exactly one stored chain per slot `(i, q, σ)`. -/
  complete : D.slotKeys.Nodup ∧ D.slotKeys.Perm D.wantKeys
  /-- (i) structure: `det γ_t = 1`, non-zero coefficient, canonical form with sign `+1`. -/
  struct : ∀ s ∈ D.slots, ∀ t ∈ s.terms,
    t.coeff ≠ 0 ∧ t.g.det = 1 ∧ D.canon t.g s.q t.sigma = ((s.q, t.sigma, t.g), 1)
  /-- (ii) = (T3): `ε(τ(s_i · ι(v))) = 1`. -/
  T3 : ∀ s ∈ D.slots, s.q = 0 → (s.terms.map Term.coeff).sum = 1
  /-- (iii) = (T2): `∂ τ(x) = τ(∂ x)` on every generator of degree 1 and 2, the right-hand side
  through the (T1) helper only. -/
  T2 : ∀ s ∈ D.slots, s.q = 1 ∨ s.q = 2 →
    Chain.eval (D.bdry s.q s.terms) = Chain.eval (D.tauBdry s.i s.q s.sigma)
  /-- (iv) = (T1) bookkeeping: for every transporter `ι(σ) = t · ι(σ')` and every `i`, the slot
  `(i, σ)` is the sign-tracked (T1)-translate of `(s_i t) · ι(σ')`. -/
  T1 : ∀ s ∈ D.slots, ∀ t ∈ D.trs, t.q = s.q → t.s1 = s.sigma →
    Chain.eval (D.canonTerms s.q 1 s.terms) =
      Chain.eval (D.tauT1 (permSign t.perm s.q) (D.sAt s.i * t.g) s.q t.s2)
  /-- (v) bounds: `B = max(max ‖γ_t‖, 4N²) ≤ 2^31 - 1` ([C] `obl:backend`(ii); the `4N²` term since
  L16-5 minor 5). -/
  bounds : (∀ s ∈ D.slots, ∀ t ∈ s.terms, t.g.maxAbs < 2 ^ 31) ∧ 4 * D.N ^ 2 < 2 ^ 31
  /-- (vii) the solve regime is recorded and integral. -/
  integral : D.regime = "Z"

/-! ### The checker -/

/-- Chain equality, decided keywise on the keys that occur. -/
def chainEqB (l₁ l₂ : Chain) : Bool := (l₁ ++ l₂).all fun p => decide (Chain.eval l₁ p.1 = Chain.eval l₂ p.1)

lemma Chain.eval_eq_zero_of_not_mem (l : Chain) (k : Key) (h : k ∉ l.map Prod.fst) :
    Chain.eval l k = 0 := by
  induction l with
  | nil => simp [Chain.eval]
  | cons p l ih =>
    simp only [List.map_cons, List.mem_cons, not_or] at h
    have hp : ¬ p.1 = k := fun e => h.1 e.symm
    simp only [Chain.eval, List.filter_cons, hp, decide_false] at ih ⊢
    exact ih h.2

theorem chainEqB_iff (l₁ l₂ : Chain) : chainEqB l₁ l₂ = true ↔ Chain.eval l₁ = Chain.eval l₂ := by
  constructor
  · intro h
    funext k
    by_cases hk : k ∈ (l₁ ++ l₂).map Prod.fst
    · obtain ⟨p, hp, rfl⟩ := List.mem_map.mp hk
      exact of_decide_eq_true (List.all_eq_true.mp h p hp)
    · simp only [List.map_append, List.mem_append, not_or] at hk
      rw [Chain.eval_eq_zero_of_not_mem _ _ hk.1, Chain.eval_eq_zero_of_not_mem _ _ hk.2]
  · intro h
    exact List.all_eq_true.mpr fun p _ => decide_eq_true (congrFun h p.1)

/-- **The checker.** Each conjunct decides the corresponding field of `TauSpec`. -/
def tauCheck (D : TauData) : Bool :=
  decide (D.g = ⟨1, 0, 0, D.N⟩ ∧ 0 < D.N ∧ D.s.length = D.m ∧ (∀ x ∈ D.s, x.det = 1) ∧
    (∀ i ∈ List.range D.m, ∀ j ∈ List.range D.m, i ≠ j →
      (D.sAt (i + 1) * (D.sAt (j + 1)).adj).c % (D.N : ℤ) ≠ 0) ∧
    (∀ x ∈ D.simplices, D.trSet x.1 x.2.1 (D.rep x.1 x.2.1) ≠ [])) &&
  D.closedB &&
  decide (D.slotKeys.Nodup ∧ D.slotKeys.Perm D.wantKeys) &&
  decide (∀ s ∈ D.slots, ∀ t ∈ s.terms,
    t.coeff ≠ 0 ∧ t.g.det = 1 ∧ D.canon t.g s.q t.sigma = ((s.q, t.sigma, t.g), 1)) &&
  decide (∀ s ∈ D.slots, s.q = 0 → (s.terms.map Term.coeff).sum = 1) &&
  D.slots.all (fun s => !(s.q = 1 ∨ s.q = 2) || chainEqB (D.bdry s.q s.terms) (D.tauBdry s.i s.q s.sigma)) &&
  D.slots.all (fun s => D.trs.all fun t => !(t.q = s.q ∧ t.s1 = s.sigma) ||
    chainEqB (D.canonTerms s.q 1 s.terms) (D.tauT1 (permSign t.perm s.q) (D.sAt s.i * t.g) s.q t.s2)) &&
  decide ((∀ s ∈ D.slots, ∀ t ∈ s.terms, t.g.maxAbs < 2 ^ 31) ∧ 4 * D.N ^ 2 < 2 ^ 31) &&
  decide (D.regime = "Z")

/-- **Soundness**: an accepted table satisfies the `obl:tau` predicate. -/
theorem tauCheck_sound (D : TauData) (h : tauCheck D = true) : TauSpec D := by
  simp only [tauCheck, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨h1, h0⟩, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩, h7⟩, h8⟩ := h
  refine ⟨h1, h0, h2, h3, h4, ?_, ?_, h7, h8⟩
  · intro s hs hq
    have := List.all_eq_true.mp h5 s hs
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at this
    rcases this with hn | hc
    · exact absurd hq hn
    · exact (chainEqB_iff _ _).mp hc
  · intro s hs t ht hq h1'
    have := List.all_eq_true.mp (List.all_eq_true.mp h6 s hs) t ht
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at this
    rcases this with hn | hc
    · exact absurd ⟨hq, h1'⟩ hn
    · exact (chainEqB_iff _ _).mp hc

/-- **Completeness**: the checker accepts every table satisfying the predicate. -/
theorem tauCheck_complete (D : TauData) (h : TauSpec D) : tauCheck D = true := by
  obtain ⟨h1, h0, h2, h3, h4, h5, h6, h7, h8⟩ := h
  simp only [tauCheck, Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨⟨⟨⟨⟨⟨⟨h1, h0⟩, h2⟩, h3⟩, h4⟩, ?_⟩, ?_⟩, h7⟩, h8⟩
  · refine List.all_eq_true.mpr fun s hs => ?_
    by_cases hq : s.q = 1 ∨ s.q = 2
    · simp only [hq, Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not]
      exact Or.inr ((chainEqB_iff _ _).mpr (h5 s hs hq))
    · simp [hq]
  · refine List.all_eq_true.mpr fun s hs => List.all_eq_true.mpr fun t ht => ?_
    by_cases hq : t.q = s.q ∧ t.s1 = s.sigma
    · simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not]
      exact Or.inr ((chainEqB_iff _ _).mpr (h6 s hs t ht hq.1 hq.2))
    · simp [hq]

/-- `TauSpec` is a decidable `Prop`: `tauCheck` decides it. -/
instance instDecidableTauSpec (D : TauData) : Decidable (TauSpec D) :=
  decidable_of_iff (tauCheck D = true) ⟨tauCheck_sound D, tauCheck_complete D⟩

/-! ### The golden tie: FNV-1a-64 exactly as `HeckePairHomotopy::tau_table_fnv64` -/

def fnvPrime : ℕ := 0x100000001b3
def fnvOffset : ℕ := 0xcbf29ce484222325

/-- `engine::FnvU64`: the 8 bytes of `v`, least significant first. -/
def fnvU64 (h v : ℕ) : ℕ :=
  (List.range 8).foldl (fun h i => ((h ^^^ ((v / 256 ^ i) % 256)) * fnvPrime) % 2 ^ 64) h

/-- Two's complement `int64 → uint64`. -/
def u64OfInt (x : ℤ) : ℕ := (x % (2 ^ 64 : ℤ)).toNat

def fnvTerms (h : ℕ) (ts : List Term) : ℕ :=
  ts.foldl (fun h t => [u64OfInt t.coeff, u64OfInt t.g.a, u64OfInt t.g.b, u64OfInt t.g.c,
    u64OfInt t.g.d, t.sigma].foldl fnvU64 h) (fnvU64 h ts.length)

/-- The table hash: `N, m`, then per `(q, σ, j)` in simplex order: `q, σ, j` and the terms. -/
def tableFnv (D : TauData) : ℕ :=
  D.simplices.foldl (fun h x => ((List.range D.m).map (· + 1)).foldl (fun h j =>
      fnvTerms ([x.1, x.2.1, j].foldl fnvU64 h) (D.lookup j x.1 x.2.1)) h)
    (fnvU64 (fnvU64 fnvOffset D.N) D.m)

/-- `engine::HashScalar64`: what the golden reference stores for a scalar object. -/
def hashScalar64 (v : ℕ) : ℕ := fnvU64 fnvOffset v

end GL2.TauChk
