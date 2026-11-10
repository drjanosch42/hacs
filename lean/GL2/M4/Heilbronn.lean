import GL2.M4.Merel
import GL2.Checkers.RhoExec

/-!
# M4-i (slot 20, L20-4): `prop:heilbronn` over `ZMod q`, an executable spec (the L3-a Heilbronn–Merel oracle)

[C] `prop:heilbronn`, `eq:heilbronn`: for a table `w_1, …, w_m` of representatives of `Γ₀(N) \ SL₂(ℤ)` (classes = the
bottom rows in `ℙ¹(ℤ/N)`, [C] `def:cosetrepsN`) and a prime `p`,
`(H_{N,p} x)_i = ∑ ρ_n(adj h_j(ε)) x_j` over the pairs `(ε, j)`, `ε ∈ Mer_p`, `(c_j, d_j)ε` primitive mod `N`,
`j·ε = i`, with `adj h_j(ε) = w_{j·ε} (adj ε) w_j⁻¹`.

* `LevelTable`, `equivMod`, `clsOf` (`j·ε`), `termOf`, `heilbronnTerms` (the term list, in the C++ order: `ε` outer,
  `j` inner), `heilbronnApply` (the operator over `ZMod q`).  All executable on tuples.
* `heilbronn_spec : heilbronn_spec_stmt`: (1) the operator is the double sum of `eq:heilbronn` over `Mer_p × {0..m-1}`
  (`merelSet`, a `Finset`); (2) on a table of `SL₂(ℤ)` matrices every term matrix is `w_i (adj ε) adj(w_j)`, has
  determinant `p` and lower-left entry `≡ 0 mod N` ([C] `lem:cosetrepsN`(4): `adj h_j(ε)` is in `Δ₀(N)`), and its
  index `i` is `j·ε`: `(c_i, d_i) ≡ u·(c_j, d_j)ε mod N` for a unit `u`.

**Not stated** (stays with [C]): (1) `H(Rel_N) ⊆ Rel_N` and (2) `[H] = [A g A]` on `M_N / Rel_N` -- they need
[C] `lem:manin` and `prop:levelorbit` (the cokernel presentation), which the M4 cut allows to stay headers.
The checker on real C++ output is `GL2/Checkers/MerelCheck.lean`.
-/

namespace GL2.M4

open Matrix

/-- Integer `2 × 2` matrices as tuples `(a, b, c, d)` (the executable layer). -/
abbrev Q4 := ℤ × ℤ × ℤ × ℤ

def qmul (x y : Q4) : Q4 :=
  (x.1 * y.1 + x.2.1 * y.2.2.1, x.1 * y.2.1 + x.2.1 * y.2.2.2,
   x.2.2.1 * y.1 + x.2.2.2 * y.2.2.1, x.2.2.1 * y.2.1 + x.2.2.2 * y.2.2.2)

/-- The adjugate `(d, -b, -c, a)`. -/
def qadj (x : Q4) : Q4 := (x.2.2.2, -x.2.1, -x.2.2.1, x.1)

def qdet (x : Q4) : ℤ := x.1 * x.2.2.2 - x.2.1 * x.2.2.1

theorem toM2_qmul (x y : Q4) : toM2 (qmul x y) = toM2 x * toM2 y := by
  simp [toM2, qmul]

theorem toM2_qadj (x : Q4) : toM2 (qadj x) = (toM2 x).adjugate := by
  simp [toM2, qadj, Matrix.adjugate_fin_two_of]

theorem det_toM2 (x : Q4) : (toM2 x).det = qdet x := by
  simp [toM2, qdet, Matrix.det_fin_two_of]

/-- The bottom row times `ε`: `(c, d) ε`. -/
def rowMul (c d : ℤ) (ε : Q4) : ℤ × ℤ := (c * ε.1 + d * ε.2.2.1, c * ε.2.1 + d * ε.2.2.2)

/-- `(c, d)` primitive modulo `N`: `gcd(c, d, N) = 1`. -/
def primMod (N : ℕ) (v : ℤ × ℤ) : Bool := Nat.gcd (Int.gcd v.1 v.2) N == 1

/-- `(c, d) ≡ u·(c', d') mod N` for a unit `u` (the relation of `ℙ¹(ℤ/N)`). -/
def equivMod (N : ℕ) (v w : ℤ × ℤ) : Bool :=
  (List.range N).any fun u => Nat.gcd u N == 1 && (v.1 - u * w.1) % N == 0 && (v.2 - u * w.2) % N == 0

/-- A level-`N` table: `w_1, …, w_m` (stored 0-based). -/
structure LevelTable where
  N : ℕ
  reps : List Q4
  deriving Repr, DecidableEq

def LevelTable.m (T : LevelTable) : ℕ := T.reps.length
def LevelTable.w (T : LevelTable) (j : ℕ) : Q4 := T.reps.getD j (1, 0, 0, 1)

/-- `cls_N`: the (first) index whose bottom row is equivalent to `v`. -/
def clsOf (T : LevelTable) (v : ℤ × ℤ) : Option ℕ :=
  (List.range T.m).find? fun i => equivMod T.N ((T.w i).2.2.1, (T.w i).2.2.2) v

/-- One pair `(ε, j)`: `none` if inadmissible, else `(j·ε, j, w_{j·ε} (adj ε) adj(w_j))`. -/
def termOf (T : LevelTable) (ε : Q4) (j : ℕ) : Option (ℕ × ℕ × Q4) :=
  let v := rowMul (T.w j).2.2.1 (T.w j).2.2.2 ε
  if primMod T.N v then (clsOf T v).map fun i => (i, j, qmul (qmul (T.w i) (qadj ε)) (qadj (T.w j))) else none

/-- **The term list of `eq:heilbronn`** (`ε ∈ Mer_p` outer, `j` inner: the C++ `HeilbronnOperator` order). -/
def heilbronnTerms (T : LevelTable) (p : ℕ) : List (ℕ × ℕ × Q4) :=
  (merelList p).flatMap fun ε => (List.range T.m).filterMap (termOf T ε)

/-- `ρ_n` of an integer matrix, over `ZMod q`. -/
noncomputable def rhoZ (n q : ℕ) (x : Q4) : Matrix (Fin (n + 1)) (Fin (n + 1)) (ZMod q) :=
  GL2.Exec.rhoMat n ((toM2 x).map (Int.cast : ℤ → ZMod q))

/-- **`H_{N,p}` over `ZMod q`**: `(H x)_i = ∑_{terms (i, j, h)} ρ_n(h) x_j`. -/
noncomputable def heilbronnApply (n q : ℕ) (T : LevelTable) (p : ℕ) (x : ℕ → Fin (n + 1) → ZMod q) (i : ℕ) :
    Fin (n + 1) → ZMod q :=
  ((heilbronnTerms T p).map fun t => if t.1 = i then rhoZ n q t.2.2 *ᵥ x t.2.1 else 0).sum

/-! ### `Mer_p` without repetitions -/

lemma nodup_flatMap_key {α β : Type*} {l : List α} {f : α → List β} (key : β → α) (hl : l.Nodup)
    (hf : ∀ a ∈ l, (f a).Nodup) (hk : ∀ a ∈ l, ∀ b ∈ f a, key b = a) : (l.flatMap f).Nodup := by
  rw [List.nodup_flatMap]
  refine ⟨hf, hl.pairwise_of_forall_ne fun a ha b hb hab => ?_⟩
  show List.Disjoint (f a) (f b)
  rw [List.disjoint_left]
  intro x hxa hxb
  exact hab ((hk a ha x hxa).symm.trans (hk b hb x hxb))

theorem merelList_nodup (n : ℕ) : (merelList n).Nodup := by
  unfold merelList
  refine nodup_flatMap_key (fun e => (e.1 - 1).toNat) List.nodup_range (fun a' _ => ?_) ?_
  · refine nodup_flatMap_key (fun e => (e.2.2.2 - 1).toNat) List.nodup_range (fun d' _ => ?_) ?_
    · refine nodup_flatMap_key (fun e => e.2.1.toNat) List.nodup_range (fun b _ => ?_) ?_
      · refine List.nodup_range.filterMap ?_
        intro c c' e hc hc'
        simp only [Option.mem_def] at hc hc'
        split at hc <;> split at hc' <;> simp only [Option.some.injEq, reduceCtorEq] at hc hc'
        subst hc
        simp only [Prod.mk.injEq] at hc'
        omega
      · intro b _ e he
        simp only [List.mem_filterMap, List.mem_range, Option.ite_none_right_eq_some, Option.some.injEq] at he
        obtain ⟨c, _, _, rfl⟩ := he
        simp
    · intro d' _ e he
      simp only [List.mem_flatMap, List.mem_filterMap, List.mem_range, Option.ite_none_right_eq_some,
        Option.some.injEq] at he
      obtain ⟨b, _, c, _, _, rfl⟩ := he
      simp
  · intro a' _ e he
    simp only [List.mem_flatMap, List.mem_filterMap, List.mem_range, Option.ite_none_right_eq_some,
      Option.some.injEq] at he
    obtain ⟨d', _, b, _, c, _, _, rfl⟩ := he
    simp

theorem toM2_injective : Function.Injective toM2 := by
  rintro ⟨a, b, c, d⟩ ⟨a', b', c', d'⟩ h
  have h00 := congrFun (congrFun h 0) 0
  have h01 := congrFun (congrFun h 0) 1
  have h10 := congrFun (congrFun h 1) 0
  have h11 := congrFun (congrFun h 1) 1
  simp only [toM2, mk00, mk01, mk10, mk11] at h00 h01 h10 h11
  rw [h00, h01, h10, h11]

/-- A sum over `merelList` is the sum over the finite set `Mer_n`. -/
theorem merelList_sum {M : Type*} [AddCommMonoid M] (n : ℕ) (f : M2Z → M) :
    ((merelList n).map fun e => f (toM2 e)).sum = ∑ ε ∈ merelSet n, f ε := by
  classical
  rw [merelSet, List.sum_toFinset _ ((merelList_nodup n).map toM2_injective), List.map_map]
  rfl

/-! ### The spec -/

lemma sum_filterMap {α β M : Type*} [AddCommMonoid M] (l : List α) (g : α → Option β) (h : β → M) :
    ((l.filterMap g).map h).sum = (l.map fun a => ((g a).map h).getD 0).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    rw [List.filterMap_cons]
    cases hg : g a with
    | none => simp [ih, hg]
    | some b => simp [ih, hg]

lemma sum_flatMap {α β M : Type*} [AddCommMonoid M] (l : List α) (f : α → List β) (h : β → M) :
    ((l.flatMap f).map h).sum = (l.map fun a => ((f a).map h).sum).sum := by
  induction l with
  | nil => simp
  | cons a l ih => simp [List.flatMap_cons, ih]

/-- `ρ_n(w_{j·ε} (adj ε) adj(w_j)) x_j` when `(ε, j)` is admissible and `j·ε = i`, else `0`. -/
noncomputable def heilbronnSummand (n q : ℕ) (T : LevelTable) (x : ℕ → Fin (n + 1) → ZMod q) (i : ℕ) (ε : M2Z)
    (j : ℕ) : Fin (n + 1) → ZMod q :=
  ((termOf T (ε 0 0, ε 0 1, ε 1 0, ε 1 1) j).map fun t => if t.1 = i then rhoZ n q t.2.2 *ᵥ x t.2.1 else 0).getD 0

lemma toM2_eta (ε : M2Z) : toM2 (ε 0 0, ε 0 1, ε 1 0, ε 1 1) = ε := by
  ext i j; fin_cases i <;> fin_cases j <;> rfl

lemma toM2_entries (e : Q4) : ((toM2 e) 0 0, (toM2 e) 0 1, (toM2 e) 1 0, (toM2 e) 1 1) = e := by
  obtain ⟨a, b, c, d⟩ := e; rfl

/-- **(1) of the spec**: `heilbronnApply` is the double sum of `eq:heilbronn` over `Mer_p × {0, …, m-1}`. -/
theorem heilbronnApply_eq_sum (n q : ℕ) (T : LevelTable) (p : ℕ) (x : ℕ → Fin (n + 1) → ZMod q) (i : ℕ) :
    heilbronnApply n q T p x i =
      ∑ ε ∈ merelSet p, ∑ j ∈ Finset.range T.m, heilbronnSummand n q T x i ε j := by
  rw [heilbronnApply, heilbronnTerms, sum_flatMap]
  simp only [sum_filterMap]
  rw [← merelList_sum p (fun ε => ∑ j ∈ Finset.range T.m, heilbronnSummand n q T x i ε j)]
  congr 1

/-- A table of `SL₂(ℤ)` matrices. -/
def LevelTable.Valid (T : LevelTable) : Prop := ∀ j < T.m, qdet (T.w j) = 1

lemma find?_some {l : List ℕ} {P : ℕ → Bool} {i : ℕ} (h : l.find? P = some i) : P i = true ∧ i ∈ l :=
  ⟨List.find?_some h, List.mem_of_find?_eq_some h⟩

lemma equivMod_spec {N : ℕ} {v w : ℤ × ℤ} (h : equivMod N v w = true) :
    ∃ u : ℕ, Nat.Coprime u N ∧ (N : ℤ) ∣ v.1 - u * w.1 ∧ (N : ℤ) ∣ v.2 - u * w.2 := by
  simp only [equivMod, List.any_eq_true, List.mem_range, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨u, _, ⟨hu, h1⟩, h2⟩ := h
  exact ⟨u, hu, Int.dvd_of_emod_eq_zero h1, Int.dvd_of_emod_eq_zero h2⟩

/-- The statement of the spec (`prop:heilbronn` as an executable operator over `ZMod q`): the double-sum form, and on
a table of `SL₂(ℤ)` matrices every term `(i, j, h)` comes from an admissible pair `(ε, j)`, `ε ∈ Mer_p`, with
`h = w_i (adj ε) adj(w_j)` of determinant `p`, lower-left entry `≡ 0 mod N`, and `i = j·ε` in `ℙ¹(ℤ/N)`. -/
def heilbronn_spec_stmt : Prop :=
  ∀ (n q : ℕ) (T : LevelTable) (p : ℕ),
    (∀ (x : ℕ → Fin (n + 1) → ZMod q) (i : ℕ),
      heilbronnApply n q T p x i = ∑ ε ∈ merelSet p, ∑ j ∈ Finset.range T.m, heilbronnSummand n q T x i ε j) ∧
    (T.Valid → ∀ t ∈ heilbronnTerms T p, ∃ ε ∈ merelSet p, t.2.1 < T.m ∧ t.1 < T.m ∧
      toM2 t.2.2 = toM2 (T.w t.1) * ε.adjugate * (toM2 (T.w t.2.1)).adjugate ∧
      (toM2 t.2.2).det = p ∧ (T.N : ℤ) ∣ toM2 t.2.2 1 0 ∧
      primMod T.N (rowMul (T.w t.2.1).2.2.1 (T.w t.2.1).2.2.2 (ε 0 0, ε 0 1, ε 1 0, ε 1 1)) = true ∧
      ∃ u : ℕ, Nat.Coprime u T.N ∧
        (T.N : ℤ) ∣ (T.w t.1).2.2.1 - u * (rowMul (T.w t.2.1).2.2.1 (T.w t.2.1).2.2.2 (ε 0 0, ε 0 1, ε 1 0, ε 1 1)).1 ∧
        (T.N : ℤ) ∣ (T.w t.1).2.2.2 - u * (rowMul (T.w t.2.1).2.2.1 (T.w t.2.1).2.2.2 (ε 0 0, ε 0 1, ε 1 0, ε 1 1)).2)

/-- **`heilbronn_spec`** ([C] `prop:heilbronn`, `eq:heilbronn`, `lem:cosetrepsN`(4); executable over `ZMod q`). -/
theorem heilbronn_spec : heilbronn_spec_stmt := by
  intro n q T p
  refine ⟨heilbronnApply_eq_sum n q T p, fun hT t ht => ?_⟩
  simp only [heilbronnTerms, List.mem_flatMap, List.mem_filterMap, List.mem_range] at ht
  obtain ⟨e, he, j, hj, hterm⟩ := ht
  simp only [termOf] at hterm
  split_ifs at hterm with hprim
  obtain ⟨i, hi, rfl⟩ := Option.map_eq_some_iff.mp hterm
  obtain ⟨hequiv, hiR⟩ := find?_some hi
  rw [List.mem_range] at hiR
  obtain ⟨u, hu, h1, h2⟩ := equivMod_spec hequiv
  have hmer : toM2 e ∈ merelSet p := by
    simp only [merelSet, List.mem_toFinset, List.mem_map]; exact ⟨e, he, rfl⟩
  refine ⟨toM2 e, hmer, ?_⟩
  dsimp only
  rw [toM2_entries]
  refine ⟨hj, hiR, ?_, ?_, ?_, hprim, u, hu, h1, h2⟩
  · simp only [toM2_qmul, toM2_qadj]
  · have hdi := hT i hiR
    have hdj := hT j hj
    have hp : qdet e = p := by
      have := (mem_merelList p e).mp he
      simp only [qdet]; linarith [this.1]
    rw [toM2_qmul, toM2_qmul, toM2_qadj, toM2_qadj, Matrix.det_mul, Matrix.det_mul, Matrix.det_adjugate,
      Matrix.det_adjugate, det_toM2, det_toM2, det_toM2, hdi, hdj, hp]
    simp
  · obtain ⟨k1, hk1⟩ := h1
    obtain ⟨k2, hk2⟩ := h2
    obtain ⟨a, b, c, d⟩ := e
    generalize T.w j = W at hk1 hk2 ⊢
    generalize T.w i = V at hk1 hk2 ⊢
    obtain ⟨wa, wb, wc, wd⟩ := W
    obtain ⟨va, vb, vc, vd⟩ := V
    simp only [rowMul] at hk1 hk2
    simp only [toM2, qmul, qadj, mk10]
    refine ⟨k1 * (d * wd + b * wc) + k2 * (-c * wd - a * wc), ?_⟩
    linear_combination (d * wd + b * wc) * hk1 + (-c * wd - a * wc) * hk2

end GL2.M4
