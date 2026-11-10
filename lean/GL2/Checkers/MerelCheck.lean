import GL2.M4.Heilbronn

/-!
# M4-i (slot 20, L20-4): the Heilbronn–Merel checker on real C++ output (L3-a / L3-b)

`merelCheck` reads one `HECKE_DUMP_MEREL` dump of `HeckeLevelNValidation` (the C++ `HeilbronnOperator` at one
`(N, n, p)` over `F_q`; turned into Lean data by `lean/tools/merel_dump_to_lean.py`) and re-derives everything from the
dumped table alone:

* the table: every `w_j ∈ SL₂(ℤ)`, bottom rows pairwise inequivalent in `ℙ¹(ℤ/N)`, every primitive class hit;
* `Mer_p` equals `merelList p` (same order);
* the term list `(j·ε, j, adj h_j(ε))` equals `heilbronnTerms` (same order);
* every dumped block is `ρ_n(adj h mod q) mod q` (`GL2.Exec.rhoNat`);
* the dumped dense `D × D` matrix is the sum of the blocks at their places.

`merelCheck_sound`: if the check passes, the dumped C++ matrix, read in `ZMod q`, **is** the matrix of the spec operator
`heilbronnApply` (`GL2/M4/Heilbronn.lean`, which `heilbronn_spec` ties to `eq:heilbronn`) built from the dumped table.
-/

namespace GL2.M4

/-- One dumped operator.  `terms`: `(dst, src, adj h, block)` with 0-based `dst`, `src` and the `(n+1)²` block
entries row-major. -/
structure MerelDump where
  N : ℕ
  n : ℕ
  p : ℕ
  q : ℕ
  reps : List Q4
  merel : List Q4
  terms : List (ℕ × ℕ × Q4 × List ℕ)
  dense : List (List ℕ)
  deriving Repr

def MerelDump.table (D : MerelDump) : LevelTable := ⟨D.N, D.reps⟩

/-- The non-negative residue of an integer, as a natural number. -/
def resNat (q : ℕ) (a : ℤ) : ℕ := (a % (q : ℤ)).toNat

/-- The bottom row of `w_j`. -/
def LevelTable.row (T : LevelTable) (j : ℕ) : ℤ × ℤ := ((T.w j).2.2.1, (T.w j).2.2.2)

/-- The table checks: `SL₂(ℤ)`, pairwise inequivalent bottom rows, every primitive class hit. -/
def tableCheck (T : LevelTable) : Bool :=
  (List.range T.m).all (fun j => qdet (T.w j) == 1) &&
  (List.range T.m).all (fun j => (List.range T.m).all fun k => j == k || !equivMod T.N (T.row j) (T.row k)) &&
  (List.range T.N).all (fun c => (List.range T.N).all fun d =>
    !primMod T.N ((c : ℤ), (d : ℤ)) || (clsOf T ((c : ℤ), (d : ℤ))).isSome)

/-- One block: shape, range, and `ρ_n(adj h mod q) mod q` entrywise. -/
def blockCheck (n q : ℕ) (t : ℕ × ℕ × Q4 × List ℕ) : Bool :=
  let x := t.2.2.1
  t.2.2.2.length == (n + 1) * (n + 1) &&
  (List.range (n + 1)).all fun r => (List.range (n + 1)).all fun c =>
    t.2.2.2.getD (r * (n + 1) + c) 0 ==
      GL2.Exec.rhoNat n (resNat q x.1) (resNat q x.2.1) (resNat q x.2.2.1) (resNat q x.2.2.2) r c % q

/-- The `(I, J)` entry of the sum of the blocks at their places. -/
def denseOf (n q : ℕ) (terms : List (ℕ × ℕ × Q4 × List ℕ)) (I J : ℕ) : ℕ :=
  ((terms.map fun t => if t.1 = I / (n + 1) ∧ t.2.1 = J / (n + 1) then
      t.2.2.2.getD (I % (n + 1) * (n + 1) + J % (n + 1)) 0 else 0).sum) % q

/-- `I mod (n+1)` as an index of `Fin (n+1)`. -/
def fmod (n I : ℕ) : Fin (n + 1) := ⟨I % (n + 1), Nat.mod_lt _ (Nat.succ_pos _)⟩

def MerelDump.dim (D : MerelDump) : ℕ := D.reps.length * (D.n + 1)

/-- **The checker.** -/
def merelCheck (D : MerelDump) : Bool :=
  D.q != 0 && tableCheck D.table && D.merel == merelList D.p &&
  D.terms.map (fun t => (t.1, t.2.1, t.2.2.1)) == heilbronnTerms D.table D.p &&
  D.terms.all (blockCheck D.n D.q) &&
  D.dense.length == D.dim &&
  (List.range D.dim).all fun I => (List.range D.dim).all fun J =>
    (D.dense.getD I []).getD J 0 == denseOf D.n D.q D.terms I J

/-! ### Soundness -/

lemma resNat_cast {q : ℕ} (hq : q ≠ 0) (a : ℤ) : ((resNat q a : ℕ) : ZMod q) = (a : ZMod q) := by
  have h0 : 0 ≤ a % (q : ℤ) := Int.emod_nonneg a (by exact_mod_cast hq)
  rw [resNat, ← Int.cast_natCast, Int.toNat_of_nonneg h0, ZMod.intCast_mod]

lemma rhoZ_apply (n q : ℕ) (x : Q4) (r c : Fin (n + 1)) :
    rhoZ n q x r c = GL2.Exec.rhoEntry n (x.1 : ZMod q) (x.2.1 : ZMod q) (x.2.2.1 : ZMod q) (x.2.2.2 : ZMod q) r c := by
  simp [rhoZ, GL2.Exec.rhoMat, toM2]

lemma blockCheck_sound {n q : ℕ} (hq : q ≠ 0) {t : ℕ × ℕ × Q4 × List ℕ} (h : blockCheck n q t = true)
    (r c : Fin (n + 1)) : ((t.2.2.2.getD (r * (n + 1) + c) 0 : ℕ) : ZMod q) = rhoZ n q t.2.2.1 r c := by
  simp only [blockCheck, Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at h
  rw [h.2 r r.2 c c.2, ZMod.natCast_mod, GL2.Exec.cast_rhoNat, resNat_cast hq, resNat_cast hq, resNat_cast hq,
    resNat_cast hq, rhoZ_apply]

/-- What `merelCheck` certifies (see `merelCheck_sound`). -/
noncomputable def MerelDump.Sound (D : MerelDump) : Prop :=
    tableCheck D.table = true ∧ D.merel = merelList D.p ∧
    D.terms.map (fun t => (t.1, t.2.1, t.2.2.1)) = heilbronnTerms D.table D.p ∧
    ∀ I < D.dim, ∀ J < D.dim,
      (((D.dense.getD I []).getD J 0 : ℕ) : ZMod D.q) =
        ((heilbronnTerms D.table D.p).map fun t => if t.1 = I / (D.n + 1) ∧ t.2.1 = J / (D.n + 1) then
          rhoZ D.n D.q t.2.2 (fmod D.n I) (fmod D.n J) else 0).sum

/-- **Soundness of `merelCheck`**: the table facts, `Mer_p` and the term list are the spec's, and the dumped dense
matrix, read in `ZMod q`, is entrywise the matrix of `heilbronnApply` on the dumped table: entry `(I, J)` is
`∑_{terms (i, j, h) with i = I / (n+1), j = J / (n+1)} ρ_n(h)_{I mod (n+1), J mod (n+1)}`. -/
theorem merelCheck_sound (D : MerelDump) (h : merelCheck D = true) : D.Sound := by
  simp only [merelCheck, Bool.and_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, List.all_eq_true, List.mem_range] at h
  obtain ⟨⟨⟨⟨⟨⟨hq, htab⟩, hmer⟩, hterms⟩, hblocks⟩, _⟩, hdense⟩ := h
  refine ⟨htab, hmer, hterms, fun I hI J hJ => ?_⟩
  rw [hdense I hI J hJ, denseOf, ZMod.natCast_mod, Nat.cast_list_sum, ← hterms, List.map_map, List.map_map]
  congr 1
  refine List.map_congr_left fun t ht => ?_
  simp only [Function.comp_apply]
  split_ifs
  · exact blockCheck_sound hq (hblocks t ht) (fmod D.n I) (fmod D.n J)
  · simp

end GL2.M4
