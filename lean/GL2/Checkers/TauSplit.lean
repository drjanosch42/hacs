import GL2.Checkers.TauCheck

/-!
# L18-3: `tauCheck` in pieces (kernel memory)

`decide +kernel` on `tauCheck D` for the largest grid table (`(p, ν) = (3, 2)`, `m = 12`, 276 slots) peaks at
≈ 23 GB, above the 20 GB `lake` budget: the kernel's cache lives as long as the declaration.  This file splits
the checker into a **global** part (coset datum, closure, completeness, `4N² < 2^31`, regime) and a
**per-slot** part over any sublist of the slots, and proves that the global part together with the per-slot
part on the `K` residue classes `s.i % K = r` gives `tauCheck D = true` (`tauCheck_of_mod`).  A data module then
proves `K + 1` small declarations instead of one large one.  Nothing about `TauSpec` changes.
-/

namespace GL2.TauChk

/-- The global conjuncts of `tauCheck` (no per-slot quantifier). -/
def tauGlobalCheck (D : TauData) : Bool :=
  decide (D.g = ⟨1, 0, 0, D.N⟩ ∧ 0 < D.N ∧ D.s.length = D.m ∧ (∀ x ∈ D.s, x.det = 1) ∧
    (∀ i ∈ List.range D.m, ∀ j ∈ List.range D.m, i ≠ j →
      (D.sAt (i + 1) * (D.sAt (j + 1)).adj).c % (D.N : ℤ) ≠ 0) ∧
    (∀ x ∈ D.simplices, D.trSet x.1 x.2.1 (D.rep x.1 x.2.1) ≠ [])) &&
  D.closedB &&
  decide (D.slotKeys.Nodup ∧ D.slotKeys.Perm D.wantKeys) &&
  decide (4 * D.N ^ 2 < 2 ^ 31) &&
  decide (D.regime = "Z")

/-- The per-slot conjuncts of `tauCheck`, over the slots `ss` (the model and the helper still read all of `D`). -/
def tauSlotsCheck (D : TauData) (ss : List Slot) : Bool :=
  decide (∀ s ∈ ss, ∀ t ∈ s.terms,
    t.coeff ≠ 0 ∧ t.g.det = 1 ∧ D.canon t.g s.q t.sigma = ((s.q, t.sigma, t.g), 1)) &&
  decide (∀ s ∈ ss, s.q = 0 → (s.terms.map Term.coeff).sum = 1) &&
  ss.all (fun s => !(s.q = 1 ∨ s.q = 2) || chainEqB (D.bdry s.q s.terms) (D.tauBdry s.i s.q s.sigma)) &&
  ss.all (fun s => D.trs.all fun t => !(t.q = s.q ∧ t.s1 = s.sigma) ||
    chainEqB (D.canonTerms s.q 1 s.terms) (D.tauT1 (permSign t.perm s.q) (D.sAt s.i * t.g) s.q t.s2)) &&
  decide (∀ s ∈ ss, ∀ t ∈ s.terms, t.g.maxAbs < 2 ^ 31)

/-- What `tauSlotsCheck` gives for one slot of `ss`. -/
theorem tauSlotsCheck_slot {D : TauData} {ss : List Slot} (h : tauSlotsCheck D ss = true) {s : Slot}
    (hs : s ∈ ss) :
    (∀ t ∈ s.terms, t.coeff ≠ 0 ∧ t.g.det = 1 ∧ D.canon t.g s.q t.sigma = ((s.q, t.sigma, t.g), 1)) ∧
    (s.q = 0 → (s.terms.map Term.coeff).sum = 1) ∧
    (s.q = 1 ∨ s.q = 2 → Chain.eval (D.bdry s.q s.terms) = Chain.eval (D.tauBdry s.i s.q s.sigma)) ∧
    (∀ t ∈ D.trs, t.q = s.q → t.s1 = s.sigma →
      Chain.eval (D.canonTerms s.q 1 s.terms) =
        Chain.eval (D.tauT1 (permSign t.perm s.q) (D.sAt s.i * t.g) s.q t.s2)) ∧
    (∀ t ∈ s.terms, t.g.maxAbs < 2 ^ 31) := by
  simp only [tauSlotsCheck, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩ := h
  refine ⟨h1 s hs, h2 s hs, fun hq => ?_, fun t ht hq h1' => ?_, h5 s hs⟩
  · have := List.all_eq_true.mp h3 s hs
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at this
    rcases this with hn | hc
    · exact absurd hq hn
    · exact (chainEqB_iff _ _).mp hc
  · have := List.all_eq_true.mp (List.all_eq_true.mp h4 s hs) t ht
    simp only [Bool.or_eq_true, Bool.not_eq_true', decide_eq_false_iff_not] at this
    rcases this with hn | hc
    · exact absurd ⟨hq, h1'⟩ hn
    · exact (chainEqB_iff _ _).mp hc

/-- **The split.** The global part and the per-slot part on every residue class `s.i % K = r` (`r < K`)
give `tauCheck D = true`. -/
theorem tauCheck_of_mod (D : TauData) (K : ℕ) (hK : 0 < K) (hg : tauGlobalCheck D = true)
    (hc : ∀ r < K, tauSlotsCheck D (D.slots.filter fun s => s.i % K = r) = true) : tauCheck D = true := by
  have slot : ∀ s ∈ D.slots, _ := fun s hs =>
    tauSlotsCheck_slot (hc (s.i % K) (Nat.mod_lt _ hK)) (List.mem_filter.mpr ⟨hs, by simp⟩)
  simp only [tauGlobalCheck, Bool.and_eq_true, decide_eq_true_eq] at hg
  obtain ⟨⟨⟨⟨h1, h0⟩, h2⟩, h8a⟩, h9⟩ := hg
  refine tauCheck_complete D ⟨h1, h0, h2, fun s hs t ht => (slot s hs).1 t ht, fun s hs => (slot s hs).2.1,
    fun s hs => (slot s hs).2.2.1, fun s hs => (slot s hs).2.2.2.1,
    ⟨fun s hs => (slot s hs).2.2.2.2, h8a⟩, h9⟩

end GL2.TauChk
