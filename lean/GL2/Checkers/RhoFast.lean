import GL2.Checkers.RhoExec

/-!
# L17-4 repair: a kernel-cheap form of `rhoCheck` (proved equal)

[GT] `chk:rhofast`. `rhoNat` (L16-4) sums over all pairs `(s, r)` with `s + r = t` and calls `Nat.choose`,
whose structural recursion the kernel unfolds at a cost exponential in `n` (and with a large `whnf` cache:
`decide +kernel` on `rhoCheck 16 13 …` alone takes ≈ 44 s and 13.7 GB). `rhoNatFast` sums over `s` only
(`r = t - s`) and computes the binomial coefficient as `descFactorial / factorial` (linear recursions, the
multiplications and the division GMP-accelerated in the kernel). `rhoCheckFast_eq` proves the two checkers
equal, so a `decide +kernel` run of `rhoCheckFast` proves the `rhoCheck` statement that `rhoCheck_sound`
consumes. No `native_decide`.
-/

namespace GL2.Exec

/-- The binomial coefficient by `descFactorial / factorial` (`Nat.choose_eq_descFactorial_div_factorial`). -/
def chooseFast (n k : ℕ) : ℕ := n.descFactorial k / k.factorial

theorem chooseFast_eq (n k : ℕ) : chooseFast n k = n.choose k :=
  (Nat.choose_eq_descFactorial_div_factorial n k).symm

/-- `rhoNat` with the inner sum collapsed to its one surviving term `r = t - s`. -/
def rhoNatFast (n a b c d t j : ℕ) : ℕ :=
  ((List.range (n - j + 1)).map fun s =>
    if s ≤ t ∧ t - s ≤ j then
      chooseFast (n - j) s * a ^ (n - j - s) * c ^ s * chooseFast j (t - s) * b ^ (j - (t - s)) * d ^ (t - s)
    else 0).sum

theorem sum_range_ite_add_eq (s t j : ℕ) (g : ℕ → ℕ) :
    ((List.range (j + 1)).map fun r => if s + r = t then g r else 0).sum =
      if s ≤ t ∧ t - s ≤ j then g (t - s) else 0 := by
  rw [list_sum_range_map]
  by_cases h : s ≤ t ∧ t - s ≤ j
  · rw [if_pos h, Finset.sum_eq_single (t - s)]
    · rw [if_pos (by omega)]
    · intro r _ hr
      rw [if_neg (by omega)]
    · intro hn
      exact absurd (Finset.mem_range.mpr (by omega)) hn
  · rw [if_neg h]
    refine Finset.sum_eq_zero fun r hr => ?_
    have := Finset.mem_range.mp hr
    rw [if_neg (by omega)]

theorem rhoNatFast_eq (n a b c d t j : ℕ) : rhoNatFast n a b c d t j = rhoNat n a b c d t j := by
  unfold rhoNatFast rhoNat
  congr 1
  refine List.map_congr_left fun s _ => ?_
  rw [sum_range_ite_add_eq s t j
    (fun r => (n - j).choose s * a ^ (n - j - s) * c ^ s * j.choose r * b ^ (j - r) * d ^ r)]
  simp only [chooseFast_eq]

def rhoCheckOneFast (n m : ℕ) (D : RhoDump) : Bool :=
  D.rows.length == n + 1 && D.rows.all (fun row => row.length == n + 1 && row.all (· < m)) &&
    (List.range (n + 1)).all fun t => (List.range (n + 1)).all fun j =>
      D.entry t j == rhoNatFast n D.key.1 D.key.2.1 D.key.2.2.1 D.key.2.2.2 t j % m

def rhoCheckFast (n m : ℕ) (Ds : List RhoDump) : Bool := Ds.all (rhoCheckOneFast n m)

/-- The fast checker **is** the L16-4 checker. -/
theorem rhoCheckFast_eq (n m : ℕ) (Ds : List RhoDump) : rhoCheckFast n m Ds = rhoCheck n m Ds := by
  have h : rhoCheckOneFast n m = rhoCheckOne n m := by
    funext D
    simp only [rhoCheckOneFast, rhoCheckOne, rhoNatFast_eq]
  simp only [rhoCheckFast, rhoCheck, h]

end GL2.Exec
