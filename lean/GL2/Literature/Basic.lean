import Mathlib

/-!
# L1, the literature register: shared vocabulary (L16-3)

Every file under `GL2/Literature/` states external (cited, not proved) mathematics as a
`structure ... : Prop` ("hypothesis structure"). Conventions, [LP] §2.6 / §3.2:

* the docstring of each structure carries the **verbatim quotation** of the source and its **locator**
  (work, theorem number, printed page), read from the library scan; where the source is not in the
  library the docstring says `NOT IN LIBRARY` and the structure is marked `UNVERIFIED`;
* a structure is a *hypothesis*: nothing here is an `axiom`, and no declaration of this folder is
  `sorry`. A theorem that uses a literature fact takes the structure as an explicit argument, so
  `#print axioms` of such a theorem is clean and the dependency is visible in its type;
* where the source's objects are not yet defined in Lean (Hecke algebras of mod `p` forms, Hida's
  ordinary parts, Hecke algebras of `J_0(N)`, Galois representations), the structure is stated over
  **interface data** passed as parameters, and the L2 development supplies the real objects. This
  fixes the statement's *shape* and locator, not the identification of the data; each docstring says
  which data are interface data.

`VGe p m x` is "`v_p(x) ≥ m`" for `x : ℚ`, written with `num` / `den` so that instances can be
decided. It avoids `padicValRat`, which is `0` at `0`.
-/

namespace GL2.Literature

/-- `v_p(x) ≥ m` for a rational `x` (with `v_p(0) = ∞`): `p ∤ den x` and `p^m ∣ num x`. -/
def VGe (p m : ℕ) (x : ℚ) : Prop := ¬ p ∣ x.den ∧ ((p : ℤ) ^ m ∣ x.num)

/-- `a ≡ b (mod p^m)` for rationals that are `p`-integral. -/
def CongModPow (p m : ℕ) (a b : ℚ) : Prop := VGe p m (a - b)

theorem VGe_zero (p m : ℕ) (hp : 2 ≤ p) : VGe p m 0 := by
  refine ⟨?_, by simp⟩
  intro h
  have : p ∣ 1 := by simpa using h
  exact absurd (Nat.le_of_dvd one_pos this) (by omega)

end GL2.Literature
