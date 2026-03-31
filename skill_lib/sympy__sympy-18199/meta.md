# Add modular zero-case guard

Addresses bugs in modular equation solvers where a degenerate residue class (typically 0 modulo the modulus) is not handled consistently across algorithm branches. Apply when a solver routes special inputs into subroutines that assume invertible/nonzero residues, leading to missing solutions or runtime errors.

## Trigger Conditions

- A modular root/solver function returns incomplete results or raises for inputs where the right-hand side is congruent to 0 modulo the modulus
- The implementation uses multiple algorithm branches with differing domain assumptions (e.g., requires multiplicative inverses, discrete logs, or group operations)
- At least one internal helper implicitly assumes the residue is nonzero/unit, but the public API accepts general residues
- Reported issue mentions “missing 0 solution”, “log of 0”, or failures that depend on parameter relationships that select a specialized branch
