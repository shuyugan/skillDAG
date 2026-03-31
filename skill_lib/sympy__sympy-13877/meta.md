# Fix symbolic algorithm NaN/crash

Diagnose and repair failures in symbolic/numeric computation routines where intermediate results become NaN/invalid or trigger comparison/type errors due to incorrect simplification, pivoting, or exceptional-case handling. Apply this when a core math/linear-algebra routine returns NaN unexpectedly or crashes on valid symbolic inputs.

## Trigger Conditions

- A deterministic computation sometimes returns NaN/Infinity or crashes with a type/comparison error on symbolic or mixed symbolic-numeric inputs
- A divide-by-zero/invalid-pivot or “invalid comparison” exception appears deep inside simplification/canonicalization code during an algorithm’s intermediate steps
- An algorithm relies on intermediate simplification/exact-division, and incorrect or missing simplification is suspected to corrupt later pivots/branches
- An alternate algorithm exists that can compute the same result more robustly, suggesting a fallback strategy for exceptional cases
