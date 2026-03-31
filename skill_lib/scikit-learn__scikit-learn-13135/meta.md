# Fix Non-Monotonic Derived Bins

Applies when a transformation or discretization routine derives ordered thresholds (e.g., bin edges) from a model output (e.g., cluster centers) but downstream logic requires those thresholds to be monotonic. Use this workflow to reproduce the failure, trace the derivation path, enforce a deterministic ordering invariant, and validate via broader checks and automated tests.

## Trigger Conditions

- A downstream routine errors because an input array of thresholds/edges must be monotonically increasing or decreasing
- Derived thresholds are computed from model outputs that are not guaranteed to be ordered (e.g., centroids, categories, quantiles from unstable steps)
- The failure appears only for certain parameter regimes (e.g., many bins/clusters relative to unique values) or specific datasets
- A data-to-bins mapping function behaves inconsistently or throws when presented with non-sorted boundaries
