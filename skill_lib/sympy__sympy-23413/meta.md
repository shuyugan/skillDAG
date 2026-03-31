# Fix Truncation in Matrix Normal Form

Diagnose and repair bugs in canonical matrix form algorithms (e.g., Hermite/Smith/row-echelon variants) where the returned result has the wrong shape due to incorrect rank/pivot detection. Apply when rectangular (especially tall) matrices unexpectedly lose rows/columns or are misclassified as rank-deficient.

## Trigger Conditions

- A matrix normal-form routine returns a result with fewer rows/columns than mathematically expected
- Failures occur primarily for rectangular inputs (e.g., more rows than columns) or when some bottom rows are zero
- The algorithm uses a pivot-scan loop plus a final slice/truncation step based on computed pivots/rank
- Transformations like transpose/row-vs-column conversion expose an output-shape discrepancy
