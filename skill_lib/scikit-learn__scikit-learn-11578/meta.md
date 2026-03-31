# Propagate Configuration in Scoring Helpers

Use this workflow when cross-validation or evaluation code constructs a temporary estimator/model for scoring and the resulting metrics do not reflect the caller’s intended configuration. It helps identify parameter-propagation gaps, implement a minimal fix, and validate the corrected scoring behavior through reproductions and tests.

## Trigger Conditions

- A probability-based (or otherwise method-dependent) metric differs between a direct model evaluation and a cross-validated/evaluation helper path
- A helper constructs an internal/temporary estimator for scoring instead of reusing the already-configured estimator instance
- A behavior switch controlled by a configuration flag (e.g., probability computation mode) appears to be ignored during scoring
- Scores change depending on which prediction method is implicitly used by the scorer (e.g., probability vs decision values), suggesting mismatched estimator configuration
