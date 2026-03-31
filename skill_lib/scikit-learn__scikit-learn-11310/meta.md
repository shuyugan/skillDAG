# Add Post-Selection Refit Timing

Implement and expose wall-clock timing for a final “refit on full data” step that occurs after a model selection or optimization routine completes. Apply this when users need to distinguish the time spent evaluating candidates from the time spent training the final chosen model, especially when candidate evaluation is parallelized.

## Trigger Conditions

- A pipeline runs candidate evaluation (e.g., cross-validation / hyperparameter search) and then optionally retrains the chosen model on all available data
- Users need a first-class metric for the duration of the final refit step, independent of parallel execution used during evaluation
- The code already reports per-candidate or per-fold fit/score times but lacks timing for the final selected-model training
- The refit step is conditional (e.g., can be disabled, or is controlled differently under multi-metric scoring)
