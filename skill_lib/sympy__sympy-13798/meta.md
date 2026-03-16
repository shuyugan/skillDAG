# Relax Enumerated Option to Custom Value

Applies when a library/API option is artificially limited to a fixed set of enumerated values but users need to supply arbitrary custom values. This workflow updates validation/mapping logic to preserve existing presets while allowing pass-through of user-provided strings, and then validates behavior and documentation.

## Trigger Conditions

- A configuration/keyword argument rejects values outside a small predefined list, but custom values are reasonable and safe
- The option is implemented as a hard-coded mapping/lookup that raises on unknown keys (or coerces unexpectedly)
- A backwards-compatible enhancement is requested: keep existing named presets while allowing arbitrary user input
- The option influences formatting/serialization/output, making it easy to verify via example outputs
