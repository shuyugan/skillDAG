# Propagate Engine Defaults into Render Context

Use this workflow when a high-level rendering/helper API should honor configuration stored on an owning object (engine/client/service), but a sub-object (context/options/request) is created with defaults that override that configuration. It guides you from locating the wrapper code path, confirming the missing propagation, implementing a minimal fix that preserves caller overrides, and validating via targeted and automated tests.

## Trigger Conditions

- A configurable component exposes an option (e.g., escaping, encoding, strictness) but a convenience method ignores it and behaves as if the default is always enabled
- The problematic behavior happens only when inputs are passed in a “raw” form (e.g., plain map/dict) but not when callers pass a fully constructed object
- Inspection suggests an internal wrapper constructs a context/options object without passing through the owning component’s configuration
- A bug report indicates a regression after refactoring that introduced new helper/wrapper code
