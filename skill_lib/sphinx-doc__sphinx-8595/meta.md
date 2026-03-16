# Respect Explicit Empty Export Lists

Applies when a documentation/introspection or enumeration feature filters items based on an optional user-defined export list (allowlist), but an explicitly empty list is mistakenly treated as “not provided.” Use this workflow to locate the selection logic, fix truthiness/None conflation, and validate behavior across absent/empty/non-empty configurations.

## Trigger Conditions

- A feature supports an optional allowlist/denylist configuration (e.g., an export list) that changes what items are shown or processed
- Setting the list to empty is expected to suppress everything, but the system behaves as if the list were not set
- The code likely uses a generic truthiness check that can’t distinguish “unset” from “set to empty”
- A bug report includes a minimal reproduction where an explicit empty configuration is ignored
