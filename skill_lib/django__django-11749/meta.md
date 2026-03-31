# Fix kwarg-to-CLI parsing gaps

Addresses bugs where programmatic invocation of a CLI-style command (passing structured keyword options) does not behave the same as invoking the command with equivalent command-line strings. Apply when an option parser enforces constraints (e.g., required groups, mutual exclusion) but the invocation layer only forwards a subset of provided options into parsing/validation.

## Trigger Conditions

- A command succeeds when called with CLI-style option strings but fails when called with equivalent keyword/structured options
- The error indicates missing required options or group constraints despite the caller providing values programmatically
- The invocation helper constructs a synthetic argv/parse list and conditionally includes options (e.g., based on per-option required flags)
- Options are part of higher-level constraints (mutual exclusion groups, conditional requirements) not represented on individual option metadata
