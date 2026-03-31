# Suppress HEAD Response Bodies in Dev Servers

Applies when an HTTP server implementation (often a development or embedded server) incorrectly sends a response body for HEAD requests or mishandles HEAD lifecycle semantics. This workflow locates the response-writing path, identifies the safest interception point, implements HEAD-specific body suppression without breaking header emission, and validates behavior on the wire and across edge cases.

## Trigger Conditions

- HEAD requests return a response body on the wire (not merely displayed by a client), violating HTTP semantics
- A server implementation assumes an upstream component will strip bodies, but that stripping does not occur in some deployment mode
- Users report connection-level noise (e.g., broken pipe/reset) correlated with HEAD requests or early client disconnects
- Fix requires changes in the server/handler layer that finalizes and writes responses (not in application/business logic)
