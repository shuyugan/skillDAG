# Swap Custom Escaper for Stdlib

Applies when a project maintains a custom HTML/text escaping implementation that duplicates functionality available in a standard library or well-maintained dependency. This workflow replaces the custom implementation with the shared one, preserves compatibility where needed (including reverse/round-trip paths), and stabilizes tests that assert string-level output or compare HTML across renderers.

## Trigger Conditions

- A core utility function duplicates escaping/sanitization behavior available in a standard library or widely used dependency
- A performance/maintenance improvement is desired by delegating to a shared implementation
- Downstream code or tests are coupled to exact escaped string spellings (e.g., numeric entity forms) and may break after the swap
- There are reverse-transform or round-trip code paths (e.g., unescape/URL-unescape/normalizers) that must remain compatible with legacy output
