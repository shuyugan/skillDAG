# Fix Unicode normalization short-circuits

Addresses bugs in text-processing utilities where Unicode normalization and downstream transformations (e.g., diacritic removal) behave inconsistently depending on whether input text is already normalized. Apply when equivalent Unicode representations yield different outputs due to early-return or bypass logic.

## Trigger Conditions

- Text-processing output differs for visually identical strings that use different Unicode compositions (precomposed vs decomposed + combining marks).
- A helper performs Unicode normalization and has conditional logic that skips subsequent processing when the normalized form matches the input.
- A bug report includes a minimal reproduction using combining characters or canonical/compatibility forms (e.g., NFC/NFD/NFKC/NFKD).
- The desired behavior is representation-invariant (same semantic text should yield the same processed output).
