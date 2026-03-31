# Fix zero-dimension join regressions

Addresses regressions where combining/concatenating structured containers (e.g., matrices, tables, tensors) produces incorrect shapes when one or more operands have zero rows/columns (or other empty dimensions). Apply when behavior differs across implementations (dense vs sparse, mutable vs immutable) or changes across versions, especially when emptiness shortcuts drop dimension information.

## Trigger Conditions

- A concatenate/stack/join operation returns an incorrect shape only when one or more inputs have a zero-size dimension (e.g., 0 rows or 0 columns)
- The same operation behaves correctly for nonzero dimensions but fails for all-zero or partially-zero shapes
- Different concrete implementations of the same abstraction (e.g., dense vs sparse) disagree on results for the same inputs
- The failure appears after a version/implementation change and suggests an emptiness/truthiness shortcut is being taken
