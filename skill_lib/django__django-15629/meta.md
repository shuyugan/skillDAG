# Propagate schema attributes to relation columns

Applies when an ORM or schema generator creates relationship/foreign-key columns whose database-specific attributes (e.g., collation/charset) must match the referenced column. Use this workflow to trace where column metadata is computed, implement attribute propagation for relation fields, and validate via focused checks plus regression tests.

## Trigger Conditions

- Generated schema/migration SQL creates a constraint that fails due to mismatched column attributes between referenced and referencing columns (e.g., collation/charset/type modifiers).
- A model/field option applied to a primary/unique column is reflected in that column’s DDL but is missing from the DDL for relation/foreign-key columns that reference it.
- Relationship fields derive their database type from a target field, but additional target-field database parameters are not being carried through to schema generation.
- A backend requires exact attribute compatibility for constraint creation, and migrations omit required attributes on referencing columns.
