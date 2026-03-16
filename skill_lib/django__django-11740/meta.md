# Fix Missing Dependencies in Schema Diff

Addresses bugs in schema-diff or migration/autogeneration systems where operations that alter existing definitions fail to include required ordering/dependency metadata (especially when introducing new references/relationships). Apply when an automatically generated change set executes in the wrong order or cannot resolve newly introduced references across modules/components.

## Trigger Conditions

- Auto-generated change scripts fail at runtime due to unresolved references to a newly related/linked entity after an alteration.
- Generated change operations differ in dependency behavior between “add” vs “alter” paths for the same kind of relationship.
- Cross-module changes work when introduced as a new field/entity but break when introduced via modification of an existing field/entity.
- Migration/order metadata is missing or incomplete for operations that introduce relationships during an alteration.
