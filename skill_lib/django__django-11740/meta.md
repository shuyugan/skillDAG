# Fix missing migration dependencies on altered relations

Use this workflow when an ORM/migration system generates schema-change operations but fails to declare required ordering dependencies after a field is altered into a relationship (especially across modules/apps). It guides you from locating the dependency computation logic, reproducing the issue with a minimal state-diff test, implementing consistent dependency inference for altered fields, and validating with targeted and regression tests.

## Trigger Conditions

- A migration that alters a field into a relationship/reference does not include a dependency on the referenced module/app.
- Runtime or migration-time failure indicates an unresolved related model/type after applying generated migrations.
- Dependency behavior differs between “add field” and “alter field” operations for relational/reference fields.
- A minimal reproduction shows empty or incomplete dependencies for a cross-module relationship change.
