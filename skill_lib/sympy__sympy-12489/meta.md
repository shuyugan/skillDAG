# Fix subclass-preserving object construction

Addresses bugs where subclassing a core type fails because internal constructors/factories hard-code the base class or use module-level aliases that bypass the dynamic class. Apply when instances created through helper functions, alternate input forms, or fast paths return base-class objects instead of subclass instances.

## Trigger Conditions

- A subclass constructor call returns an instance of the base class for some inputs or code paths
- Object creation flows through internal helper/factory functions rather than directly using the class passed to __new__
- There are module-level aliases or cached references to the base class used in allocation (e.g., Base = ...; helper uses Base)
- Early-return/fast-path branches in __new__ (copy/resize/coercion paths) allocate objects without referencing the calling class
