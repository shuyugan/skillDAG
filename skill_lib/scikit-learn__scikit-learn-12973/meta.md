# Resolve Duplicated Configuration Parameters

Addresses bugs where the same configuration option is exposed in multiple API layers (e.g., constructor and method arguments) and different internal steps accidentally consult different sources. Apply when a defaulted method parameter silently overrides an instance-level setting or when internal stages show inconsistent behavior due to mixed configuration precedence.

## Trigger Conditions

- A class-level configuration option is duplicated as a method argument with its own default value
- User-provided instance configuration is silently overridden when calling a method without explicitly passing the duplicated argument
- Different internal phases (e.g., validation/preprocessing vs core algorithm) read the configuration from different places
- A fix requires preserving backward compatibility while clarifying precedence rules
