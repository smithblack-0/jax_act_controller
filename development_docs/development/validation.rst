Development notes
#################

Needed planning:
^^^^^^^^^^^^^^^


- Inline
    - Validation occurs


Planned features
^^^^^^^^^^^^^^^^

Modes definition:

- Validation_Details_Level enum
    - Three modes
        - Off
        - Basics
        - Advanced: Default
    - Off: No validation occurs
    - Basics: Only things that do not break jit occur
    - Advance: All validation occurs

- Validation_Jit_Config
    - Debug: Cannot jit
    - Warnings: Can jit, raises warning to console
    - Checkify: Can jit, but must stage checkify. Raises errors in checkify.

- Logging_Mode enum:
    - Three modes:
        - Console: Raises issues to console
        - File: Dumps errors into a file along a particular path
        - Callback: Captures issue, pushes them into callback

Validation_Config dataclass:
    validation_detail_level: str =



Validation Operators:
- ValidationOperator:
    - Contains a single validation operation
    - Will have an validation predicate function. Must return true to pass
    - Will have a validation failed function. This will accept a formatted message
      and should promise to raise an error in some way.
    - Will have a raising
    - Will have a details message function. This is a format string
    - Will have a flag called
    - Has a __call__ mechanism that accepts
        - A operand mode
        - A validation mode
        -

- "Validator" class:

    - This is a combination of a builder and a callable.
      You might refer to it as a prototype? Anyhow,
      It is designed to be built once with the validation
      and warnings definitions provided, then executed many times.

      Unlike just about every other class, it is stateful. This is because,
      since it will be built once and only jitted after building, there will
      be nothing that can change between runs.

    - Builder mode -
        - Can attach error raising validation predicate function and issue details message
        - Can attach warning raising validation predicate function and issue details message
        - May define kwargs with message which to fill in with details message.
    - Callable mode - Will execute validation
            - Called with operand to validate,validation mode, and error message arguments
            - Executes validation.
            - Raises errors or prints warnings as appropriate.

Validation Registry:
    - A registry into which various validators can be placed.
    - Emphasizes the define-once, use-many nature of validators.
    - Register_validator:
        - Adds validator to the registry
        - Must define a name to get it from
    - Get_Validator:
        - Gets a validator that has been registered by name.

Support features
^^^^^^^^^^^^^^^^