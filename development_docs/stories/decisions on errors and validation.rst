Validation and error handling
#############################

Validation objectives
^^^^^^^^^^^^^^^^^^^^^

I would like to be permitted to have pretty good
validation in the project. I would like to be able
to

1) Validate when an input is insane
2) Validate that probabilities and other values are sane
3) Let validation have minimal performance impact
4) Let the validation operate in a pythonic manner that is easy to use without
   excessive rethinking from the user.
5) Have validation mechanisms compatible with jit.
6) Support for logging would be great
7) Be able to do these all at the same time.
8) Not have to support something insane.

Problems
^^^^^^^^

Error handling under jax has proven... significantly harder than I expected

- At first, I tried validation in much the same way as would be found in pytorch.
  This caused major problems when compiling, however.
- For simplistic tasks involving only tensor metainfo jax will happily raise errors
- However, anything that involves checking the value of a tensor, such as asserting probability
  tensors are between 0 and 1, fails to jit if jax cannot statically resolve the values
  during compilation but needs to leave them as a parameter
- Additionally, there are performance concerns. Validation takes computation time. Ideally,
  we would like to be able to shut this off if we need the extra performance

What breaks jit?
^^^^^^^^^^^^^^^

Ultimately, the thing that breaks XLA compilation is when we are raising
errors based on the values contained within tensors.

What validations tasks conflict?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main validation tasks that conflict with this are things like validating
probabilities are sane, and constants are between the right values.

Approaches
^^^^^^^^^^

Several approaches have been considered to resolve this.
They actually caused me to stop for a bit. Some of the options,
and their issues, are listed below.

- No validation whatsoever.
    - I want to avoid this at all costs, however, as I do not want to relearn how
      classes work whenever I make a stupid error. Leaving this as an option for performance reasons is
      viable
    - Breaks #1-7
- Provide only jit-compatible validation
    - Only validate things which can be jit compiled
    - Ignore other issues
        - Breaks models in hard-to-notice ways
    - Clamp tensors to between values they should be in
        - Breaks models in other difficult-to-notice ways.
    - breaks #2
- Give up on jit
    - No.
    - Not an option.
    - It is the whole point of jax.
    - breaks #6
- Use checkify
    - Works, but breaks point #7. Now users have to explicitly support checkify, staging,
      then raising in their programs.
- Use checkify. The experimental jax library will capture the side effects
    - This IS an option... but it breaks other parts of the framework's promise. This
      is suppose to be very easy to use - but forcing checkify on people will now
      demand they learn how to use it's framework
    - Nonetheless, it has been shown to work
    - Breaks #4
- Place hooks on jax somehow to make jit compilation raise checkify errors automatically
    - The jit framework itself has no way to register hooks.
- Provide the user with a custom jit function that properly interfaces with checkify
    - This does work, and if the user used it in place of jax.jit we would fine. However,
      it breaks the drop-in replacement nature of the project.
    - Breaks #4
- Monkeypatch jax.jit and other related functions to use the above wrapper.
    - No.
    - Breaks #8
    - I certainly do not want to support it if jax changes their internals.
- Use a python based debug callback to perform value based validation
    - jax.debug.callback might do this
    - It WILL have performance consequences
- Warn the user through print statements when violating certain constraints
    - May have performance issues
    - There is no way to skip
- Use jax.cond if possible
    - It IS possible to use jax.cond to conditionally print a message to the
      console, interestingly enough.
    - It should actually have minimal performance impact.
    - This might just do it. I could print the error message to the console
      instead, and trust the programmer debugging it to notice.

Conclusion? There is no way to satisfy all my objectives. One of them
has to give

What to yield
^^^^^^^^^^^
We clearly cannot do everything. Any attempt to do so WILL fail
under the current architecture. We must give up something

We sacrifice #8 - doing everything at once. We also sacrifice some
of #2.



Validation architecture
^^^^^^^^^^^^^^^^^^^^^^^

We divide validation into several regimes. These regimes are

- Off: No attempts at validation are performed. Fast
- Basic: Default. Provides validation on the things that do not break
         jit, and prints issues to the console where raising an error
         is not an option.
- Checkify: Debugging occurs through checkify checks. YOU control how
            the erros are handled
Design and need
^^^^^^^^^^^^^^^^^^^