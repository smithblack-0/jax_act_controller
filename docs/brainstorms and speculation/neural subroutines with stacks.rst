ACt, Subroutines, and Stacks
============================
It might be useful to allow the ACT instance to
operate multiple instances of itself, in
a stack.

Imagine you provide the model itself some mechnaism
Say you deplete some amount of probability for
each level deeper you go, and you figure
out a way to sanely account for the loss function.

Then perhaps you let an act instance choose to
spawn an act instance, which acts as a subroutine,
doing some activity, finalizing the result,
and returning the result plus the original
question for further processing.

This is not dissimilar, it should be noted,
to how real logic often works. However, difficulties
may arise in managing the loss functions.

There are, however, some problems. For one,
different batches may open subroutines
at different times. This could be handled several
different ways, none all that elegant.

The straightforward option is to, when a subroutine
run is decided upon, open a subroutine call on all batches
whether or not they will be used, execute it and get the
result tensors, merge those into an update, and then
only commit the update where I actually needed a subroutine
call.

The thing here is it should be the case that subroutines
are sometimes run on only one or two batches at a time, which
will mean the majority of the TPU is doing calculations that
will be discarded. Now, ACT already has this problem, but this
will likely magnify it significantly.

Another option might be to have an exposed state computation
that can be stored in a finite-depth differential stack.

Computation would consist of a push step, a computation
action, and a pop-merge step. The program gets to decide
when to push using a push probability. When the halting
probability is reached, the last stack state is popped and
restored, then merged with the subroutine results to produce
an updated state tensor. Then computation resumes.

This is far more technically demanding, but would be effective.
In theory, unless processing has halted entirely for a batch,
every iteration will result in some computation happening.

In practice, I will have to think this over to see if
this can be made even close to reasonable. It might
require special handling in particular frameworks as
well. I likely will need to be able to explicitly access all
state to even get close to making this work as well, which
will significantly complicate framework integration.

A simplified stack driven machine
=================================

Lets start by making a completely normal,
batchless, recursive, version of what would be desired.
This should give us some idea how practical it might be.
Can I turn it into a tail recursion by any chance?

def neural_subroutine(directive,
                      state,
                      starting_probabilities
                      )->state:

    # Test, break immediately if we are never going
    # to loop

    halting_probabilities = make_probabilities(state)
    if halting_probabilities >= halt_threshold:
        return zeros_like(state)

    # Setup tools and execute
    tool_parameters = make_tool(directive)
    act_unit = make_act_unit(starting_probabilities)
    while act_unit.halting_probability < halt_threshold:
        state, subroutine_dispatch = update_state(tool_parameters, state)
        if get_halting_probabilities(subroutine_dispatch) + act_unit.probability >= act_unit.halting_threshold:
            return state

        subroutine_result = neural_subroutine(subroutine_dispatch,
                                              state,
                                              act_unit.probabilities)
        state = merge_result(state, subroutine_result)
        halting_probabilities = make_probabilities(state)

        act_unit = act_unit.cache_update("output", state)
        act_unit = act_unit.iterate_act(halting_probabilities)

    return act_unit["state"]

Problems:
---------

- No accounting for batches
- Make a subroutine which may have only one batch running frequently.
- It is possible it will get quite bad.

Neural subroutine architecture
==============================

Instead, we present the model with a differential stack, and
the ability to push or pop its state onto the stack at any time

def run_pop(stack, state):
    new_stack, features = stack.pop()
    parameters, act_unit, parent_state = features
    state = merge_results(state, parent_state)
    return new_stack, parameters, act_unit, state

def run_update(stack, parameters, state, act_unit):
    state = run_update(parameters, state)
    output = make_output(state)
    halting_probabilities = make_halting_probabilities(state)

    act_unit = act_unit.cache_update("output", output)
    act_unit = act_unit.iterate_act(halting_probabilities)
    return stack, parameters, act_unit, state

def run_push(stack, parameters, act_unit, state):
    new_stack = stack.push(parameters, act_unit, state)
    parameters = make_subparameters(state)
    act_unit = make_act_unit(act_unit.probabilities)
    state = make_substate(state)
    return new_stack, parameters, act_unit, state


def neural_subroutine(directive, state):

    parameters = make_tool(directive)
    act_unit = make_act_unit(directive)
    stack = make_stack()
+
    while not act_unit.is_finished:
        action_probabilities = make_stack_probs(state)

        pop_branch = run_pop(stack, state)
        update_branch = run_update(stack, parameters, act_unit, state)
        push_branch = run_push(stack, parameters, act_unit, state)

        stack, parameters, act_unit, state = merge_branches(action_probabilities,
                                                            pop_branch,
                                                            update_branch,
                                                            push_branch)
    return act_unit['output']



