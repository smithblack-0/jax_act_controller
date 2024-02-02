Extra act iterations and a stack subroutine approach
----------------------------------------------------
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
