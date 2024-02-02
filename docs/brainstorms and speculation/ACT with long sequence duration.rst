
Act and associated gradient descent is not
great at properly training when handling
long sequences.

The issue at hand
=================

There is an issue with ACT. Basically, with
randomly generated models, it is very difficult
to encourage a model to iterate over exceptionally
long sequences. This is because any created model
will tend to get stuck in the first local optimum

Lets consider a computation that illustrates the problem.
You randomly initialize a problem that, secretly, requires around
ten act iterations worth of work for ideal results. However,
you can get a crappy result for 4 iterations of work.

The random projection from a dense into a sigmoid is about 0.5, meaning
at first you do two iterations of ACT work. As your model trains, it
encourages a longer sequence... until it finds the crappy solution 4 iterations.
in. Then.... it stops! It does not want to train any further, because it is
stuck in a local optimum.

Insufficient dimensions
=======================

What is going on is that we are getting stuck in local
optimums because we have insufficient dimensions.

Normally, neural processes have vectors with many dimensions,
avoiding these issues. This is because usually you see saddlepoints
in higher dimensions rather than local optimums, and those local optimums
that exist tend to be pretty good.

However, we only have one dimension that matters here - the dimension of the
act computation. There is no way to "Go around" as in a saddlepoint. Another
approach is needed.


What are our options?
=====================

There are basically two solutions that are worth a damn.
These are to either increase the effective dimensions of
the act process by nesting, or force the act process
to take longer using scheduling.

Nested act sessions and dimensional multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One thing worth noting is that simple
perceptrons also suffer from this issue
when insufficient dimensions are available.

They get around this by adding extra dimensions,
quite easy to do when you are just doing basic
matrix operations.

This suggests a possibility. Maybe a solution
might be to nest act sessions within each other.
Under such a circumstance, perhaps each one
can only go 3-4 deep... but together
they can have a multiplicative effect to
go much deeper. This also adds extra effective
dimensions.

This would also link nicely, it should be noted,
with any attempt at subroutine machines.

Options for scheduling
=======================


Scheduled Probability Depression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One simple option to handling this issue is to simply reduce the maximum
amount a probability can contribute to the cumulative halting probability.

We define a quantity called the "probability depression". This is a number
ranging from 0.0 to 1.0 which will be multiplied by the halting probabilities
and which can depress how much each iteration can actually contribute to ACT.
Lets see how this can help.

Lets consider the situation where you want to let the model explore, as it warms
up, over long sequences. Normally, this is unlikely to happen. Sigmoid with a random
projection will likely result in a halting probability of around 0.5. Two steps, and
you are done.

Instead, we start with a probability depression of 0.1. Now, we get that the halting probability
will typically be around 0.1*0.5 = 0.05. The average time until that adds up to one will
be around 20 iterations. this means there are far more entries we will typically be exposed
to! And the model should be able to begin converging on the best even under this
situation.

This is good when beginning to train, but it will not be needed as the model
begins to converge. Thus, we can schedule the probability depression. A reasonable
starting point might be from [0, 1.0], over 500 steps, with a warmup of 200 steps.
After 700 steps, depression is off.

But the key thing is the model was forced to train while considering the possibility of
longer act sequences.

====================