# Intro

This is a roadmap regarding how this project will work

# Objectives

* Make a jax act controller capable of handling explictly passed state
* Make it capable of handling pytrees of state
* Make it capable of resetting a channel if demanded
* Make a simple builder system

# Setup

Setting up an ACT instance should be flexible. To ensure compatible with 
jax scripting all shapes must be defined during setup. We must ensure that
happens somehow. 

Initialization will have to cover the shape of the probabilities - 
the batch like shape - and the shapes of the accumulators. 

Setup should be able to "resume" a previous ACT section, meaning that
the user can provide previous accumulators, probabilities, and residuals.



## main shape

The main shape will have to match for all probabilities and tensors. It is
also the shape the probability accumulators will be found in

## 

# ACTController

A controller for ACT states, that explicitly initializes its states and consists of 
static methods. 