# Intro

This is a roadmap regarding how this project will work

# Objectives

* Make a jax act controller capable of handling explictly passed state
* Make it capable of handling pytrees of state
* Make it capable of resetting a channel if demanded
* Make a simple builder system

# Setup

Setting up an ACT instance should be flexible. To ensure compatible with jax scripting
all shapes must be defined during setup. 

* Shapes like
* Force initialization
* 


# ACTController

A controller for ACT states, that explicitly initializes its states and consists of 
static methods. 