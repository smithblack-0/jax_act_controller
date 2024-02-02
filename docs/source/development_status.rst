Development status
=================

This is a document for keeping track of various
development details. This includes proposed
features, implemented features, and
generally anything else related to development

Implementation of features
--------------------

Development tags
^^^^^^^^^^^^^^^^

- It is useful to understand what the development tags mean:

  * **Possible**
    Still being considered.

  * **Unstarted**
    Going to be included, but not started.

  * **Developing**
    Under active development.

  * **Needs Update**
    A change has been made, and something
    needs an update

  * **Experimental**
    Code is functioning, but no promises on backwards compatibility.

  * **Stable**
    Objects may have methods or properties added, but will never break their existing API.

  * **Finished**
    Code will never be changed.

Core features
^^^^^^^^^^^^^

These are core pieces of logic
in the main method and mechanism.
This means they are not bound to
any particular framework beyond jax, and
should be fairly framework agonistic.



.. list-table:: Implementation status
   :widths: 25 50 25 25 50
   :header-rows: 1

   * - Name
     - Purpose
     - Status
     - Unit Testing
     - Comment

   * - Controller
     - Accumulates and updates state
     - Stable, Need update: comments: new properties
     - Needs update: Coverage, comment
     - May still add properties

   * - Builder
     - Creates the controller
     - Stable
     - Stable
     - May still add new properties or definition methods

   * - TensorEditor
     - Allows editing of act state during computation
     - Experimental
     - Experimental
     -

   * - GroupEditor
     - Allows manipulation of controllers using magic operators
     - Possible
     - Possible
     -

   * - Viewer
     - Helps view internal state using masks
     - Stable
     - Stable
     - May add more views if they are useful

   * - AbstractLayerTemplate
     - An abstract layer designed to be subclassed into
       particular framework layers.
     - Developing, Needs Update: Comments, methods
     - Developing
     - Cannot be marked stable until all common frameworks are
       covered.

Integration Testing
^^^^^^^^^^^^^^^^^^^

Integration testing of core features.

.. list-table::


Framework features
^^^^^^^^^^^^^^^^^^

Framework specific layers and functions. The end goal
will be that if there is a framework, there will
be a function that can be be called from here to return
a layer or function usable in that framework.

.. list-table:: Implementation Status
   :widths: 25, 25, 25, 25, 50
   :header-rows: 1

   * - Framework Name
     - Supported?
     - Layer Status
     - Tests Status
     - Comment

   * - Jax
     - Yes
     - Stable
     - Stable
     - Core, so automatically supported.

   * - Flax
     - Developing
     - Developing
     - Developing
     - Just learning

   * - Trax
     - Planned
     - Unstarted
     - Unstarted
     - None

   * - Optax
     - Planned
     - Unstarted
     - Unstarted
     - Unstarted

   * - Haiku
     - Possible
     - Possible
     - Possible
     - I have never used

   * - Elegy
     - Possible
     - Possible
     - Possible
     - I have never used

