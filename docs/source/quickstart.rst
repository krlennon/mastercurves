Quick Start
===========

.. _installation:

Installation
------------

To use mastercurves, first install it using pip:

.. code-block:: console

   $ pip install mastercurves

Creating a master curve
-----------------------

First import the package, then use the ``MasterCurve()`` constructor to initialize
a master curve object:

.. code-block:: python

   from mastercurves import MasterCurve
   mc = MasterCurve()

Adding data to a master curve
-----------------------------

Next, collect data into three lists: ``x_data``, ``y_data``, and ``states``. The elements
of ``x_data`` and ``y_data`` should be arrays containing the x- and y-coordinates for a
single state (i.e. one data set, which will be superposed with data sets comprising
the other elements of ``x_data`` and ``y_data``). The elements of ``states`` should be
numeric values labeling the corresponding states.

When the data is ready, add it to the master curve:

.. code-block:: python
   mc.add_data(x_data, y_data, states)

Defining the coordinate transformations
---------------------------------------

Then, add coordinate transformations to the master curve. If only horizontal shifting
by a scale factor is required (the typical case for time-temperature superposition),
this can be done as follows:

.. code-block:: python
   from mastercurves.transforms import Multiply
   mc.add_htransform(Multiply())

Superposing the data
--------------------

The master curve is now ready for superposition:

.. code-block:: python
   mc.superpose()

Plotting the results
--------------------

Once superposition is complete, you can generate plots of the raw data,
data with Gaussian process interpolants, and the superposed mastercurve!

.. code-block:: python
   mc.plot()

