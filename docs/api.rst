API
===
.. module:: krithika

:code:`krithika` has several classes and functions that are used for different purposes. They are listed below:

To perform aperture photometry, the :code:`ApPhoto` class can be used. It has several methods to perform aperture photometry on the data:

.. autoclass:: krithika.ApPhoto
   :members:

The :code:`ApPhoto` class is wrapped around several other classes, :code:`CHEOPSData`, :code:`TESSData`, and :code:`KeplerData`, that allows the users to download CHEOPS, TESS, and Kepler data directly from command lines and use :code:`ApPhoto` class to perform aperture photometry on these datasets. These classes have the following methods:

.. autoclass:: krithika.CHEOPSData
   :members:

.. autoclass:: krithika.TESSData
   :members:

.. autoclass:: krithika.KeplerData
   :members:

If you used :code:`juliet` for fitting the light curves, you can use the :code:`julietPlots` class to plot the light curves and the fits. It has the following methods:

.. autoclass:: krithika.julietPlots
   :members:

The :code:`SpectroscopicLC` class can be used to perform analysis of spectroscopic light curves from, e.g., JWST. It has the following methods:

.. autoclass:: krithika.SpectroscopicLC
   :members:

The :code:`InvertCowanAgolPC` class can be used to invert the Cowan & Agol (2008) phase curve model to get the thermal properties such as the brightness temperature map and Bond albedo. It has the following methods:

.. autoclass:: krithika.InvertCowanAgolPC
   :members:

:code:`krithika` has a simple light weight class to view the 2D, 3D, or 4D data cubes, with functionalities similar to DS9. The :code:`NDImageViewer` class has the following methods:

.. autoclass:: krithika.NDImageViewer
   :members:

Apart from these classes, there are several functions and classes that can be used for different purposes. They are listed below:

.. autofunction:: krithika.utils.t14

.. autofunction:: krithika.utils.b_to_inc

.. autofunction:: krithika.utils.inc_to_b

.. autofunction:: krithika.utils.rho_to_ar

.. autofunction:: krithika.utils.pipe_mad

.. autofunction:: krithika.utils.rms

.. autofunction:: krithika.utils.fake_allan_deviation

.. autofunction:: krithika.utils.corner_plot

.. autofunction:: krithika.utils.make_psd

.. autofunction:: krithika.utils.generate_times_with_gaps

.. autoclass:: krithika.BrightnessTemperatureCalculator
   :members: