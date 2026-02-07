krithika
--------


A comprehensive Python toolkit for exoplanet time-series data analysis, encompassing photometry, spectroscopy, and multi-wavelength observations from space telescopes.

Overview
--------
--------
`krithika` provides end-to-end analysis capabilities for exoplanet transit and eclipse observations, with particular focus on data from CHEOPS, JWST, TESS, and Kepler missions. The package integrates photometric extraction, light-curve fitting, spectroscopic analysis, and interactive visualization tools.

## Features

### Core Capabilities
**Multi-mission Support:** Support for CHEOPS, JWST, TESS, and Kepler data formats

- **Photometry:**

    - Aperture photometry with customizable apertures and sky regions
    - Sub-array image handling for detailed pixel-level analysis
    - Background and contamination correction

- **Spectroscopy:**

    - Multi-wavelength light-curve analysis
    - Parallel fitting across spectroscopic channels
    - Parameter spectra extraction (transit depth vs wavelength)

- **Visualization:**

    - Interactive ND image viewer (2D/3D/4D support)
    - Draggable cut-profile tools for pixel analysis
    - 2D spectro-temporal heatmaps
    - Publication-quality parameter spectra plots

- **Noise Analysis:**

    - Allan deviation and white-noise floor estimation
    - Robust noise metrics across binning scales
    - Multi-wavelength noise comparison


## Installation
---------------
Requirements
    - Python 3.8+
    - numpy, scipy, matplotlib
    - astropy, astroquery
    - scikit-learn
    - photutils (optional, for advanced aperture photometry)
    - juliet
    - corner
    - dace_query (optional, for CHEOPS data download)

### Quick install

Latest stable version:

```pip install krithika```

Developement version from Github:

```
git clone https://github.com/Jayshil/krithika.git
cd krithika
pip install -e .
```


License
-------

This project is Copyright (c) Jayshil A. Patel and licensed under
the terms of the GNU GPL v3+ license. This package is based upon
the `Openastronomy packaging guide <https://github.com/OpenAstronomy/packaging-guide>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.

Contributing
------------

We love contributions! krithika is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
krithika based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
