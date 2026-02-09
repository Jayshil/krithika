# Krithika

A Python toolkit for exoplanet time‑series analysis across CHEOPS, JWST, TESS, and Kepler/K2, including photometry, spectroscopic light‑curve fitting, and visualization.

## Highlights

- **CHEOPS** data access (DRP light curves, subarrays) and photometry
- **Spectroscopic light‑curve analysis** with channel binning and parallel fitting
- **Interactive ND image viewer** for 2D/3D/4D datasets
- **Noise analysis** (Allan‑deviation style) and power spectra
- **PLD/PCA** utilities for systematics analysis
- **Brightness temperature** estimation from eclipse depths

---

## Installation

### From source
```bash
git clone https://github.com/Jayshil/krithika.git
cd krithika
pip install -e .
```

### From pip
```bash
pip install krithika
```

### Dependencies (core)
- `numpy`, `scipy`, `matplotlib`
- `astropy`, `astroquery`
- `juliet` (light curve fitting and `juliet` plots)
- `tqdm`, `corner`

### Optional dependencies
- `photutils` (aperture photometry)
- `dace_query` (CHEOPS archive access)

---

## Quick Start

### ApPhoto (Aperture Photometry)
```python
from krithika import ApPhoto

# Load aperture photometry data (times, frames, errors, bad-pixel map, and aperture information)
## Generating fake data
times = np.arange(1000)
frames, errors = np.ones( (1000, 100, 100) ), np.ones( (1000, 100, 100) )
badpix = np.ones( (1000, 100, 100) ) # 1 means good pixel, 0 means bad-pixel

## In case of circular aperture with sky annulus
phot = ApPhoto(times=times, frames=frames, errors=errors, badpix=badpix, aprad=20, sky_rad1=40, sky_rad2=50)

## In case of aperture made up of N brightest pixels and background made up of M faintest pixels
phot = ApPhoto(times=times, frames=frames, errors=errors, badpix=badpix, brightpix=True, nos_brightest=20, nos_faintest=50)

## Centroids
cenr, cenc, _, _ = phot.find_center()

## Simple aperture photometry
fl, fle, bkg, _, _ = phot.simple_aperture_photometry(method='photutils', bkg_corr='median', robust_err=True, plot=False)

## Pixel-level decorrelation
_, _, PCA = phot.pixel_level_decorrelation()
fl_aper, fl_pred, _, _ = phot.pld_correction()
```

Please look at the docstrings for more functionalities and for description of the class. The `ApPhoto` class is wrapped around several other classes, `CHEOPSData`, `TESSData`, and `KeplerData`, that allows the users to download CHEOPS, TESS, and Kepler data directly from command lines and use `ApPhoto` class to perform aperture photometry on these datasets.

### CHEOPS: DRP light curves and subarrays
```python
from krithika import CHEOPSData

cheops = CHEOPSData(object_name="WASP-189b")

## This will load DRP light curves
drp = cheops.get_drp_lightcurves(pout="./cheops_data")

## This will download subarrays so that we can then perform aperture photometry (ApPhoto class)
cheops.get_subarrays(pout="./cheops_data")
phot = cheops.ApPhoto(visit_nos=1, aprad=20, sky_rad1=30, sky_rad2=50)
flux, flux_err, _, _, _ = phot.simple_aperture_photometry(method='photutils', bkg_corr='median', robust_err=True, plot=False)
```

Similarly we can download and work with TESS and Kepler/K2 datasets as follows:

### TESS and Kepler/K2 datasets
```python
from krithika import TESSData, KeplerData

## For TESS data
data = TESSData(object_name='WASP-189')
sector = 'TESS51'
## For Kepler data
data = KeplerData(object_name='WASP-107')
sector = 'Kep0000-C10-1'

### To download PDC-SAP light curves
tim, fl, fle, _ = data.get_lightcurves(self, pdc=True)

### To download the target-pixel files
data.get_tpfs()
phot = data.ApPhoto(sector=sector, brightpix=True, nos_brightest=12, nos_faintest=50)
flux, flux_err, _, _, _ = phot.simple_aperture_photometry(method='photutils', bkg_corr='median', robust_err=True, plot=False)
```

It is also possible to analyse spectroscopic light curves using `juliet`. Using `SpectroscopicLC` class, we can fit light curves parallelly. Finally, we can use some of the functions in the class to plot results.

### Spectroscopic light‑curve analysis
```python
import numpy as np
import juliet
from krithika import SpectroscopicLC

times = np.load("times.npy")
lc = np.load("spec_lc.npy")
lc_errs = np.load("spec_lc_err.npy")
wavelengths = np.load("wavelengths.npy")

def get_priors(ch_name):
    par = ['P_p1', 't0_p1', 'p_p1_' + ch_name, 'b_p1', 'q1_' + ch_name, 'q2_' + ch_name, 'a_p1', 'ecc_p1', 'omega_p1']
    dist = ['fixed', 'fixed', 'uniform', 'fixed', 'uniform', 'uniform', 'fixed', 'fixed', 'fixed']
    hypers = [1., 0., [0.,1.], 0., [0., 1.], [0., 1.], 10., 0., 90.]
    return juliet.utils.generate_priors(par, dist, hypers)

spec = SpectroscopicLC(
    times=times,
    lc=lc,
    lc_errs=lc_errs,
    wavelengths=wavelengths,
    priors=get_priors,
    pout="./results"
)

## Just plotting the 2D data
plot2Ddata(self, cmap='plasma')
plt.show()

spec.analyse_lc_parallel(nthreads=4, ch_nos=10)
fig, ax = spec.plot_parameter_spectrum("p_p1", bins=10, plot_white=True)
plt.show()

## Plot the 2D data and model
plot2D_data_model_resids(cmap='plasma')
plt.show()
```

### ND Image Viewer
```python
from krithika import NDImageViewer
import numpy as np

data = np.random.rand(50, 128, 128)
viewer = NDImageViewer(data=data, cmap="magma")
viewer.show()
```

### Plots for `juliet`-fitting

Finally, using `julietPlots` class, we can plot light curve models, phase folded light curves, gp models, corner plots, and allan deviation proxy plots for `juliet`-fitted models.
```python
from krithika import julietPlots
import os

data = julietPlots(input_folder=os.getcwd(), N=5000, sampler='dynamic_dynesty')

## full model with data
fig, ax1, ax2 = data.full_model_lc(['TESS31'], quantile_models=False)
plt.show()

## Phase-folded model
figs, axs1, axs2, axs3, axs4 = data.phase_folded_lc(phmin=0.8, nrandom=30, highres=True, quantile_models=True)
plt.tight_layout()
plt.show()

## To plot GP model
fig, axs = data.plot_gp(instruments=None, highres=True, one_plot=True, pycheops_binning=False)
plt.tight_layout()
plt.show()

## "Allan" deviation plots
fig, axs, _, _, _ = data.plot_fake_allan_deviation(instruments=['TESS31'], method='pipe')
plt.show()

data.plot_corner(planet_only=False, save=True)
```

---

## Modules & Classes

### `ApPhoto`
Aperture photometry on time‑series cubes:
- circular or brightest‑pixel apertures
- sky annulus handling
- PLD/PCA tools

### `TESSData`, `KeplerData`
MAST queries for light curves and target pixel files.

### `CHEOPSData`
- `pipe_data()` — read PIPE FITS light curves
- `get_drp_lightcurves()` — download DRP light curves
- `get_subarrays()` — download subarray cubes
- `ApPhoto()` — aperture photometry wrapper

### `SpectroscopicLC`
- `generating_lightcurves()` — channel binning
- `analyse_lc_parallel()` — parallel fitting
- `plot_parameter_spectrum()` — spectral parameter plot
- `plot2Ddata()` / `plot2D_data_model_resids()` — 2D maps
- `joint_fake_allan_deviation()` — combined noise plots

### `NDImageViewer`
Interactive 2D/3D/4D viewer with:
- scale modes (Linear/Log/Asinh/Zscale)
- sliders for frames/groups
- draggable cut profiles and live plots

---

## Utilities (`utils.py`)

- Light‑curve binning (`lcbin`)
- Allan deviation proxy (`fake_allan_deviation`)
- Lomb–Scargle PSD (`make_psd`)
- PCA (`classic_PCA`)
- Orbital geometry utilities (`t14`, `b_to_inc`, `inc_to_b`)
- Brightness temperature calculation (`BrightnessTemperatureCalculator`)

---

## Notes

- Many functions cache outputs; delete cached files to recompute.
- For CHEOPS downloads, `dace_query` must be installed and configured.

---

## License

MIT (if not otherwise specified).

---

## Citation

If you use this package in a publication, please cite the repository:

```
@software{krithika,
  author = {Jayshil},
  title = {Krithika: Exoplanet Time-Series Analysis Toolkit},
  year = {2026},
  url = {https://github.com/Jayshil/krithika}
}
```