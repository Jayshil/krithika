# krithika

<p align="center">
  <img src="krithika_logo_highres.png" alt="Krithika logo" width="700">
</p>

A Python toolkit for exoplanet time‑series analysis across CHEOPS, JWST, TESS, and Kepler/K2, including photometry, spectroscopic light‑curve fitting, and visualization.

## Highlights

- Aperture photometry for any dataset
- **CHEOPS** data access (DRP light curves, subarrays) and photometry
- **TESS** and **Kepler/K2** data access and photometry
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

### Inverting Cowan & Agol (2008) phase curve model

If we used Cowan & Agol (2008) phase curve model, then we can use `InvertCowanAgolPC` class to invert the fitted phase curve to find thermal physical properties of the planet as shown below:

```python
import numpy as np
from astropy import units as u
from krithika import InvertCowanAgolPC

# Load posterior samples from eclipse depths and phase-curve fitting
E = np.random.normal(0.01, 0.001, 1000)  # Eclipse depths (F_p/F_*)
C1 = np.random.normal(0.005, 0.0005, 1000)  # Phase-curve cosine (first harmonic)
D1 = np.random.normal(0.002, 0.0002, 1000)  # Phase-curve sine (first harmonic)
C2 = np.random.normal(0.001, 0.0001, 1000)  # Phase-curve cosine (second harmonic)
D2 = np.random.normal(0.0005, 0.00005, 1000)  # Phase-curve sine (second harmonic)
rprs = 0.1  # Planet-to-star radius ratio

# Define instrument bandpass
bandpass = {
    'WAVE': np.linspace(0.3, 5.0, 100) * u.um,
    'RESPONSE': np.ones(100)
}

# Create inversion object
invert = InvertCowanAgolPC(
    E=E, C1=C1, D1=D1, C2=C2, D2=D2,
    rprs=rprs,
    bandpass=bandpass,
    teff_star=5800 * u.K,
    pout="./phase_curve_results"
)

## Day/night brightness temperatures
T_day, T_night = invert.TdayTnight()

## Temperature maps across posterior samples
temp_maps = invert.temperature_map_distribution(nsamples=2000)

## Median temperature map
fig, ax = invert.median_temperature_map(plot=True, cmap='plasma')
plt.show()

## Equatorial temperature profile
fig, ax = invert.equatorial_temp_map(plot=True, nsamples=2000)
plt.show()

## Bond albedo and heat redistribution efficiency
a_by_Rst = 10.0  # Semi-major axis in units of stellar radius
A_B, eps_kelp, _, _, _ = invert.albedo_eps_from_temp_map(a_by_Rst)

## Phase offsets (hotspot shift and phase shift)
phi_off, phi_off_err, phase_off, phase_off_err = invert.phase_offsets(method='root')
```

### Selecting linear detrending regressors

The `SelectLinDetrend` class provides Bayesian model selection to identify optimal linear detrending regressors for light-curve fitting across multiple instruments. The method is mainly designed with CHEOPS in mind, but it can be used for other instruments.

```python
from krithika import SelectLinDetrend
import numpy as np

# Prepare data per instrument
time = {'CHEOPS1': np.linspace(0, 100, 1000), 'CHEOPS2': np.linspace(0, 100, 1000)}
flux = {'CHEOPS1': np.random.randn(1000) * 0.001 + 1, 'CHEOPS2': np.random.randn(1000) * 0.002 + 1}
flux_err = {'CHEOPS1': np.ones(1000) * 0.001, 'CHEOPS2': np.ones(1000) * 0.002}
roll_angle = {'CHEOPS1': np.random.uniform(0,360,1000), 'CHEOPS2': np.random.uniform(0,360,1000)}

# Define priors
def get_priors(instrument):
    ## juliet compatible priors
    par = ['P_p1', 't0_p1', 'p_p1', 'b_p1', 'q1_' + instrument, 'q2_' + instrument, 'ecc_p1', 'omega_p1', 'a_p1']
    dist = ['fixed', 'fixed', 'uniform', 'fixed', 'uniform', 'uniform', 'fixed', 'fixed', 'fixed']
    hypers = [1.0, 0.0, [0., 0.1], 0.0, [0., 1.], [0., 1.], 0., 90., 10.]

    par = par + ['GP_sigma_' + ins, 'GP_rho_' + ins]
    dist = dist + ['loguniform', 'loguniform']
    hypers = hypers + [[1e-3, 1e2], [1, 500]]
    return par, dist, hypers

# Define linear regressors
linear_regressors = {
    'CHEOPS1': {'time': time['CHEOPS1'], 'time_sq': time['CHEOPS1']**2},
    'CHEOPS2': {'time': time['CHEOPS2'], 'time_sq': time['CHEOPS2']**2}
}

# Create selector and run optimization
selector = SelectLinDetrend(
    time=time,
    flux=flux,
    flux_err=flux_err,
    priors=get_priors,
    linear_regressors=linear_regressors,
    roll_degree=3,
    roll=roll_angle,
    pout="./detrend_results"
)

# Find optimal regressors (using lnZ -- Bayesian evidence -- as a selection criteria)
selected_regressors = selector.select_optimal_parameters(n_parallel=4, delta_lnZ_threshold=2.0, selection_method='lnZ')

# Find optimal regressors (using  scatter in the residuals as a selection criteria)
selected_regressors = selector.select_optimal_parameters(n_parallel=4, selection_method='scatter')
```

Please read the API documentation for more details.


## Modules & Classes

### `ApPhoto`
Aperture photometry on time‑series cubes.

- `find_center()` — estimate centroids  
- `simple_aperture_photometry()` — generate simple aperture photometry
- `pixel_level_decorrelation()` — perform pixel-level decorrelation (PLD) 
- `pld_correction()` — PLD‑based light‑curve correction  

---

### `CHEOPSData`
Tools for CHEOPS light‑curve and subarray data handling.

- `pipe_data()` — read PIPE light curves and extract time‑series arrays  
- `get_drp_lightcurves()` — download/load DRP light curves from DACE  
- `get_subarrays()` — download/load CHEOPS subarray cubes  
- `ApPhoto()` — convenience wrapper to run aperture photometry on subarrays  

---

### `julietPlots`
Plotting utilities for `juliet` results.

- `full_model_lc()` — plot full light‑curve model with data  
- `phase_folded_lc()` — plot phase‑folded light curves  
- `plot_gp()` — visualize GP components  
- `plot_fake_allan_deviation()` — Allan‑deviation proxy plots  
- `plot_corner()` — posterior corner plots  

---

### `TESSData`
MAST query interface for TESS.

- `get_lightcurves()` — download SAP/PDC light curves  
- `get_tpfs()` — download target‑pixel files  
- `ApPhoto()` — aperture photometry on TPFs  

---

### `KeplerData`
MAST query interface for Kepler/K2.

- `get_lightcurves()` — download SAP/PDC light curves  
- `get_tpfs()` — download target‑pixel files  
- `ApPhoto()` — aperture photometry on TPFs  

---

### `SpectroscopicLC`
Spectroscopic light‑curve analysis and fitting.
 
- `generating_lightcurves()` — bin spectra into channels   
- `white_light_lc()` — compute white‑light curve
- `analyse_lc_parallel()` — end‑to‑end light curve analysis using multi-processing
- `plot_parameter_spectrum()` — plot parameter vs wavelength  
- `plot2Ddata()` — 2D time‑wavelength map  
- `plot2D_data_model_resids()` — 2D data/model/residual maps  
- `joint_fake_allan_deviation()` — combined noise vs binning  

---

### `NDImageViewer`
Interactive viewer for 2D/3D/4D image cubes.

- `show()` — display the viewer  

---

### `BrightnessTemperatureCalculator`
Compute brightness temperature from eclipse depths.

- `compute()` — run calculation and return temperature(s) using multi-processing, if an array for `fp` is provided.

---

### `InvertCowanAgolPC`
Invert phase curve observations to thermal properties using Cowan & Agol (2008) model.

- `TdayTnight()` — compute dayside and nightside brightness temperatures  
- `temperature_map_distribution()` — compute 2D temperature map distribution across posterior samples  
- `median_temperature_map()` — compute and optionally plot median temperature map  
- `equatorial_temp_map()` — compute and plot temperature along equator  
- `phase_offsets()` — compute hotspot offset and phase offset from phase curve  
- `albedo_eps_from_temp_map()` — compute Bond albedo and heat redistribution from temperature maps  
- `forward_phase_curve_model()` — compute forward-model phase curve from 2D flux distribution  

---

### `SelectLinDetrend`
Select optimal linear detrending regressors using Bayesian model selection.

- `select_optimal_parameters()` — iteratively identify best combination of linear regressors by maximizing log-evidence  
- Supports multi-instrument datasets with parallel fitting  
- Automatically generates Fourier series regressors from roll angle data  

---

## Utilities (`utils.py`)

Core functions used across modules:

- `t14()` — compute transit duration  
- `b_to_inc()` — impact parameter → inclination  
- `inc_to_b()` — inclination → impact parameter  
- `rho_to_ar()` — stellar density → a/R★  
- `pipe_mad()` — MAD‑based scatter estimator  
- `lcbin()` — bin light curves with gap handling  
- `rms()` — root‑mean‑square statistic  
- `fake_allan_deviation()` — noise vs bin size  
- `corner_plot()` — convenience wrapper for corner plots  
- `make_psd()` — Lomb–Scargle power spectral density  
- `standarize_data()` — standardization helper
- `generate_times_with_gaps()` — simulate time series with gaps  
- `planck_func()` — Planck function for blackbody spectra


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