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

### Dependencies (core)
- `numpy`, `scipy`, `matplotlib`
- `astropy`, `astroquery`
- `tqdm`, `corner`

### Optional dependencies
- `photutils` (aperture photometry)
- `juliet` (Bayesian fitting)
- `dace_query` (CHEOPS archive access)

---

## Quick Start

### ApPhoto (Aperture Photometry)
```python
from krithika import CHEOPSData

# Load subarray data and run aperture photometry
cheops = CHEOPSData(object_name="WASP-189b")
cheops.get_subarrays(pout="./cheops_data")

phot = cheops.ApPhoto(visit_nos=1, aprad=20, sky_rad1=30, sky_rad2=50)
flux, flux_err, _ = phot.simple_aperture_photometry()
```

### CHEOPS: DRP light curves and subarrays
```python
from krithika import CHEOPSData

cheops = CHEOPSData(object_name="WASP-189b")
drp = cheops.get_drp_lightcurves(pout="./cheops_data")

cheops.get_subarrays(pout="./cheops_data")
phot = cheops.ApPhoto(visit_nos=1, aprad=20, sky_rad1=30, sky_rad2=50)
flux, flux_err, _ = phot.simple_aperture_photometry()
```

### Spectroscopic light‑curve analysis
```python
from krithika import SpectroscopicLC
import numpy as np

times = np.load("times.npy")
lc = np.load("spec_lc.npy")
lc_errs = np.load("spec_lc_err.npy")
wavelengths = np.load("wavelengths.npy")

def get_priors(ch_name):
    return "/path/to/priors.dat"

spec = SpectroscopicLC(
    times=times,
    lc=lc,
    lc_errs=lc_errs,
    wavelengths=wavelengths,
    priors=get_priors,
    pout="./results"
)

spec.analyse_lc_parallel(nthreads=4, ch_nos=10)
fig, ax = spec.plot_parameter_spectrum("p_p1", bins=10, plot_white=True)
```

### ND Image Viewer
```python
from krithika import NDImageViewer
import numpy as np

data = np.random.rand(50, 128, 128)
viewer = NDImageViewer(data=data, cmap="magma")
viewer.show()
```

---

## Modules & Classes

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

### `ApPhoto`
Aperture photometry on time‑series cubes:
- circular or brightest‑pixel apertures
- sky annulus handling
- PLD/PCA tools

### `TESSData`, `KeplerData`
MAST queries for light curves and target pixel files.

---

## Utilities (`utils.py`)

- Light‑curve binning (`lcbin`)
- Allan deviation proxy (`fake_allan_deviation`)
- Lomb–Scargle PSD (`make_psd`)
- PCA (`classic_PCA`)
- Orbital geometry utilities (`t14`, `b_to_inc`, `inc_to_b`)
- Brightness temperature calculation (`BrightnessTemperatureCalculator`)

---

## Output Structure (typical)

```
results/
├── spectroscopic_lc_ch_10.pkl
├── CH0/
│   ├── _dynesty_DNS_posteriors.pkl
│   ├── model_resids.dat
│   └── posteriors.dat
├── CH1/
│   └── ...
└── Brightness_temp.npy
```

---

## Notes

- Many functions cache outputs; delete cached files to recompute.
- For CHEOPS downloads, `dace_query` must be installed and configured.
- For fitting, `juliet` must be installed.

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