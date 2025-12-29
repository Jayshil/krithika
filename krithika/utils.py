import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from astropy.timeseries import LombScargle
import astropy.constants as con
import astropy.units as u
import corner

def t14(per, ar, rprs, b, ecc=0, omega=90, transit=True):
    """
    To compute transit/eclipse duration from Period, a/R*, Rp/R*, b, eccentricity, and omega

    Parameters:
    -----------
    per : float, or numpy.ndarray
        Orbital period (in days) of the planet
    aR : float, or numpy.ndarray
        Scaled semi-major axis, a/R*
    rprs : float, or numpy.ndarray
        Planet-to-star radius ratio, Rp/R*
    bb : float, or numpy.ndarray
        Impact parameter
    ecc: float, or numpy.ndarray
        Eccentricity of the orbit
    omega: float, or numpy.ndarray
        Argument of periastron (deg)
    transit: bool
        Whether to compute transit or occultation duration
    -----------
    return
    -----------
    t14 : float, or numpy.ndarray
        Transit/occultation duration, in days
    """

    inc_rad = np.radians( b_to_inc(b=b, ar=ar, ecc=ecc, omega=omega) )
    omega = np.radians(omega)

    # Computing inclination from impact parameter
    term1 = np.sqrt( (1 + rprs)**2 -  b**2  )
    t14 = ( per / np.pi ) * np.arcsin( term1 / ar / np.sin( inc_rad ) )

    # Factor accounting for eccentric orbit
    if transit:
        fac = np.sqrt( 1 - ecc**2 ) / ( 1 + (ecc*np.sin(omega)) )
    else:
        fac = np.sqrt( 1 - ecc**2 ) / ( 1 - (ecc*np.sin(omega)) )
    
    return t14 * fac

def b_to_inc(b, ar, ecc=0., omega=90., transit=True):
    """Computing inclination from impact parameter"""
    # First converting all angles to radians
    omega = np.radians(omega)
    if transit:
        cosi = b * ( 1 + ( ecc * np.sin(omega) ) ) / ar / (1 - ecc**2)
    else:
        cosi = b * ( 1 - ( ecc * np.sin(omega) ) ) / ar / (1 - ecc**2)
    return np.rad2deg( np.arccos( cosi ) )

def inc_to_b(inc, ar, ecc=0., omega=90., transit=True):
    """To compute impact parameter from inclination"""
    # First convert all angles to radians
    inc, omega = np.radians(inc), np.radians(omega)
    if transit:
        bb = ar * np.cos(inc) * (1 - ecc**2) / ( 1 + ( ecc * np.sin(omega) ) )
    else:
        bb = ar * np.cos(inc) * (1 - ecc**2) / ( 1 - ( ecc * np.sin(omega) ) )
    return bb

def rho_to_ar(rho, per):
    """To compute a/R* from rho"""
    G = con.G.value
    a = ((rho * G * ((per * 24. * 3600.)**2)) / (3. * np.pi))**(1. / 3.)
    return a

def pipe_mad(x, axis=0):
    return np.nanmedian(np.abs(np.diff(x, axis=axis)), axis=axis)

def lcbin(time, flux, binwidth=0.06859, nmin=4, time0=None,
        robust=False, tmid=False):
    """
    ------ A function from pycheops -------
    Calculate average flux and error in time bins of equal width.
    The default bin width is equivalent to one CHEOPS orbit in units of days.
    To avoid binning data on either side of the gaps in the light curve due to
    the CHEOPS orbit, the algorithm searches for the largest gap in the data
    shorter than binwidth and places the bin edges so that they fall at the
    centre of this gap. This behaviour can be avoided by setting a value for
    the parameter time0.
    The time values for the output bins can be either the average time value
    of the input points or, if tmid is True, the centre of the time bin.
    If robust is True, the output bin values are the median of the flux values
    of the bin and the standard error is estimated from their mean absolute
    deviation. Otherwise, the mean and standard deviation are used.
    The output values are as follows.
    * t_bin - average time of binned data points or centre of time bin.
    * f_bin - mean or median of the input flux values.
    * e_bin - standard error of flux points in the bin.
    * n_bin - number of flux points in the bin.
    :param time: time
    :param flux: flux (or other quantity to be time-binned)
    :param binwidth:  bin width in the same units as time
    :param nmin: minimum number of points for output bins
    :param time0: time value at the lower edge of one bin
    :param robust: use median and robust estimate of standard deviation
    :param tmid: return centre of time bins instead of mean time value
    :returns: t_bin, f_bin, e_bin, n_bin
    """
    if time0 is None:
        tgap = (time[1:]+time[:-1])/2
        gap = time[1:]-time[:-1]
        j = gap < binwidth
        gap = gap[j]
        tgap = tgap[j]
        time0 = tgap[np.argmax(gap)]
        time0 = time0 - binwidth*np.ceil((time0-min(time))/binwidth)

    n = int(1+np.ceil(np.ptp(time)/binwidth))
    r = (time0,time0+n*binwidth)
    n_in_bin,bin_edges = np.histogram(time,bins=n,range=r)
    bin_indices = np.digitize(time,bin_edges)

    t_bin = np.zeros(n)
    f_bin = np.zeros(n)
    e_bin = np.zeros(n)
    n_bin = np.zeros(n, dtype=int)

    for i,n in enumerate(n_in_bin):
        if n >= nmin:
            j = bin_indices == i+1
            n_bin[i] = n
            if tmid:
                t_bin[i] = (bin_edges[i]+bin_edges[i+1])/2
            else:
                t_bin[i] = np.nanmean(time[j])
            if robust:
                f_bin[i] = np.nanmedian(flux[j])
                e_bin[i] = 1.25*np.nanmean(abs(flux[j] - f_bin[i]))/np.sqrt(n)
            else:
                f_bin[i] = np.nanmean(flux[j])
                e_bin[i] = np.std(flux[j])/np.sqrt(n-1)

    j = (n_bin >= nmin)
    return t_bin[j], f_bin[j], e_bin[j], n_bin[j]

def rms(x):
    return np.sqrt( np.nanmean( (x - np.nanmean(x))**2 ) )

def fake_allan_deviation(times, residuals, binmax=10, method='pipe', timeunit=None, plot=True):
    """Estimate and optionally plot a noise-vs-bin-size curve (a proxy for Allan deviation).

    The function computes the scatter of binned residuals for a sequence
    of integer bin sizes and returns both the computed noise (in ppm) and
    a white-noise expectation for comparison. When ``plot=True`` a
    log-log figure is produced with a secondary x-axis that maps bin size
    to a time-unit (days/hours/minutes) according to the total span of
    ``times``.

    Parameters
    ----------
    times : array-like
        Time stamps of the observations (assumed in days).
    residuals : array-like
        Residual fluxes (same length as ``times``).
    binmax : int, optional
        Parameter controlling the maximum number of bins. The function
        will evaluate binsizes from 1..int(len(times)/binmax). Default 10.
    method : {'pipe', 'std', 'rms', 'astropy'}, optional
        Method used to estimate the scatter of the binned residuals:
        - 'pipe' : use ``pipe_mad`` (median absolute differences estimator)
        - 'std'  : use ``np.nanstd``
        - 'rms'  : use the root-mean-square (``rms`` helper)
        - 'astropy' : use ``astropy.stats.mad_std`` (robust MAD-based std)
    timeunit : {'d', 'hr', 'min'}, optional
        If provided, forces the time unit used in the secondary x-axis
        to days ('d'), hours ('hr'), or minutes ('min'). By default the
        function selects the most appropriate unit based on the total
        span of ``times``.
    plot : bool, optional
        If ``True`` (default), create and display a matplotlib figure with
        the computed noise curve and the white-noise expectation.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The created figure if ``plot=True``, otherwise ``None``.
    axs : matplotlib.axes.Axes or None
        The axes object for the figure if ``plot=True``, otherwise ``None``.
    binsize : ndarray
        Integer array of evaluated bin sizes.
    noise_ppm : ndarray
        Computed scatter of binned residuals converted to parts-per-million
        (ppm), corresponding to ``binsize``.
    white_noise_expec : ndarray
        White-noise expectation (ppm) computed as
        ``noise_func(residuals) / sqrt(binsize)`` (with a small-sample
        correction applied). Values for ``binsize==0`` will be ``inf`` or
        undefined.
    """

    # Computing bin-size
    binsize = np.arange(1, int(len(times)/binmax), 1)
    # Array to save noise
    noise = np.zeros( len(binsize) )
    # Number of data points in each bin
    nbin = np.zeros( len(binsize) )

    # Choosing the method to compute noise
    if method == 'pipe':
        noise_func = pipe_mad
    elif method == 'std':
        noise_func = np.nanstd
    elif method == 'rms':
        noise_func = rms
    elif method == 'astropy':
        noise_func = lambda x: mad_std(x, ignore_nan=True)
    else:
        raise ValueError("Method should be one of 'pipe', 'std', 'rms', or 'astropy'.")

    # Computing binning for each binsize, and then computing the stddev of the binned residuals
    for i in range(len(binsize)):
        _, binned_flux, _, _ = lcbin(time=times, flux=residuals, nmin=1, binwidth=binsize[i] * np.nanmedian(np.diff(times)) )
    
        noise[i] = noise_func(binned_flux)
        nbin[i] = int( np.floor( len(residuals) / binsize[i] ) )

    # Converting binsize to time units
    time_binsize = binsize * np.nanmedian( np.diff(times) )

    # First, let's estimate the time units we need from times array:
    if timeunit is None:
        if np.ptp(time_binsize) >= 1.:
            ## If times duration is greater than 2 hours, we use days
            time_unit_multiplication_factor = 1.     # Units are already in days
            time_unit_label = 'Time period [d]'
        elif ( np.ptp(time_binsize) > 5/24 ) and ( np.ptp(time_binsize) < 1. ):
            ## If times duration is greater than 2 hours, but less than 2 days, we use hours
            time_unit_multiplication_factor = 24.    # Converting days to hours
            time_unit_label = 'Time period [hr]'
        else:
            ## If times duration is less than 2 hours, we use minutes
            time_unit_multiplication_factor = 24 * 60
            time_unit_label = 'Time period [min]'
    else:
        if timeunit == 'd':
            time_unit_multiplication_factor = 1.
            time_unit_label = 'Time period [d]'
        elif timeunit == 'hr':
            time_unit_multiplication_factor = 24.
            time_unit_label = 'Time period [hr]'
        elif timeunit == 'min':
            time_unit_multiplication_factor = 24 * 60
            time_unit_label = 'Time period [min]'
        else:
            raise ValueError("timeunit should be one of 'd', 'hr', or 'min'.")
    
    # The following two functions will convert binsize to time (in minutes) and vice versa
    def bin2time(binsize):
        return binsize * np.nanmedian(np.diff(times)) * time_unit_multiplication_factor

    def time2bin(bintime):
        return bintime / ( np.nanmedian(np.diff(times)) * time_unit_multiplication_factor )
    
    # Computing the white-noise expectation
    white_noise_expec = noise_func(residuals)/np.sqrt(binsize) * 1e6 * np.sqrt(nbin / (nbin - 1.))
    
    if plot:
        # ----------------------------------
        #  And generating the plot
        # ----------------------------------
        fig, axs = plt.subplots()
        
        ## First plotting the computed noise
        axs.plot(binsize, noise * 1e6, color='dodgerblue', lw=1., label='Computed noise', zorder=10)
        ## Now plotting the white-noise expectation
        axs.plot(binsize, white_noise_expec, color='orangered', label='White-noise expectation')

        ## Adding secondary x-axis for time
        secax = axs.secondary_xaxis('top', functions=(bin2time, time2bin))
        secax.set_xlabel(time_unit_label, labelpad=10)

        axs.set_xscale('log')
        axs.set_yscale('log')

        axs.set_xlabel('Bin size [Number of points]')
        axs.set_ylabel('Noise estimate [ppm]')

        axs.legend()
    else:
        fig, axs = None, None

    return fig, axs, binsize, noise*1e6, white_noise_expec


def corner_plot(samples, labels, **kwargs):
    """Generate a corner plot from MCMC samples.

    Parameters
    ----------
    samples : ndarray
        2D array of shape (n_samples, n_parameters) containing the MCMC samples.
    labels : list of str
        List of parameter names for labeling the axes.
    figsize : tuple, optional
        Size of the figure to create. Default is (9, 9).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated corner plot figure.
    """

    samples = np.transpose( np.vstack( samples ) )

    fig = corner.corner(samples, labels=labels,
                        show_titles=True, title_fmt=".2f",
                        quantiles=[0.16,0.5,0.84], **kwargs)

    return fig

def make_psd(times, flux, plot=True, plot_max_freq=True, timeunit=None):
    """Compute a Lomb-Scargle power spectral density (PSD) for a light curve.

    The routine converts ``times`` (assumed in days) to seconds and the
    input ``flux`` to parts-per-million (ppm), computes a dense frequency
    grid, evaluates a Lomb-Scargle PSD on that grid and (optionally)
    produces a log-log plot with a secondary axis showing period in an
    appropriate time unit.

    Parameters
    ----------
    times : array-like
        Time stamps of the observations (in days).
    flux : array-like
        Relative flux measurements (unitless, e.g. normalised to 1.0). The
        function converts these to ppm internally.
    plot : bool, optional
        If ``True`` (default) create and return a matplotlib figure and
        axes containing the PSD plot. If ``False``, no plot is created and
        ``fig`` and ``axs`` in the return tuple will be ``None``.
    plot_max_freq : bool, optional
        If ``True`` (default) mark the frequency with maximum power on the
        plot with a vertical dashed line.
    timeunit : {'d','hr','min'} or None, optional
        Force the secondary x-axis unit for the period conversion. If
        ``None`` (default) the function selects days/hours/minutes
        automatically based on the frequency grid span.

    Returns
    -------
    freq_grid : astropy.units.Quantity (Hz)
        Frequency grid used to evaluate the PSD.
    psd1 : astropy.units.Quantity
        Lomb-Scargle power spectral density evaluated on ``freq_grid``
        (units: ppm^2 Hz^-1).
    fig : matplotlib.figure.Figure or None
        Figure with the PSD plot when ``plot=True``, otherwise ``None``.
    axs : matplotlib.axes.Axes or None
        Axes for the PSD plot when ``plot=True``, otherwise ``None``.
    per_max_pow : astropy.units.Quantity
        Period corresponding to the maximum power in the PSD (converted
        to days).

    Notes
    -----
    The function uses ``astropy.timeseries.LombScargle`` for the PSD and
    multiplies the input flux by 1e6 to report power in ppm^2 Hz^-1.
    """

    # Computing time in units of days to seconds
    times = times * 24. * 60. * 60. * u.s
    # Converting flux to ppm
    fl1 = 1e6 * flux * u.dimensionless_unscaled

    ## Frequency grid
    min_freq, max_freq = 1 / np.ptp(times), 1 / np.nanmedian(np.diff(times)) * 0.5
    freq_grid = np.linspace(min_freq, max_freq, 100000)

    ## And the Lomb-Scargle periodogram
    psd1 = LombScargle(times, fl1, normalization='psd').power(frequency=freq_grid)
    
    ## Computing the period corresponding to the highest power
    idx_max_pow = np.argmax(psd1.value)
    freq_max_pow = freq_grid[idx_max_pow]
    per_max_pow = ( 1/freq_max_pow ).to(u.day)

    # First, let's estimate the time units we need from times array:
    time_grid = (1 / freq_grid ).to(u.d).value
    if timeunit is None:
        if np.ptp(time_grid) >= 1.:
            ## If times duration is greater than 2 hours, we use days
            time_unit = u.d
            time_unit_label = 'Time [d]'
        elif ( np.ptp(time_grid) > 5/24 ) and ( np.ptp(time_grid) < 1. ):
            ## If times duration is greater than 2 hours, but less than 2 days, we use hours
            time_unit = u.hr    # Converting days to hours
            time_unit_label = 'Time [hr]'
        else:
            ## If times duration is less than 2 hours, we use minutes
            time_unit = u.min
            time_unit_label = 'Time [min]'
    else:
        if timeunit == 'd':
            time_unit = u.d
            time_unit_label = 'Time [d]'
        elif timeunit == 'hr':
            time_unit = u.hr
            time_unit_label = 'Time [hr]'
        elif timeunit == 'min':
            time_unit = u.min
            time_unit_label = 'Time [min]'
        else:
            raise ValueError("timeunit should be one of 'd', 'hr', or 'min'.")

    if plot:
        # Making the plot:
        fig, axs = plt.subplots()

        ## Un-binned data
        axs.plot(freq_grid, psd1, alpha=1., color='orangered', lw=1., zorder=10)
        if plot_max_freq:
            axs.axvline(freq_max_pow.value, ls='--', c='cornflowerblue', lw=1., zorder=5)

        # Define properties to define upper axis as well:
        def freq2tim(x):
            x = x * u.Hz
            return (1/x).to(time_unit).value
        
        def tim2freq(x):
            x = x * u.s
            return (1/x).to(u.Hz).value

        ax2 = axs.secondary_xaxis("top", functions=(freq2tim, tim2freq))
        ax2.tick_params(axis='both', which='major')
        ax2.set_xlabel(time_unit_label, labelpad=10)

        axs.set_xscale('log')
        axs.set_yscale('log')
        
        axs.set_xlabel(r'Frequency [Hz]')
        axs.set_ylabel(r'Power [ppm$^2$ Hz$^{-1}$]')
        
        axs.set_xlim([np.min(freq_grid.value), np.max(freq_grid.value)])
    else:
        fig, axs = None, None

    return freq_grid, psd1, fig, axs, per_max_pow