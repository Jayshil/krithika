import numpy as np
import astropy.constants as con

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