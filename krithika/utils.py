import numpy as np

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