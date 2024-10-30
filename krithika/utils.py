import jax.numpy as jnp
from jaxoplanet.orbits import keplerian
from jaxoplanet.units import unit_registry as ureg
import astropy.constants as con

# Some constants that we need
G = con.G.value
rho_sun = (con.M_sun/(con.R_sun**3)).value

import jax
jax.config.update(
    "jax_enable_x64", True
)  # For 64-bit precision since JAX defaults to 32-bit

def init_keplerian(nplanets, inc=False):
    """Initialising jaxoplanet Keplerian orbit"""
    # Central body (star; density in kg/m3)
    # Jaxoplanet takes mass and radius of the star and planet the in Solar units
    # However, we may not know the stellar mass and radius in the beginning
    # Besides, Rp/R* is the primary observable of the transit and not Rp
    # So, we have created a system with Rp=R_sun and stellar density relative to that of the Sun
    # Moreover, since Rs=R_sun, we can set Rp=Rp/Rs R_sun units
    # The code below tested against batman and radvel models, and these models are consistent with 
    # what is computed below
    star = keplerian.Central(density=400./rho_sun)
    # Planets
    planets = []
    for _ in range(nplanets):
        if inc:
            body = keplerian.Body(time_transit=0., period=5., inclination=jnp.radians(inc), eccentricity=0., omega_peri=jnp.pi/2, radius=0.1, radial_velocity_semiamplitude=20.*ureg.m/ureg.s)
        else:
            body = keplerian.Body(time_transit=0., period=5., impact_param=0.5, eccentricity=0., omega_peri=jnp.pi/2, radius=0.1, radial_velocity_semiamplitude=20.*ureg.m/ureg.s)
        planets.append(body)

    return star, planets