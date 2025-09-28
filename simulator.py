import numpy as np

from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.time import Time

from amosutils.projections import Projection, BorovickaProjection
from amosutils.catalogue import Catalogue


class MeteorSimulator:
    def __init__(self):
        self.catalogue = Catalogue('HYG30.tsv')
        self.projection = BorovickaProjection(V=5)
        self.location = EarthLocation.from_geodetic(48 * u.deg, 17 * u.deg, 185 * u.m)
        self.time = Time()
