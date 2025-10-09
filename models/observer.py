import numpy as np
from astropy.coordinates import EarthLocation, AltAz, ITRS, CartesianRepresentation

from .meteor import Meteor
from pointsource import PointSource


class Observer:
    def __init__(self,
                 position: EarthLocation):
        self.position = position
        self.altaz = AltAz(location=self.position)

    def observe(self,
                meteor: Meteor) -> PointSource:
        pos = CartesianRepresentation(meteor.position.to_geocentric())
        position = ITRS(pos).transform_to(self.altaz)
        brightness = meteor.brightness / (4 * np.pi * position.distance**2)

        result = PointSource(
            position.alt, position.az, brightness, meteor.time
        )

        return result
