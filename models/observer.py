import numpy as np
from astropy.coordinates import EarthLocation, AltAz, ITRS, CartesianRepresentation, SkyCoord
from astropy.time import Time

from models.meteor import Meteor
from models.skypointsource import SkyPointSource


class Observer:
    def __init__(self,
                 position: EarthLocation,
                 *,
                 name: str = ""):
        self.position = position
        self.name = name

    def observe(self,
                meteor: Meteor) -> SkyPointSource:

        self.altaz = AltAz(obstime=meteor.time, location=self.position)
        local = meteor.position.get_itrs(meteor.time, location=self.position).transform_to(self.altaz)
        brightness = meteor.brightness / (4 * np.pi * local.distance**2)

        result = SkyPointSource(
            local.alt, local.az, brightness, meteor.time
        )

        return result

    def __str__(self):
        return f"Observer({self.position.to_geodetic()})"