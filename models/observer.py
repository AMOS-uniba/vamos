import dotmap
import numpy as np
from astropy.coordinates import EarthLocation, AltAz, ITRS, CartesianRepresentation, SkyCoord
from astropy.time import Time

from models.meteor import Meteor
from models.skypointsource import SkyPointSource


class Observer:
    def __init__(self,
                 location: EarthLocation,
                 *,
                 name: str = ""):
        self.location = location
        self.name = name

    def observe(self,
                meteor: Meteor) -> SkyPointSource:

        self.altaz = AltAz(obstime=meteor.time, location=self.location)
        local = meteor.position.get_itrs(meteor.time, location=self.location).transform_to(self.altaz)
        brightness = meteor.brightness / (4 * np.pi * local.distance**2)

        result = SkyPointSource(
            local.alt, local.az, brightness, meteor.time
        )

        return result

    def __str__(self):
        return f"Observer({self.location.to_geodetic()})"

    @staticmethod
    def load_dict(data: dict):
        return Observer(
            EarthLocation(
                data['location']['longitude'],
                data['location']['latitude'],
                data['location']['altitude'],
            ),
            name=data['name'],
        )