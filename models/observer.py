import numpy as np

from astropy.coordinates import EarthLocation, AltAz, ITRS, CartesianRepresentation, SkyCoord
import astropy.units as u

from enschema import Schema

from models.meteor import Meteor
from models.skypointsource import SkyPointSource


class Observer:
    _schema = Schema({
        'name': str,
        'location': {
            'latitude': float,
            'longitude': float,
            'altitude': float,
        }
    })

    def __init__(self,
                 location: EarthLocation,
                 *,
                 name: str = ""):
        self.location = location
        self.name = name

    def observe(self,
                meteor: Meteor) -> SkyPointSource:

        self.altaz = AltAz(obstime=meteor.time, location=self.location)
        local = meteor.position.get_itrs(obstime=meteor.time, location=self.location).transform_to(self.altaz)
        brightness = meteor.brightness / (4 * np.pi * local.distance**2)

        result = SkyPointSource(local.alt, local.az, local.distance, brightness, meteor.time)

        return result

    def __str__(self):
        return f"Observer({self.location.to_geodetic()})"

    def as_dict(self):
        return self._schema.validate({
            'name': self.name,
            'location': {
                'latitude': float(self.location.lat.to(u.deg).value),
                'longitude': float(self.location.lon.to(u.deg).value),
                'altitude': float(self.location.height.to(u.m).value),
            }
        })

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