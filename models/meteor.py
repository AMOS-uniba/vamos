import itertools
from typing import TextIO

import numpy as np

import astropy.units as u
import yaml
from astropy.coordinates import EarthLocation, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
from astropy.units import Quantity

from pointsource import PointSource


class Meteor:
    def __init__(self,
                 initial_time: Time,
                 initial_mass: u.Quantity[u.kg],
                 initial_position: EarthLocation,
                 initial_velocity: CartesianDifferential):
        self.time: Time = initial_time
        self.mass: u.Quantity[u.kg] = initial_mass
        self.position: EarthLocation = initial_position
        self.velocity: CartesianDifferential = initial_velocity
        self.brightness: u.Quantity[u.watt] = 0 * u.watt

    def simulate(self,
                 steps: int,
                 dt: u.Quantity[u.s]):
        n = 0
        pos = [self.position]
        vel = [self.velocity]

        while n < steps:
            cp = CartesianRepresentation(pos[-1].to_geocentric())
            cv = vel[-1]
            newpos = cp + cv * dt
            pos.append(EarthLocation.from_geocentric(newpos.x, newpos.y, newpos.z))
            vel.append(cv)

            n += 1

        self.position = EarthLocation.from_geocentric(
            u.Quantity([p.x for p in pos]),
            u.Quantity([p.y for p in pos]),
            u.Quantity([p.z for p in pos]),
        )
        self.velocity = CartesianDifferential(
            u.Quantity([v.d_x for v in vel]),
            u.Quantity([v.d_y for v in vel]),
            u.Quantity([v.d_z for v in vel]),
        )
        self.time = Time(
            self.time + np.arange(0, len(self.position)) * dt,
        )
        tau = np.linspace(0, 1, steps + 1)
        #self.brightness = 1e6 * u.W * (1 - tau)**3 * tau**5
        self.brightness = 1e4 * u.W * np.ones_like(tau)
        #self.brightness = 1e6 * u.W * np.exp(- self.position.height / u.km / 10)

    def as_dict(self):
        """
        Returns a dictionary representation of the meteor, suitable for saving.
        """
        return {
            index: {
                'time': time.iso,
                'pos': {
                    'lat': float(position.lat.value),
                    'lon': float(position.lon.value),
                    'alt': float(position.height.to(u.m).value),
                },
                'vel': {
                    'vx': float(velocity.d_x.value),
                    'vy': float(velocity.d_y.value),
                    'vz': float(velocity.d_z.value),
                },
                'i': float(brightness.to(u.W).value),
            }
            for index, time, position, velocity, brightness in
            zip(itertools.count(), self.time, self.position, self.velocity, self.brightness)
        }

    def dump_yaml(self, filename: TextIO):
        """
        Dumps the meteor to a YAML file.
        """
        yaml.safe_dump(self.as_dict(), filename)