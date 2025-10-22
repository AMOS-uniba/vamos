import itertools
import logging
from typing import TextIO

import numpy as np

import astropy.units as u
import astropy.constants as const
import yaml
from astropy.coordinates import EarthLocation, CartesianRepresentation, CartesianDifferential
from astropy.time import Time
from astropy.units import Quantity

log = logging.getLogger('root')

from models.skypointsource import SkyPointSource


class Meteor:
    def __init__(self,
                 initial_time: Time,
                 initial_mass: u.Quantity[u.kg],
                 initial_position: EarthLocation,
                 initial_velocity: CartesianDifferential,
                 initial_brightness: u.Quantity[u.W] = 0 * u.W):
        self.time: Time = initial_time
        self.mass: u.Quantity[u.kg] = initial_mass
        self.position: EarthLocation = initial_position
        self.velocity: CartesianDifferential = initial_velocity
        self.brightness: u.Quantity[u.watt] = initial_brightness

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

    @staticmethod
    def acceleration(position: EarthLocation, velocity: CartesianDifferential) -> u.Quantity[u.N]:
        """
        Calculate the gravitational acceleration and inertial accelerations
        acting on a body at defined position and velocity.
        """

        pos = position.get_itrs().cartesian
        r = pos.norm()
        v = velocity.norm()
        omega = CartesianRepresentation(0 * u.rad / u.s, 0 * u.rad / u.s, 7.2921550e-5 * u.rad / u.s).xyz
        return (
            -((const.G * const.M_earth / r**3) * pos).xyz.T
            - 2 * np.cross(omega.to_value(u.rad / u.s), velocity.d_xyz.T.to_value(u.m / u.s)) * u.m / u.s**2
            - np.cross(omega, np.cross(omega, pos.xyz.T)) / (u.rad**2)
        ).to(u.m / u.s**2)

    def __str__(self):
        return f"<Meteor at {self.time[0]}>"

    def __repr__(self):
        return f"<Meteor at {self.time[0]}>"

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
                    'x': float(position.x.to(u.m).value),
                    'y': float(position.y.to(u.m).value),
                    'z': float(position.z.to(u.m).value),
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

    @staticmethod
    def load_yaml(filename: TextIO):
        data = yaml.safe_load(filename)
        log.info(f"Loading meteor from YAML {filename.name}")

        time = Time([frame['time'] for index, frame in data.items()])
        position = EarthLocation.from_geodetic(
            [frame['pos']['lon'] * u.deg for index, frame in data.items()],
            [frame['pos']['lat'] * u.deg for index, frame in data.items()],
            [frame['pos']['alt'] * u.m for index, frame in data.items()],
        )
        velocity = CartesianDifferential(
            [frame['vel']['vx'] for index, frame in data.items()] * u.m/u.s,
            [frame['vel']['vy'] for index, frame in data.items()] * u.m/u.s,
            [frame['vel']['vz'] for index, frame in data.items()] * u.m/u.s,
        )
        brightness = u.Quantity([frame['i'] for index, frame in data.items()]) * u.W
        return Meteor(time, 0 * u.kg, position, velocity, brightness)