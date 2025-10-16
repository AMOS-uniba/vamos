import itertools
import math
from typing import TextIO

import numpy as np

import astropy.units as u
import yaml
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity


class PointSource:
    """
    A point source moving in the sky, defined as the brightness
    of the source in the alt-az system at a specified time, I(alt, az, t)
    as a sequence of positions, times and brightnesses.
    Can be interpolated to any specified time,
    or extrapolated (in which case the brightness is assumed to be zero).
    """

    def __init__(self,
                 alt: Angle,            # Angle
                 az: Angle,             # Angle
                 intensity: Quantity,   # Watt per square metre
                 time: Time):
        assert alt.shape == az.shape == intensity.shape == time.shape, \
            f"Shapes do not match: {alt.shape=}, {az.shape=}, {intensity.shape=}, {time.shape=}"

        self._alt = alt
        self._az = az
        self._intensity = intensity
        self._time = time

        assert self._intensity.unit.is_equivalent(u.W / u.m**2)

    @property
    def alt(self):
        return self._alt

    @property
    def az(self):
        return self._az

    @property
    def intensity(self):
        return self._intensity

    @property
    def time(self):
        return self._time

    def at_time(self, time: Time) -> tuple[Angle, Angle, Quantity]:
        """
        Return positions and intensities at time, as an (inter|extra)polation.
        """
        time = time.jd2
        stime = self.time.jd2
        alt = np.interp(time, stime, self.alt)
        az = np.interp(time, stime, self.az, period=math.tau)
        intensity = np.interp(time, stime, self.intensity, left=0, right=0)
        return alt, az, intensity

    def __str__(self):
        return f"{self.alt=}, {self.az=}, {self.intensity=}, {self.time=}"

    def __repr__(self):
        return self.__str__()

    def as_dict(self):
        return {
            index: {
                'time': time.iso,
                'alt': float(alt.to(u.deg).value),
                'az': float(az.to(u.deg).value),
                'i': float(u.to(u.Wm2).value),
            }
            for index, time, alt, az, intensity in zip(itertools.count, self.time, self.alt, self.az, self.intensity)
        }

    def dump_yaml(self, file: TextIO):
        yaml.safe_dump(self.as_dict(), file)
