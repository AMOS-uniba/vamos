import math
import numpy as np

from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity


class PointSource:
    """
    A collection of point light sources in the sky, in the alt-az system.
    """

    def __init__(self,
                 alt: Angle,
                 az: Angle,
                 intensity: Quantity,
                 time: Time):
        assert alt.shape == az.shape == intensity.shape == time.shape, \
            f"Shapes do not match: {alt.shape=}, {az.shape=}, {intensity.shape=}, {time.shape=}"

        self._alt = alt
        self._az = az
        self._intensity = intensity
        self._time = time

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
        Return positions and intensities at time, as an interpolation
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
