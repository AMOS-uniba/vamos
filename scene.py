from abc import abstractmethod

import numpy as np

from astropy.coordinates import EarthLocation
from astropy.time import Time

from numpy.typing import NDArray

from effects.sky import SkySource


class DetectorReadout:
    def __init__(self,
                 xres: int,
                 yres: int):
        self._data = np.zeros(shape=(yres, xres))
        self.xs, self.ys = np.meshgrid(np.arange(0, yres), np.arange(0, xres))

    def add_source(self,
                   effect: SkySource):
        self._data = effect(self.xs, self.ys)

    @property
    def data(self):
        return self._data

