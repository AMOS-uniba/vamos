import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from amosutils.projections import BorovickaProjection

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from effects.sky import SkySource, Sunlight, Airglow

mpl.use('qtagg')

class DetectorReadout:

    def add_source(self,
                   effect: SkySource):
        self._data = effect(self.xs, self.ys)


class Scene:
    def __init__(self,
                 xres: int,
                 yres: int,
                 time: Time = None):
        self._data = np.zeros(shape=(yres, xres))
        self.xs, self.ys = np.meshgrid(np.arange(0, xres), np.arange(0, yres))
        self.projection = BorovickaProjection(800, 600, 0, 0, 0, 0.0025)

        self.location = EarthLocation(17 * u.deg, 48 * u.deg, 531 * u.m)
        self.time = Time.now() if time is None else time

    @property
    def data(self):
        return self._data

    def render(self, filename = None):
        alt, az = self.projection(self.xs, self.ys)
        alt = np.pi / 2 - alt

        source = Sunlight(self.location, self.time)
        self._data = source(self.data, alt, az)
        airglow = Airglow(self.location, self.time, intensity=0.1)
        self._data = airglow(self.data, alt, az)

        plt.figure(figsize=(16, 12))
        plt.imshow(self.data, cmap='Grays_r', norm=mpl.colors.Normalize(vmin=0, vmax=1))
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

        plt.close('all')