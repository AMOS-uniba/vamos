import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from amosutils.projections.shifters import Shifter, ScalingShifter
from numpy.typing import ArrayLike

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from amosutils.projections import BorovickaProjection, Projection

from effects.sky import SkySource, Sunlight, Airglow

mpl.use('qtagg')


class Scene:
    def __init__(self,
                 xres: int,
                 yres: int,
                 projection: Projection,
                 scaler: ScalingShifter,
                 location: EarthLocation,
                 time: Time = None):
        self.xres = xres
        self.yres = yres
        self._data = np.zeros(shape=(yres, xres))
        self.xs, self.ys = np.meshgrid(np.arange(0, xres), np.arange(0, yres))

        self.location = location
        self.time = Time.now() if time is None else time
        self.projection = projection
        self.scaler = scaler

        # Pre-compute sky altitude and azimuth for every pixel in the scene
        self.alt, self.az = self.projection(*self.scaler(self.xs, self.ys))
        self.alt = np.pi / 2 - self.alt

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data

    def build(self):
        source = Sunlight(self.location, self.time)
        airglow = Airglow(self.location, self.time, intensity=0.1)
        self.add_sky_effects([airglow])

    def render(self, filename = None):
        """
        Render the scene into a numpy array
        """
        plt.figure(figsize=(16, 12))
        plt.imshow(self.data, cmap='Grays_r', norm=mpl.colors.Normalize(vmin=0, vmax=1), origin='lower')
        plt.tight_layout()
        plt.title(f"{self.location} {self.time}")

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

        plt.close('all')

    def add_sky_effects(self,
                        sources: list[SkySource]):
        for source in sources:
            self.data = source(self.data, self.alt, self.az)

    def add_points(self,
                   alt: ArrayLike,
                   az: ArrayLike,
                   intensities: ArrayLike) -> None:
        """
        Add a collection of point sources (defined by `alt`, `az`) to the scene.
        """
        assert alt.shape == az.shape == intensities.shape, \
            f"Coordinates and intensities have a wrong shape: {alt.shape=}, {az.shape=}, {intensities.shape=}"

        # Obtain coordinates in the detector coordinates
        print(alt, az)
        mx, my = self.projection.invert(np.pi / 2 - alt, az)
        print(mx, my)
        x, y = self.scaler.invert(mx, my)
        print(x, y)

        mask = (x >= 0) & (x < self.xres) & (y >= 0) & (y < self.yres)
        nx = np.int_(x[mask])
        ny = np.int_(y[mask])
        intensities = intensities[mask]

        # Thanks to ChatGPT this sort of works
        flat_idx = ny * self.xres + nx
        arr_flat = np.bincount(flat_idx, weights=intensities, minlength=self.data.size)
        self.data += arr_flat.reshape(self.data.shape)
