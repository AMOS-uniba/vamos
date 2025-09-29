import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from amosutils.projections.shifters import Shifter, ScalingShifter
from numpy.typing import ArrayLike
from PIL import Image

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
        self.data = np.zeros(shape=(yres, xres))
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

    def render_raw(self, filename = None):

        Image.fromarray(np.flip(self.data * 255, axis=0).astype(np.uint8)).save(filename)

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
        mx, my = self.projection.invert(np.pi / 2 - alt, az)
        x, y = self.scaler.invert(mx, my)

        mask = (x >= 0) & (x < self.xres) & (y >= 0) & (y < self.yres)
        nx, ny = x[mask], y[mask]
        intensities = intensities[mask] / 5
        sigmas = np.ones_like(nx) * 0.5
        truncate = 4.0

        # ChatGPT: Loop over deltas, but vectorized over each patch
        for xi, yi, ai, si in zip(nx, ny, intensities, sigmas):
            rad = int(np.ceil(truncate * si))
            xmin, xmax = max(0, int(np.floor(xi)) - rad), min(self.xres, int(np.floor(xi)) + rad + 1)
            ymin, ymax = max(0, int(np.floor(yi)) - rad), min(self.yres, int(np.floor(yi)) + rad + 1)

            # Local patch grid
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
            g = np.exp(-((xx - xi)**2 + (yy - yi)**2) / (2 * si**2))

            # Normalize and scale
            g /= g.sum()
            g *= ai

            # Add patch to result
            self.data[ymin:ymax, xmin:xmax] += g

    def add_gaussian_noise(self, sigma=0.1):
        noise = np.random.normal(1, sigma, size=(self.yres, self.xres))
        self.data *= noise

    def add_thermal_noise(self, lam=0.1):
        noise = np.random.poisson(lam, size=(self.yres, self.xres)) * 0.05
        self.data += noise