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

from effects.sky import SkySource, Sunlight, Airglow, Moonlight

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
        sun = Sunlight(self.location, self.time)
        moon = Moonlight(self.location, self.time)
        airglow = Airglow(self.location, self.time, intensity=0.04)
        self.add_sky_effects([sun, moon, airglow])

    def render(self, filename = None):
        print(f"Rendering {filename} {self.data.T.shape}")
        bitmap = np.flip(self.data * 255, axis=0).astype(np.uint8)
        Image.fromarray(bitmap).save(filename)

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
        for xi, yi, ii, si in zip(nx, ny, intensities, sigmas):
            rad = int(np.ceil(truncate * si) + 1)
            xmin, xmax = max(0, int(np.floor(xi)) - rad), min(self.xres, int(np.floor(xi)) + rad + 1)
            ymin, ymax = max(0, int(np.floor(yi)) - rad), min(self.yres, int(np.floor(yi)) + rad + 1)

            # Local patch grid
            yy, xx = np.mgrid[ymin:ymax, xmin:xmax]
            g = np.exp(-((xx - xi)**2 + (yy - yi)**2) / (2 * si**2))

            # Normalize and scale
            g /= g.sum()
            g *= ii

            # Add the patch to result
            self.data[ymin:ymax, xmin:xmax] += g

    def add_intensifier_noise(self,
                              lam: float = 0.1,
                              *,
                              spatial_sigma: float = 50,
                              intensity: float = 0.1,
                              brightening: float = 0.5):
        noise = (brightening + np.random.poisson(lam, size=(self.yres, self.xres))) * intensity
        xc, yc = self.xres / 2, self.yres / 2
        profile = np.exp(-((self.xs - xc)**2 + (self.ys - yc)**2) / (2 * spatial_sigma**2))
        self.data += noise * profile

    def add_gaussian_noise(self, sigma=0.1):
        noise = np.random.normal(1, sigma, size=(self.yres, self.xres))
        self.data *= noise

    def add_thermal_noise(self, lam=0.1, intensity=0.05):
        noise = np.random.poisson(lam, size=(self.yres, self.xres)) * intensity
        self.data += noise