#!/usr/bin/env python
import argparse
import yaml

import numpy as np
import matplotlib as mpl
import dotmap
from amosutils.projections.shifters import ScalingShifter

from matplotlib import pyplot as plt

from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u

from dotcollection import SkyDotCollection
from amosutils.projections import Projection
from amosutils.catalogue import Catalogue

from scene import Scene


class MeteorSimulatorCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--meteor-file', '-m')
        self.parser.add_argument('config', type=argparse.FileType('r'))
        self.args = self.parser.parse_args()

        self.config = dotmap.DotMap(yaml.safe_load(self.args.config), _dynamic=False)

        self.location = EarthLocation(self.config.location.longitude, self.config.location.latitude, self.config.location.altitude)
        self.catalogue = Catalogue(self.config.catalogue)
        self.projection = Projection.from_dict(self.config['projection'])
        self.scaler = ScalingShifter(x0=self.config.pixels.x0, y0=self.config.pixels.y0,
                                     xs=self.config.pixels.xs, ys=self.config.pixels.ys)

        self.simulate(Time(self.config.start))

    def simulate(self, time: Time):
        for i in range(0, 100):
            stime = time + i * 3 * u.s
            scene = Scene(self.config.detector.xres, self.config.detector.yres,
                          projection=self.projection,
                          scaler=self.scaler,
                          location=self.location,
                          time=stime)

            altaz = self.catalogue.altaz(self.location, time=stime, masked=False)
            mask = altaz.alt > 0
            self.catalogue.mask = mask
            altaz = altaz[mask]

            scene.build()
            ints = 100 * np.exp(-1.5 * self.catalogue.vmag(self.location, masked=True))
            scene.add_points(altaz.alt.radian, altaz.az.radian, ints)

            scene.add_intensifier_noise(lam=0.1, intensity=0.25, spatial_sigma=50, brightening=0.4)
            scene.add_gaussian_noise(sigma=0.1)
            scene.add_thermal_noise(lam=0.1, intensity=0.1)

            scene.render(f'output/{i:03}.png')


simulator = MeteorSimulatorCLI()