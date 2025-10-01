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

from pointsource import PointSource
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


        self.dz = 0.06
        self.da = -1.31
        t0 = Time(self.config.start)
        x = np.linspace(0, 1, self.config.count + 1, endpoint=True)
        frame = 0.05
        t = x * self.config.count * frame
        times = t0 + t * u.s
        alts = 0.53 + x * self.dz
        azs = 4.53 - x * self.da
        ints = 2e11 * (1 - x)**5 * x**11

        self.meteor = PointSource(alts, azs, ints, times)

        self.simulate(t0)

    def simulate(self, time: Time):
        for i in range(0, self.config.count):
            stime = time + i * 0.05 * u.s
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
            ints = 265536 * np.exp(-0.921034 * self.catalogue.vmag(self.location, masked=True))
            scene.add_points(altaz.alt.radian, altaz.az.radian, ints)
            alt, az, inten = self.meteor.at_time(stime)

            length = 0.07
            for n in np.linspace(0, 1, 50):
                scene.add_points(alt - length * self.dz * n, az + length * self.da * n, inten * np.exp(-30 * n**1.5))
            #print(scene.data)

            scene.add_intensifier_noise(rate=1000, radius=70)
            #print(scene.data)
            scene.render_as_poisson()
            #print(scene.data)
            scene.add_thermal_noise(rate=40)
            #print(scene.data)

            scene.render(f'output/{i:03}.png')


simulator = MeteorSimulatorCLI()