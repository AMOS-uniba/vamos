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
        self.catalogue = Catalogue('/home/kvik/amos/vasco/catalogues/HYG30.tsv')
        self.projection = Projection.from_dict(self.config['projection'])
        self.scaler = ScalingShifter(x0=self.config.pixels.x0, y0=self.config.pixels.y0,
                                     xs=self.config.pixels.xs, ys=self.config.pixels.ys)

        self.simulate(Time(self.config.start))

    def simulate(self, time: Time):
        for i in range(0, 24):
            stime = time + i * 60 * u.s
            scene = Scene(1600, 1200,
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

            print(f"Rendering {i:03}.png")
            scene.render(f'{i:03}.png')

simulator = MeteorSimulatorCLI()