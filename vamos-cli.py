#!/usr/bin/env python
import argparse
import yaml

import numpy as np
import matplotlib as mpl
import dotmap

from matplotlib import pyplot as plt

from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u

from dotcollection import SkyDotCollection
from amosutils.projections import Projection
from amosutils.catalogue import Catalogue


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

    def simulate(self, time: Time):
        dots = SkyDotCollection(
            np.random.uniform(0, np.pi, size=20),
            np.random.uniform(0, np.pi * 2, size=20),
            np.random.uniform(0, 100, size=20),
        )

        for i in range(0, 600):
            altaz = self.catalogue.altaz(self.location, time=time + i * 60 * u.s, masked=False)
            mask = altaz.alt > 0
            self.catalogue.mask = mask
            altaz = altaz[mask]

            starx, stary = self.projection.invert(np.pi / 2 - altaz.alt.radian, np.pi / 2 - altaz.az.radian)
            projected = self.projection.invert(dots.alt, dots.az)

            plt.style.use('dark_background')
            plt.figure(figsize=(14, 10))
            plt.ylim(-1, 1)
            plt.xlim(-1.5, 1.5)
            #plt.scatter(starx, -stary, s=1, c=np.exp(-self.catalogue.vmag(self.location, masked=True)), cmap='bone_r')
            plt.scatter(starx, -stary,
                        s=3,
                        c=self.catalogue.vmag(self.location, masked=True),
                        cmap='bone_r',
                        norm=mpl.colors.Normalize(vmin=0, vmax=5))
            plt.savefig(f'{i:03d}.png')
            plt.close('all')
            print(f"Plotted {i:03d}")


simulator = MeteorSimulatorCLI()
simulator.simulate(Time.now())


## Show wedding ring
# Ground truth: only in case of droppers


## Investigate how noise & compression artifacts impact the precision (x264)
