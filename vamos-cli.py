#!/usr/bin/env python
import argparse
import os
import time

import yaml

import numpy as np
import dotmap

from multiprocessing import Pool

from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u

from pointsource import PointSource
from amosutils.projections import Projection
from amosutils.catalogue import Catalogue
from amosutils.projections.shifters import ScalingShifter

from models.scene import Scene


class MeteorSimulatorCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--meteor-file', '-m')
        self.parser.add_argument('config', type=argparse.FileType('r'))
        self.args = self.parser.parse_args()

        self.config = dotmap.DotMap(yaml.safe_load(self.args.config), _dynamic=False)

        self.location = EarthLocation(self.config.location.longitude,
                                      self.config.location.latitude,
                                      self.config.location.altitude)
        self.catalogue = Catalogue(self.config.catalogue)
        self.projection = Projection.from_dict(self.config['projection'])
        self.scaler = ScalingShifter(x0=self.config.pixels.x0, y0=self.config.pixels.y0,
                                     xs=self.config.pixels.xs, ys=self.config.pixels.ys)


        dz = 0.06 * u.rad / u.s
        da = -0.71 * u.rad / u.s
        length = 4 # Length of visible trail

        t0 = Time(self.config.start)
        dt = 0.05 * u.s

        # Dimensionless time scaled to 1
        tau = np.linspace(0, 1, self.config.count + 1, endpoint=True)

        t = tau * self.config.count * dt
        times = t0 + t
        alts = 14 * u.deg + t * dz
        azs = 278 * u.deg + t * da
        ints = 2e10 * (1 - tau)**5 * tau**11

        fragments = []
        for fn in np.linspace(0, 1, 51, endpoint=True):
            fragments.append(PointSource(alts - fn * dz * length * dt, azs - fn * da * length * dt, ints * np.exp(-20 * fn * 1.5), times))

        fragments[0].intensity[21] *= 30
        fragments[0].intensity[20] *= 10

        self.simulate(fragments, times)

    def simulate(self, fragments, times):
        args = [(self.config.detector.xres, self.config.detector.yres,
                 self.projection, self.scaler,
                 self.location, self.catalogue, fragments, i, time) for i, time in enumerate(times)]
        pool = Pool(self.config.cores)
        print(f"Rendering {len(fragments)} fragments at {len(times)} times")
        pool.starmap(render, args, 1)

def render(xres: int, yres: int,
           projection: Projection,
           scaler: ScalingShifter,
           location: EarthLocation,
           catalogue: Catalogue,
           fragments: list[PointSource],
           i, timestamp: Time) -> int:

    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    scene = Scene(xres, yres, projection=projection, scaler=scaler, location=location, catalogue=catalogue, time=timestamp)

    scene.build(fragments)
    # print(scene.data)

    scene.add_intensifier_noise(100, 10, radius=70)
    scene.render_as_poisson()
    scene.add_thermal_noise(rate=40)

    scene.render(f'output/{i:03}.png')
    return 1


simulator = MeteorSimulatorCLI()