#!/usr/bin/env python
import argparse
import os
import time

import yaml

import numpy as np
import dotmap

from multiprocessing import Pool

from astropy.coordinates import EarthLocation, CartesianDifferential
from astropy.time import Time
import astropy.units as u

from models.meteor import Meteor
from models.observer import Observer
from pointsource import PointSource
from amosutils.projections import Projection
from amosutils.catalogue import Catalogue
from amosutils.projections.shifters import ScalingShifter

from models.scene import Scene


np.set_printoptions(edgeitems=100)


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

        t0 = Time(self.config.start)
        dt = 0.05 * u.s
        times = t0 + np.arange(0, self.config.count) * dt

        m = Meteor(t0, 1 * u.kg,
                   EarthLocation.from_geodetic(lat=49 * u.deg, lon=18 * u.deg, height=101 * u.km),
                   CartesianDifferential(15800 * u.m / u.s, -43000 * u.m / u.s, -16750 * u.m / u.s))

        m.simulate(self.config.count, dt)

        o = Observer(self.location)
        ps = o.observe(m)
        self.simulate([ps], times)

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

    # This is needed so that noise is not generated using the same seed across workers
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    scene = Scene(xres, yres, projection=projection, scaler=scaler, location=location, catalogue=catalogue, time=timestamp)
    scene.build(fragments)

    scene.render_as_poisson()
    scene.add_intensifier_noise(100, 10, radius=70)
    scene.add_thermal_noise(rate=40)

    scene.render(f'output/{i:03}.png')
    return 1


simulator = MeteorSimulatorCLI()