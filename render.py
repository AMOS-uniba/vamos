#!/usr/bin/env python
import argparse
import os
import time
from typing import Optional

import dotmap
import numpy as np

from multiprocessing import Pool

import yaml
from astropy.coordinates import EarthLocation
from astropy.time import Time

from scalyca import Scalyca

from amosutils.projections import Projection
from amosutils.catalogue import Catalogue
from amosutils.projections.shifters import ScalingShifter

from models.skypointsource import SkyPointSource
from models.scene import Scene


np.set_printoptions(edgeitems=100)


class MeteorRenderer(Scalyca):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.location: Optional[EarthLocation] = None
        self.catalogue: Optional[Catalogue] = None
        self.projection: Optional[Projection] = None
        self.scaler = None

    def add_arguments(self):
        self.add_argument('source', type=argparse.FileType('r'))
        self.add_argument('camera', type=argparse.FileType('r'))

    def initialize(self):
        self.location = EarthLocation(self.config.location.longitude,
                                      self.config.location.latitude,
                                      self.config.location.altitude)
        self.catalogue = Catalogue(self.config.catalogue)
        self.projection = Projection.from_dict(self.config['projection'])

        self.camera = dotmap.DotMap(yaml.safe_load(self.args.camera), _dynamic=False)

        self.scaler = ScalingShifter(x0=self.camera.pixels.x0, y0=self.camera.pixels.y0,
                                     xs=self.camera.pixels.xs / 1000, ys=self.camera.pixels.ys / 1000)

    def main(self):
        fragments = [SkyPointSource.load_yaml(self.args.source)]

        times = fragments[0].time

        args = [(self.camera.detector.xres, self.camera.detector.yres,
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
           fragments: list[SkyPointSource],
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


simulator = MeteorRenderer().run()