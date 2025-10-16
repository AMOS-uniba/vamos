#!/usr/bin/env python
import logging

import argparse
import yaml
import numpy as np
import dotmap
from scalyca import Scalyca

from multiprocessing import Pool

from astropy.coordinates import EarthLocation, CartesianDifferential
from astropy.time import Time

from models.meteor import Meteor
from models.observer import Observer

log = logging.getLogger('root')
VERSION = '0.1.1'


class MeteorObserverCLI(Scalyca):
    _prog = 'Vamos meteor observer'
    _version = VERSION

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meteors = None
        self.observers = None

    def add_arguments(self):
        self.add_argument('meteors', type=argparse.FileType('r'))
        self.add_argument('observers', type=argparse.FileType('r'))
        self.add_argument('--cores', type=int, default=4)

    def initialize(self):
        self.meteors = yaml.safe_load(self.args.meteors)
        self.observers = dotmap.DotMap(yaml.safe_load(self.args.observers)['available'], _dynamic=False)

    def main(self):
        for oname, obs in self.observers.items():
            observer = Observer(EarthLocation.from_geodetic(lat=obs.latitude, lon=obs.longitude, height=obs.altitude))

            for mname, met in self.meteors.items():
                meteor = Meteor()

                log.info(f"Now observing {meteor} by {observer}")
                observer.observe(meteor)

    def simulate(self, fragments, times):

        args = [(self.config.detector.xres, self.config.detector.yres,
                 self.projection, self.scaler,
                 self.location, self.catalogue, fragments, i, time) for i, time in enumerate(times)]
        pool = Pool(self.config.cores)
        print(f"Rendering {len(fragments)} fragments at {len(times)} times")
        pool.starmap(render, args, 1)

    def simulate(
            self,
            position: EarthLocation,
            velocity: CartesianDifferential,
            time: Time,
        ):
        pool = Pool(self.config.cores)



observer_cli = MeteorObserverCLI().run()