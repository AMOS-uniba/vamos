#!/usr/bin/env python
import argparse
import logging
import sys

import numpy as np

from multiprocessing import Pool

from astropy.coordinates import EarthLocation, CartesianDifferential
from astropy.time import Time
import astropy.units as u
from scalyca import Scalyca

from models.meteor import Meteor

log = logging.getLogger('root')
VERSION = '0.1.1'


class MeteorSimulatorCLI(Scalyca):
    _prog = 'Vamos meteor simulator'
    _version = VERSION

    def add_arguments(self):
        self.add_argument('-o', '--outfile', type=argparse.FileType('w'), default=sys.stdout)

    def initialize(self):
        self.t0 = Time(self.config.start)
        self.dt = 0.05 * u.s
        self.times = self.t0 + np.arange(0, self.config.count) * self.dt

    def main(self):
        m = Meteor(self.t0, 1 * u.kg,
                   EarthLocation.from_geodetic(lat=48.372763 * u.deg, lon=17.373933 * u.deg, height=10 * u.km),
                   CartesianDifferential(-8570 * u.m / u.s, 28250 * u.m / u.s, -2750 * u.m / u.s))

        m.simulate(self.config.count, self.dt)

        if self.args.debug:
            log.debug(f"Meteoroid data:")
            for time, pos, br in zip(m.time, m.position, m.brightness):
                log.debug(
                    f"{time}, {pos.to_geodetic()}, {br}"
                )

        if self.args.outfile:
            m.dump_yaml(self.args.outfile)


def simulate(
        position: EarthLocation,
        velocity: CartesianDifferential,
        mass: u.Quantity[u.kg],
        time: Time,
    ):
    pool = Pool(self.config.cores)



simulator = MeteorSimulatorCLI().run()