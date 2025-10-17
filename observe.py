#!/usr/bin/env python
import logging

import argparse
import sys

import yaml
import numpy as np
import dotmap
from scalyca import Scalyca

from multiprocessing import Pool

import astropy.units as u
from astropy.coordinates import EarthLocation, CartesianDifferential, AltAz, SkyCoord, CartesianRepresentation
from astropy.time import Time

from models.meteor import Meteor
from models.observer import Observer

log = logging.getLogger('root')
VERSION = '0.1.1'




def alt_az_from_earthlocations(obs: EarthLocation, target: EarthLocation, obstime=None):
    """
    Compute altitude, azimuth, and distance from one EarthLocation to another.

    Parameters
    ----------
    obs : EarthLocation
        Observer location.
    target : EarthLocation
        Target location.
    obstime : Time, optional
        Observation time. Required for proper AltAz transformation. Defaults to current time.

    Returns
    -------
    alt : Quantity
        Altitude in degrees.
    az : Quantity
        Azimuth in degrees.
    distance : Quantity
        Distance in meters.
    """
    if obstime is None:
        obstime = Time.now()

    # AltAz frame at observer
    altaz_frame = AltAz(location=obs, obstime=obstime)

    # Target as ITRS SkyCoord
    target_coord = SkyCoord(target.get_itrs(obstime=obstime))

    # Transform to observer's AltAz frame
    rpos = target_coord.transform_to(altaz_frame)

    return rpos.alt, rpos.az, rpos.distance


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
        self.add_argument('-o', '--outfile', type=argparse.FileType('w'), default=sys.stdout)

    def initialize(self):
        self.observers = dotmap.DotMap(yaml.safe_load(self.args.observers)['enabled'], _dynamic=False)

    def main(self):
        meteor = Meteor.load_yaml(self.args.meteors)

        for oname, obs in self.observers.items():
            observer = Observer(EarthLocation.from_geodetic(lat=obs.latitude, lon=obs.longitude, height=obs.altitude),
                                name=obs.name)
            log.info(f"Now observing {meteor} by {observer}")

            points = observer.observe(meteor)

            points.dump_yaml(self.args.outfile)

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
