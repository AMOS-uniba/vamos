import numpy as np

from abc import abstractmethod

from typing import Union, Optional

from amosutils.metrics import spherical
from numpy.typing import NDArray

from astropy.coordinates import EarthLocation, get_body, AltAz
from astropy.time import Time
import astropy.units as u


class SkyEffect:
    """
    Effects defined in terms of angles in the sky.
    In general can be purely additive (sources, such as stars, scattered light, airglow)
    or subtractive / dividing (clouds, obstructions, ...).
    """
    def __init__(self,
                 location: EarthLocation,
                 time: Optional[Time] = None):
        self.location = location
        self.time = time if time is not None else Time.now()
        self.altaz = AltAz(obstime=self.time, location=self.location)

    @abstractmethod
    def __call__(self,
                 data: Union[float, NDArray],
                 alt: Union[float, NDArray],
                 az: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Transform the value at data by the corresponding value at (`alt`, `az`)
        """


class SkySource(SkyEffect):
    """
    SkySource is a source -- its function is purely additive.
    """
    def __call__(self,
                 data: Union[float, NDArray],
                 alt: Union[float, NDArray],
                 az: Union[float, NDArray]) -> Union[float, NDArray]:
        return data + self.func(alt, az)

    @abstractmethod
    def func(self,
             alt: Union[float, NDArray],
             az: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        The inner function of the source, defined at (`alt`, `az`)
        """


class Sunlight(SkySource):
    def func(self,
             alt: Union[float, NDArray],
             az: Union[float, NDArray]) -> Union[float, NDArray]:
        sun = get_body('sun', self.time, self.location)
        sun = sun.transform_to(self.altaz)
        brightness = -26.74 * u.mag

        pixels = np.stack((alt, az), axis=2)
        dist = spherical(pixels, np.stack((sun.alt.radian, sun.az.radian), axis=0).T)

        intensity = 10 * (0.8 * np.sin(sun.alt) + 0.2) if sun.alt >= 0 else 0.1 * np.exp(sun.alt.radian * 10)

        return np.where(
            alt <= 0,
            0,
            25 * np.exp(-alt * 5) * np.exp(-dist**2) * intensity
        )


class Airglow(SkySource):
    def __init__(self, location: EarthLocation, time: Optional[Time] = None, **kwargs):
        super().__init__(location, time)
        self.intensity = kwargs.pop('intensity', 0)

    def func(self,
             alt: Union[float, NDArray],
             az: Union[float, NDArray]) -> Union[float, NDArray]:
        return np.where(
            alt <= 0, 0,
            self.intensity * np.exp(-alt * 5)
        )


class Moonlight(SkySource):
    def func(self,
                 alt: Union[float, NDArray],
                 az: Union[float, NDArray]) -> Union[float, NDArray]:
        moon = get_body('moon', self.time, self.location)
        brightness = get_moon_brightness(self.location, self.time)

        return np.where(
            alt <= 0,
            brightness,
            25 * np.cos(moon.alt) * np.sin(az / 2)
        )