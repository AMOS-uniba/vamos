from abc import abstractmethod

from typing import Union
from numpy.typing import NDArray

from astropy.coordinates import EarthLocation
from astropy.time import Time
import astropy.units as u


class SkyEffect:
    """
    Effects defined in terms of angles in the sky.
    In general can be purely additive (sources, such as stars, scattered light, airglow)
    or subtractive / dividing (clouds, obstructions).
    """
    def __init__(self,
                 location: EarthLocation,
                 time: Optional[Time] = None):
        self.location = location
        self.time = time if time is not None else Time.now()

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
        sun = get_sun(self.location, self.time)
        brightness = -26.74 * u.mad

        return np.where(
            alt <= 0,
            0,
            25 * np.cos(alt) * np.sin(az / 2)
        )


class Moonlight(SkySource):
    def func(self,
                 alt: Union[float, NDArray],
                 az: Union[float, NDArray]) -> Union[float, NDArray]:
        moon = get_moon(self.location, self.time)
        brightness = get_moon_brightness(self.location, self.time)

        return np.where(
            alt <= 0,
            brightness,
            25 * np.cos(moon.alt) * np.sin(az / 2)
        )
