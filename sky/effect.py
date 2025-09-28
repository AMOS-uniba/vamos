from abc import abstractmethod

from typing import Union
from numpy.typing import NDArray

from astropy.coordinates import EarthLocation
from astropy.time import Time


class SkyEffect:
    @abstractmethod
    def __call__(self,
                 alt: Union[float, NDArray],
                 az: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Return the intensity at (`alt`, `az`)
        """

    def __init__(self,
                 location: EarthLocation,
                 time: Optional[Time] = None):
        self.location = location
        self.time = time if time is not None else Time.now()


class Sunlight(SkyEffect):
    def __call__(self,
                 alt: Union[float, NDArray],
                 az: Union[float, NDArray]) -> Union[float, NDArray]:
        sun = get_sun(location, time)

        return np.where(
            alt <= 0,
            0,
            25 * np.cos(alt) * np.sin(az / 2)
        )


class Moonlight(SkyEffect):
    def __call__(self,
                 alt: Union[float, NDArray],
                 az: Union[float, NDArray]) -> Union[float, NDArray]:
        moon = get_moon(location, time)
        brightness = get_moon_brightness(location, time)

        return np.where(
            alt <= 0,
            brightness,
            25 * np.cos(moon.alt) * np.sin(az / 2)
        )
