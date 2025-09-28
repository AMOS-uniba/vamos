import numpy as np

from amosutils.projections import Projection


class SkyDotCollection:
    """
    A collection of point light sources in the sky, in the alt-az system.
    """

    def __init__(self,
                 alt: np.ndarray,
                 az: np.ndarray,
                 intensity: np.ndarray):
        assert alt.shape == az.shape == intensity.shape, \
            f"Shapes do not match: {alt.shape=}, {az.shape=}, {intensity.shape=}"

        self._alt = alt
        self._az = az
        self._intensity = intensity

    @property
    def alt(self):
        return self._alt

    @property
    def az(self):
        return self._az

    def project(self,
                projection: Projection) -> np.ndarray[np.float64, np.float64]:
        """
        Project this collection onto a canvas
        """

        x, y = projection.invert(self.alt, self.az)

