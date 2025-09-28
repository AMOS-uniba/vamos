import astropy


class Frame:
    """
    A collection of meteor light sources and reference stars at one point in time.
    Starts with a local alt-az frame (z, a, I)
    """

    def __init__(self,
                 time: astropy.time.Time,
                 particles: [Meteoroid]):
        self._time = time

    def render(self,
               res_x: int,
               res_y: int,
               *,
               supersample: int = 1) -> np.ndarray[np.float64, np.float64]:
        """
        Render the entire frame to a numpy array (res_y, res_x), optionally using supersampling.
        """
        canvas = np.zeros((res_y * supersampling, res_x * supersampling), dtype=np.float64)

        for dot in self.dots:
            canvas[dot.y, dot.x] = dot.intensity

        cat = Catalogue('data/HYG30.tsv')


        canvas = np.reshape(canvas, (supersampling, supersampling, res_y, res_x))
        canvas = np.mean(canvas, axis=(0, 1))

        return canvas
