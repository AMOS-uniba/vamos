import numpy as np
import random


class DetectorEffect(Effect):
    def __call__(self, data):
        return self.func(data)


class GaussianNoise(DetectorEffect):
    def func(self, data):
        return data + np.abs(random.gauss(0, 0.01, shape=data.shape))
