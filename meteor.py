import numpy as np
import pandas as pd

from pathlib import Path


class Meteor:
    """
    A class storing a single meteor
    """
    @classmethod
    def load(cls,
             filename: Path):
        self.frames = pd.DataFrame.read_csv(filename, sep='\t')


