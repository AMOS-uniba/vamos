import numpy as np
import pandas as pd

from pathlib import Path


class Meteor:
    def __init__(self,
                 filename: Path):
        self.frames = pd.DataFrame.read_csv(filename, sep='\t')

    
