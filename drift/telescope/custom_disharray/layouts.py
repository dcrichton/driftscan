"""Helpers for parameterised construction of array layouts.
"""

from typing import Optional

import numpy as np
from caput import config


class GridLayout(config.Reader):

    grid_ew = config.Property(proptype=int, default=3)
    grid_ns = config.Property(proptype=int, default=3)
    spacing_ew = config.Property(proptype=float, default=6.0)
    spacing_ns = config.Property(proptype=float, default=6.0)

    def __init__(self, expand_pol: Optional[bool] = False):
        self.expand_pol = expand_pol

    def _finalise_config(self):
        n_dish = self.grid_ew * self.grid_ns
        d_ew = -self.spacing_ew * (np.arange(n_dish) % self.grid_ew)
        d_ns = self.spacing_ns * (np.arange(n_dish) // self.grid_ns)

        self.feedpositions = np.column_stack([d_ew, d_ns])
        self.polarisation = None

        if self.expand_pol:
            self.polarisation = np.tile(["X", "Y"], len(self.feedpositions))
            self.feedpositions = np.repeat(self.feedpositions, 2, axis=0)


class SimpleLayoutFile(config.Reader):

    filenames = config.list_type(type_=str, default=[])
    spacing_ew = config.Property(proptype=float, default=1.0)
    spacing_ns = config.Property(proptype=float, default=1.0)

    def __init__(self, expand_pol: Optional[bool] = False):
        self.expand_pol = expand_pol

    def _finalise_config(self):
        d_ew, d_ns = [], []

        for filename in self.filenames:
            curr_d_ew, curr_d_ns = np.loadtxt(filename, unpack=True)
            d_ew.extend(curr_d_ew)
            d_ns.extend(curr_d_ns)
        d_ew = np.array(d_ew)
        d_ns = np.array(d_ns)

        self.feedpositions = np.column_stack(
            [self.spacing_ew * d_ew, self.spacing_ns * d_ns]
        )
        self.polarisation = None

        if self.expand_pol:
            self.polarisation = np.tile(["X", "Y"], len(self.feedpositions))
            self.feedpositions = np.repeat(self.feedpositions, 2, axis=0)


AVAILABLE_LAYOUTS = {
    "file": SimpleLayoutFile,
    "grid": GridLayout,
}
