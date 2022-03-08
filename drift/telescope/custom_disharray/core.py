"""`TransitTelescope` Mixins and implementations.
"""

import inspect
from typing import Tuple

import numpy as np
import scipy.linalg as linalg
from caput import config
import abc

from drift.core.telescope import _remap_keyarray
from drift.core.telescope import PolarisedTelescope, UnpolarisedTelescope

from .layouts import AVAILABLE_LAYOUTS
from .beams import AVAILABLE_BEAMS, rotate_thetaphi_beam


def _confdict_from_classes(list_of_classes):
    # Later overrides earlier
    confdict = {}
    for cl in list_of_classes:
        for cl_attr in cl.__dict__.values():
            if isinstance(cl_attr, config.Property):
                val = cl_attr.__get__(cl_attr, None)
                # This will miss cases where None is a non-default value
                if val is not None:
                    confdict[cl_attr.propname] = val
    return confdict


class CustomDishArray(config.Reader, metaclass=abc.ABCMeta):
    """
    Mixin for providing configurable array beams and array layouts, for dish array
    surveys.

    Attributes:

    min_u, min_v: scalar
        Minimum EW and NS element separation [in metres] used for m, l range calculations.
    latitude: scaler
        Telescope latitude in degrees.
    longitude: scaler
        Telescope longitude in degrees.
    altitude: scaler
        Telescope altitude in metres.
    layout_spec: dict
        Dictionary of configuration options for the array layout. See supported layouts
        and their parameters at :py:mod:`.layouts`.
        Default: 3x3 grid with 6m spacing
    beam_spec: dict
        Dictionary of configuration options for the primary beams. See supported beams
        and their parameters at :py:mod:`.beams`.
        Default: Gaussian beam with FWHM = wavelength/6m
    """

    min_u = config.Property(proptype=float, default=6.0)
    min_v = config.Property(proptype=float, default=6.0)

    latitude = config.Property(proptype=float, default=0)
    longitude = config.Property(proptype=float, default=0)
    altitude = config.Property(proptype=float, default=0)

    layout_spec = config.Property(proptype=dict, default={"type": "grid",})

    beam_spec = config.Property(proptype=dict, default={"type": "gaussian",},)

    def __init__(self):
        super().__init__(latitude=self.latitude, longitude=self.longitude)

    def _finalise_config(self):
        self.layout_obj = AVAILABLE_LAYOUTS[self.layout_spec["type"]].from_config(
            self.layout_spec, expand_pol=self.num_pol_sky > 1
        )
        self.beam_obj = AVAILABLE_BEAMS[self.beam_spec["type"]].from_config(
            self.beam_spec
        )

    @property
    def feedpositions(self) -> np.ndarray:
        return self.layout_obj.feedpositions

    @property
    def polarisation(self) -> np.ndarray:
        return self.layout_obj.polarisation

    @property
    def beamclass(self) -> np.ndarray:
        if self.layout_obj.polarisation is None:
            return np.zeros(self.layout_obj.feedpositions.shape[0])
        else:
            return (self.layout_obj.polarisation == 'Y').astype(int)

    def beam(self, feed_ind: int, freq_ind: int) -> np.ndarray:
        return self.beam_obj(self, feed_ind, freq_ind)

    @property
    def u_width(self) -> float:
        return self.min_u

    @property
    def v_width(self) -> float:
        return self.min_v


class MultiElevationSurvey(config.Reader, metaclass=abc.ABCMeta):
    """ 
    Mixin Class for Multi-elevation survey
    Insert caveats here. 
    """

    # 7 pointings up to 10 degrees off zenith in either direction
    elevation_start = config.Property(proptype=float, default=-10)
    elevation_stop = config.Property(proptype=float, default=10)
    npointings = config.Property(proptype=int, default=7)

    # ndays is now ndays per pointing
    ndays = config.Property(proptype=int, default=120)

    def _finalise_config(self):
        super()._finalise_config()
        mro = inspect.getmro(self.__class__)
        single_pointing_class_ind = mro.index(MultiElevationSurvey) + 1
        self.single_pointing_class = mro[single_pointing_class_ind]
        # Fetch all config properties up to single_pointing_class and initialize
        # them in single_pointing_telescope
        single_pointing_config = _confdict_from_classes(mro[::-1])
        # And update with instance values
        single_pointing_config.update(
            {k: v for k, v in self.__dict__.items() if k in single_pointing_config}
        )
        self.single_pointing_telescope = self.single_pointing_class.from_config(
            single_pointing_config
        )

    @property
    def elevation_pointings(self) -> np.ndarray:
        return np.linspace(self.elevation_start, self.elevation_stop, self.npointings)

    @property
    def pointing_feedmap(self) -> np.ndarray:
        return np.repeat(
            np.arange(len(self.elevation_pointings)),
            self.single_pointing_telescope.nfeed,
        )

    @property
    def pointing_baseline_map(self) -> np.ndarray:
        return self.pointing_feedmap[self.uniquepairs[:, 0]]

    def calculate_feedpairs(self):
        self.single_pointing_telescope.calculate_feedpairs()
        super().calculate_feedpairs()

    def _init_trans(self, nside: int):
        super()._init_trans(nside)
        self.single_pointing_telescope._init_trans(nside)

    @property
    def feedpositions(self) -> np.ndarray:
        return np.tile(
            self.single_pointing_telescope.feedpositions,
            (len(self.elevation_pointings), 1),
        )

    @property
    def nfeed_actual(self) -> int:
        """ 
        The number of physical feeds, nfeed has become nfeed * npointings 
        for implentation purposes.
        """
        return self.nfeed // self.npointings

    def _unique_baselines(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensures baselines from different pointings are treated as unique
        """
        fmap, mask = self.single_pointing_telescope._unique_baselines()
        nfeed = self.single_pointing_telescope.nfeed

        block_fmap = linalg.block_diag(
            *[fmap + i * nfeed for i, _ in enumerate(self.elevation_pointings)]
        )
        block_mask = linalg.block_diag(*[mask for _ in self.elevation_pointings])

        return _remap_keyarray(block_fmap, block_mask), block_mask

    def _unique_beams(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensures beams from different pointings are treated as unique
        """
        nfeed = self.single_pointing_telescope.nfeed
        bmap, mask = self.single_pointing_telescope._unique_beams()

        block_bmap = linalg.block_diag(
            *[bmap + i * nfeed for i, _ in enumerate(self.elevation_pointings)]
        )
        block_mask = linalg.block_diag(*[mask for _ in self.elevation_pointings])

        return block_bmap, block_mask

    @property
    def beamclass(self) -> np.ndarray:
        orig_bc = self.single_pointing_telescope.beamclass
        bc = (
            orig_bc + (orig_bc.max() + 1) * np.arange(self.npointings)[:, None]
        ).ravel()
        return bc

    @property
    def polarisation(self) -> np.ndarray:
        orig_pol = self.single_pointing_telescope.polarisation
        return np.tile(orig_pol, len(self.elevation_pointings))

    def beam(self, feed_ind: int, freq_ind: int) -> np.ndarray:
        ddec = self.elevation_pointings[self.pointing_feedmap[feed_ind]]  # In degrees
        if hasattr(self, "beam_obj") and getattr(
            self.beam_obj, "supports_pointing", False
        ):
            # We're working on a CustomDishArray with a beam object that supports a pointing
            # argument.
            altaz_pointing = np.radians(np.array([90 + ddec, 180]))
            return self.beam_obj(
                self, feed_ind, freq_ind, altaz_pointing=altaz_pointing
            )
        else:
            # We manually rotate the beam which is assumed to be at the zenith
            # pointing in sky Eth, Eph.
            beam = super().beam(feed_ind, freq_ind)
            return rotate_thetaphi_beam(beam, np.radians(-ddec), self._angpos)


class PolarisedDishArray(CustomDishArray, PolarisedTelescope):
    pass


class PolarisedDishArraySurvey(MultiElevationSurvey, PolarisedDishArray):
    pass


class UnpolarisedDishArray(CustomDishArray, UnpolarisedTelescope):
    pass


class UnpolarisedDishArraySurvey(MultiElevationSurvey, UnpolarisedDishArray):
    pass
