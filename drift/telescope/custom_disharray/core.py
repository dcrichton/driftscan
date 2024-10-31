"""`TransitTelescope` mixins for dish array surveys with concrete implementations.
"""

import inspect
from typing import Tuple, Union

import numpy as np
import scipy.linalg as linalg
from caput import config
import abc

from drift.core.telescope import _remap_keyarray
from drift.core.telescope import PolarisedTelescope, UnpolarisedTelescope

from .layouts import AVAILABLE_LAYOUTS
from .beams import AVAILABLE_BEAMS, rotate_thetaphi_beam


def _confdict_from_classes(list_of_classes: list) -> dict:
    """Generate a dictionary of the :py:class:`caput.config.Property`'s of a list
    of classes. Used to reconstructs the single pointing telescope
    from the class hierarchy of a MultiElevationSurvey.

    Parameters
    ----------
    list_of_classes
        List of classes to find config.Property attributes from.

    Returns
    -------
        Dictionary of config.Property names and values.
    """
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
    Mixin for :py:class:`drift.core.telescope.TransitTelescope` that
    provides configurable primary beams and array layouts for dish array
    surveys.

    Attributes
    ----------
    min_u, min_v: :py:class:`caput.config.Property(proptype=float)`
        Minimum EW and NS element separation [in metres] used for l, m range calculations.
    latitude: :py:class:`caput.config.Property(proptype=float)`
        Telescope latitude in degrees.
    longitude: :py:class:`caput.config.Property(proptype=float)`
        Telescope longitude in degrees.
    altitude: :py:class:`caput.config.Property(proptype=float)`
        Telescope altitude in metres.
    layout_spec: :py:class:`caput.config.Property(proptype=dict)`
        Dictionary of configuration options for the array layout. See supported layouts
        and their parameters at :py:mod:`.layouts`.
        Default: 3x3 grid with 6m spacing
    beam_spec: :py:class:`caput.config.Property(proptype=dict)`
        Dictionary of configuration options for the primary beams. See supported beams
        and their parameters at :py:mod:`.beams`.
        Default: Gaussian beam with FWHM = wavelength/6m
    """

    min_u = config.Property(proptype=float, default=6.0)
    min_v = config.Property(proptype=float, default=6.0)

    latitude = config.Property(proptype=float, default=0)
    longitude = config.Property(proptype=float, default=0)
    altitude = config.Property(proptype=float, default=0)

    layout_spec = config.Property(
        proptype=dict,
        default={
            "type": "grid",
        },
    )

    beam_spec = config.Property(
        proptype=dict,
        default={
            "type": "gaussian",
        },
    )

    def __init__(self):
        """Initialise a telescope object. The observatory coordinates are derived
        from the class attributes/configuration system and not passed like parameters
        as in :py:class:`drift.core.telescope.TransitTelescope`
        """

        # I don't think this is needed but it could be if things get added to
        # the transit telescope __init__ method.
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
        """(nfeed, 2) array of the relative positions of the feeds extracted from the `layout_obj`.
        Units in metres, packed EW, NS.
        """
        return self.layout_obj.feedpositions

    @property
    def polarisation(self) -> Union[np.ndarray, None]:
        """(nfeed,) array of "X" (East-like) or "Y" (North-like)
        polarisation indices. If the mixin is applied to an
        unpolarised telescope, returns None.
        """
        return self.layout_obj.polarisation

    @property
    def beamclass(self) -> np.ndarray:
        """(nfeed, ) array of beamclass indices. Currently only supports different
        beamclasses for different polarisation indices, should be
        overridden if something more complicated is needed.
        """
        if self.layout_obj.polarisation is None:
            return np.zeros(self.layout_obj.feedpositions.shape[0])
        else:
            return (self.layout_obj.polarisation == "Y").astype(int)

    def beam(self, feed_ind: int, freq_ind: int) -> np.ndarray:
        """Primary beam pattern extracted from the `beam_obj`.

        Parameters
        ----------
        feed_ind
            Index of the feed to pass to the `beam_obj`.
        freq_ind
            Frequency index to pass to the `beam_obj`.

        Returns
        -------
            (npix, 2) Beam pattern in sky theta, phi directions. May be complex.
        """
        return self.beam_obj(self, feed_ind, freq_ind)

    @property
    def u_width(self) -> float:
        """Minimum EW element separation [in metres] used for m, l range calculations."""
        return self.min_u

    @property
    def v_width(self) -> float:
        """Minimum NS element separation [in metres] used for m, l range calculations."""
        return self.min_v


class MultiElevationSurvey(config.Reader, metaclass=abc.ABCMeta):
    """Mixin for :py:class:`drift.core.telescope.TransitTelescope` that
    enables multi-pointed surveys in elevation.

    This works by duplicating the telescope feeds (and hence baselines),
    for each pointing. However pairs across pointings are masked so we
    only linearly increase the number of baselines. Primary beams from the
    :py:class:`CustomDishArray` may support a pointing  argument. Otherwise,
    the polarised HEALPix beam pattern is directly rotated. A caveat to this
    approach is that the feed and baselines indices as well as related telescope
    state is now a mixture of physical indices and pointings. Some helper
    methods are provided to assist with decoupling this.

    .. note::
        For implementation reasons it is also strongly recommended that this
        mixin be the bottom of the class hierarchy.

    Attributes
    ----------
    elevation_start: :py:class:`caput.config.Property(proptype=float)`
        Start point of the elevation offset pointings, relative to zenith in degrees.
        Positive is north of zenith, negative is south of zenith.
        Default: -10 [degrees]
    elevation_stop: :py:class:`caput.config.Property(proptype=float)`
        End point of the elevation offset pointings, relative to zenith in degrees.
        Positive is north of zenith, negative is south of zenith.
        Default: 10 [degrees]
    npointings: :py:class:`caput.config.Property(proptype=int)`
        Number of elevation pointings in the multi-pointed survey.
        Default: 7
    """

    elevation_start = config.Property(proptype=float, default=-10)
    elevation_stop = config.Property(proptype=float, default=10)
    npointings = config.Property(proptype=int, default=7)

    def _finalise_config(self):
        super()._finalise_config()
        mro = inspect.getmro(self.__class__)
        single_pointing_class_ind = mro.index(MultiElevationSurvey) + 1
        self.single_pointing_class = mro[single_pointing_class_ind]
        single_pointing_config = _confdict_from_classes(mro[::-1])
        # And update with instance values
        single_pointing_config.update(
            {
                k: v
                for k, v in self.__dict__.items()
                if k not in ["elevation_start", "elevation_stop", "npointings"]
            }
        )
        self.single_pointing_telescope = self.single_pointing_class.from_config(
            single_pointing_config
        )

    # Short-circuit calculations
    @property
    def npairs(self):
        return self.single_pointing_telescope.npairs * self.npointings

    @property
    def lmax(self):
        return self.single_pointing_telescope.lmax

    @property
    def mmax(self):
        return self.single_pointing_telescope.mmax

    def _quick_uniquepairs(self):
        # Not fully general
        single_telescope_up = self.single_pointing_telescope.uniquepairs
        # Generalize for different numpols
        up = np.tile(
            single_telescope_up.reshape((single_telescope_up.shape[0] // 4, 4, -1)),
            reps=(1, self.npointings, 1),
        )
        offsets = np.repeat(
            self.single_pointing_telescope.nfeed * np.arange(self.npointings), 4
        )
        up += offsets[None, :, None]
        return up.reshape((-1, 2))
    
    @property
    def uniquepairs(self):
        return self._quick_uniquepairs()

    @property
    def elevation_pointings(self) -> np.ndarray:
        """Elevation pointings relative to zenith in degrees. Positive is
        North of zenith.
        """
        return np.linspace(self.elevation_start, self.elevation_stop, self.npointings)

    @property
    def pointing_feedmap(self) -> np.ndarray:
        """Mapping between "feed" index and pointing index where the feed index has a copy
        of each physical feed for each pointing.
        """
        return np.repeat(
            np.arange(len(self.elevation_pointings)),
            self.single_pointing_telescope.nfeed,
        )

    @property
    def pointing_baseline_map(self) -> np.ndarray:
        """Mapping between "baseline" index and pointing index where the feed baseline has a copy
        of each pair for each pointing.
        """
        return self.pointing_feedmap[self.uniquepairs[:, 0]]

    def calculate_feedpairs(self):
        self.single_pointing_telescope.calculate_feedpairs()
        super().calculate_feedpairs()

    def _init_trans(self, nside: int):
        super()._init_trans(nside)
        self.single_pointing_telescope._init_trans(nside)

    @property
    def feedpositions(self) -> np.ndarray:
        """(nfeed_actual x npointings, 2) array of the relative positions of the
        feeds repeated for each pointings. Units in metres, packed EW, NS.
        """
        return np.tile(
            self.single_pointing_telescope.feedpositions,
            (len(self.elevation_pointings), 1),
        )

    @property
    def nfeed_actual(self) -> int:
        """The number of physical feeds, nfeed has become nfeed x npointings
        to implement this Mixin.
        """
        return self.nfeed // self.npointings

    def _unique_baselines(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ensures baselines from different pointings are treated as unique"""
        fmap, mask = self.single_pointing_telescope._unique_baselines()
        nfeed = self.single_pointing_telescope.nfeed

        block_fmap = linalg.block_diag(
            *[fmap + i * nfeed for i, _ in enumerate(self.elevation_pointings)]
        )
        block_mask = linalg.block_diag(*[mask for _ in self.elevation_pointings])

        return _remap_keyarray(block_fmap, block_mask), block_mask

    def _unique_beams(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ensures beams from different pointings are treated as unique"""
        nfeed = self.single_pointing_telescope.nfeed
        bmap, mask = self.single_pointing_telescope._unique_beams()

        block_bmap = linalg.block_diag(
            *[bmap + i * nfeed for i, _ in enumerate(self.elevation_pointings)]
        )
        block_mask = linalg.block_diag(*[mask for _ in self.elevation_pointings])

        return block_bmap, block_mask

    @property
    def beamclass(self) -> np.ndarray:
        """(nfeed_actual x npointings,) array of beamclass indices repeated and
        made unique for each pointing.
        """
        orig_bc = self.single_pointing_telescope.beamclass
        bc = (
            orig_bc + (orig_bc.max() + 1) * np.arange(self.npointings)[:, None]
        ).ravel()
        return bc

    @property
    def polarisation(self) -> Union[np.ndarray, None]:
        """(nfeed_actual x npointings,) array of "X" (East-like) or "Y" (North-like)
        polarisation indices. If the mixin is applied to an
        unpolarised telescope, returns None.
        """
        if self.single_pointing_telescope.polarisation is None:
            return None
        else:
            orig_pol = self.single_pointing_telescope.polarisation
        return np.tile(orig_pol, len(self.elevation_pointings))

    def beam(self, feed_ind: int, freq_ind: int) -> np.ndarray:
        """Primary beam pattern. If a beam_obj from a `:py:class:.CustomDishArray`
        that supports a pointing argument is detected, the offset pointing is
        passed through. Otherwise the evaluated HEALPix beam pattern of the
        `single_pointing_telescope` is directly rotated.

        Parameters
        ----------
        feed_ind
            Index of the feed to pass to the `beam_obj`.
        freq_ind
            Frequency index to pass to the `beam_obj`.

        Returns
        -------
            (npix, 2) Beam pattern in sky theta, phi directions. May be complex.
        """
        elev_pointing = np.radians(
            self.elevation_pointings[self.pointing_feedmap[feed_ind]]
        )
        if hasattr(self, "beam_obj") and getattr(
            self.beam_obj, "supports_pointing", False
        ):
            # We're working on a CustomDishArray with a beam object that supports a pointing
            # argument.
            return self.beam_obj(
                self, feed_ind, freq_ind, elevation_pointing=elev_pointing
            )
        else:
            # We manually rotate the beam which is assumed to be at the zenith
            # pointing in sky Eth, Eph.
            beam = super().beam(feed_ind, freq_ind)
            return rotate_thetaphi_beam(beam, elev_pointing, self._angpos)


class PolarisedDishArray(CustomDishArray, PolarisedTelescope):
    """A polarised, configurable dish array."""

    pass


class PolarisedDishArraySurvey(MultiElevationSurvey, PolarisedDishArray):
    """A polarised, configurable dish array survey with multiple elevation pointings."""

    pass


class UnpolarisedDishArray(CustomDishArray, UnpolarisedTelescope):
    """An unpolarised, configurable dish array."""

    pass


class UnpolarisedDishArraySurvey(MultiElevationSurvey, UnpolarisedDishArray):
    """An unpolarised, configurable dish array survey with multiple elevation pointings."""

    pass
