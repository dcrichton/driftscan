"""Implementations of simple models of the HIRAX telescope.
"""

from pkg_resources import resource_filename
import abc
from caput import config

from drift.core.telescope import PolarisedTelescope
from .core import MultiElevationSurvey, CustomDishArray

# These are approximate for the Swartfontein site.
HIRAX_LATITUDE = -30.69638889  # 30°41'47.0"S
HIRAX_LONGITUDE = 21.57222222  # 21°34'20.0"E
HIRAX_ALTITUDE = 1113.0  # m


class _HIRAXDefaults(CustomDishArray, config.Reader, metaclass=abc.ABCMeta):

    freq_start = config.Property(proptype=float, default=400)
    freq_end = config.Property(proptype=float, default=800)
    num_freq = config.Property(proptype=int, default=1024)

    latitude = config.Property(proptype=float, default=HIRAX_LATITUDE)
    longitude = config.Property(proptype=float, default=HIRAX_LONGITUDE)
    altitude = config.Property(proptype=float, default=HIRAX_ALTITUDE)

    layout_spec = config.Property(
        proptype=dict,
        default={
            "type": "grid",
            "grid_ew": 32,
            "grid_ns": 32,
            "spacing_ew": 6.5,
            "spacing_ns": 8.5,
        },
    )


class HIRAX(_HIRAXDefaults, PolarisedTelescope):
    pass


class HIRAXSurvey(MultiElevationSurvey, HIRAX):
    pass


class _HIRAXHexTile(_HIRAXDefaults, config.Reader, metaclass=abc.ABCMeta):

    layout_spec = config.Property(
        proptype=dict,
        default={
            "type": "file",
            "filenames": [
                resource_filename(
                    "drift.telescope.custom_disharray",
                    "data/hirax_hextile_template_1024.dat",
                )
            ],
            "spacing_ew": 6.5 / 2,
            "spacing_ns": 8.5 / 2,
        },
    )


class HIRAXHexTile(_HIRAXHexTile, PolarisedTelescope):
    pass


class HIRAXHexTileSurvey(MultiElevationSurvey, HIRAX):
    pass
