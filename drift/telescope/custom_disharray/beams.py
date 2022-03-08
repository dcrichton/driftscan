"""Helpers for parameterised primary beams for pointed dish arrays.
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from typing import Literal, Optional, Tuple, Union
import abc

from scipy.special import j1 as bessel_j1
import numpy as np
import healpy as hp

from caput import config
from cora.util import coord

from drift.core.telescope import TransitTelescope
from drift.telescope.cylbeam import polpattern

# Types
POLTYPE = Literal["X", "Y"]
FloatArrayLike = Union[np.ndarray, float]

# Constants
FWHM2SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def rot_mats_thetaphi(
    theta: np.ndarray, phi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    rot_mat_theta = np.array(
        [
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)],
        ]
    )

    rot_mat_phi = np.array(
        [[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )

    return rot_mat_theta, rot_mat_phi


def pointing_rot_mat(zenith: np.ndarray, altaz_pointing: np.ndarray) -> np.ndarray:
    rth, rphi = rot_mats_thetaphi(zenith[0], zenith[1])
    rzenith = rth @ rphi
    hp_alt = np.pi / 2 - altaz_pointing[0]
    hp_az = np.pi - altaz_pointing[1]
    ralt, raz = rot_mats_thetaphi(hp_alt, hp_az)
    rpoint = ralt @ raz

    return rpoint @ rzenith


def rotate_thetaphi_beam(
    beam: np.ndarray, rot_theta: np.ndarray, angpos: np.ndarray
) -> np.ndarray:

    # Rotate amplitude of Eth and Eph beam terms
    hp_rot = hp.Rotator(rot=(0, rot_theta, 0), deg=False)
    thph_amp_rot = np.stack(
        [hp_rot.rotate_map_alms(beam[:, 0]), hp_rot.rotate_map_alms(beam[:, 1])],
        axis=-1,
    )

    # Need to correct for change of polarisation coordinate system
    # Work out projection of theta_hat and phi_hat onto theta_hat and
    # phi_hat from inverse rotation
    cart_sky = coord.sph_to_cart(angpos)
    inv_rot_cart_sky = cart_sky @ hp_rot.mat  # Equivalent to (hp_rot.mat.T @ cart_sky)
    inv_rot_angpos = coord.cart_to_sph(inv_rot_cart_sky)[..., 1:]

    thetaphi_hat_sky = np.stack(coord.thetaphi_plane_cart(angpos), axis=1)
    thetaphi_hat_beam = np.stack(coord.thetaphi_plane_cart(inv_rot_angpos), axis=-1)
    beam_to_sky = thetaphi_hat_sky @ thetaphi_hat_beam

    return (beam_to_sky @ thph_amp_rot[..., None]).squeeze()


def pointing_offset_thetaphi_coords(
    angpos: np.ndarray, zenith: np.ndarray, altaz_pointing: np.ndarray
) -> np.ndarray:
    cart_sky = coord.sph_to_cart(angpos)

    full_rot = pointing_rot_mat(zenith, altaz_pointing)

    off_cart = cart_sky @ full_rot.T
    off_thetaphi = coord.cart_to_sph(off_cart)[:, 1:]

    # Set phi convention as anti-clockwise from local vertical for
    # alt < 90. For alt >= 90, use an "over-the-top" extension of this
    # convention.
    off_thetaphi[:, 1] = np.pi - off_thetaphi[:, 1]

    return off_thetaphi


def cocr_to_thetaphi(
    cocr_beam: np.ndarray,
    pol_type: POLTYPE,
    angpos: np.ndarray,
    zenith: np.ndarray,
    altaz_pointing: np.ndarray,
) -> np.ndarray:

    prot = pointing_rot_mat(zenith, altaz_pointing)
    # Local vertical is in local -xhat direction, local horizontal (phi=90 in pointing
    # offset convention) in local yhat direction. Convert these to the sky coordinates
    # using the inverse of the pointing transform.
    vertical = prot.T @ np.array([-1, 0, 0])
    horizontal = prot.T @ np.array([0, 1, 0])

    if pol_type == "X":
        co_dir, cr_dir = horizontal, vertical
    elif pol_type == "Y":
        co_dir, cr_dir = vertical, horizontal

    EthEph = (
        polpattern(angpos, co_dir) * cocr_beam[:, 0, None]
        + polpattern(angpos, cr_dir) * cocr_beam[:, 1, None]
    )

    return EthEph


def pointing_offset_separation(
    angpos: np.ndarray,
    zenith: np.ndarray,
    altaz_pointing: np.ndarray,
    degrees: Optional[bool] = False,
) -> np.ndarray:

    off_theta = pointing_offset_thetaphi_coords(angpos, zenith, altaz_pointing)[:, 0]

    if degrees:
        return np.degrees(off_theta)
    else:
        return off_theta


def pointing_offset_angles(
    angpos: np.ndarray,
    zenith: np.ndarray,
    altaz_pointing: np.ndarray,
    degrees: Optional[bool] = False,
) -> np.ndarray:

    off_thetaphi = pointing_offset_thetaphi_coords(angpos, zenith, altaz_pointing)

    off_lat = off_thetaphi[:, 0] * np.cos(off_thetaphi[:, 1])
    off_lon = off_thetaphi[:, 0] * np.sin(off_thetaphi[:, 1])

    latlon = np.stack([off_lat, off_lon], axis=-1)
    if degrees:
        return np.degrees(latlon)
    else:
        return latlon


def airy_beam(
    separations: FloatArrayLike,
    wavelength: FloatArrayLike,
    diameter: FloatArrayLike,
    voltage: Optional[bool] = True,
    zero_over_horizon: Optional[bool] = True,
) -> FloatArrayLike:

    x = np.pi * diameter / wavelength * np.sin(separations)
    out = 2 * bessel_j1(x) / x  # (Voltage Beam)

    if zero_over_horizon:
        out[separations > np.pi / 2] = 0

    if voltage:
        return out
    else:
        return out ** 2


def gaussian(
    separations: FloatArrayLike,
    wavelength: FloatArrayLike,
    diameter: FloatArrayLike,
    fwhm_factor: FloatArrayLike,
    voltage: Optional[bool] = True,
) -> FloatArrayLike:

    fwhm = fwhm_factor * wavelength / diameter

    sigma = FWHM2SIGMA * fwhm
    arg = -(separations ** 2) / 2 / sigma ** 2

    if voltage:
        return np.exp(arg / 2)
    else:
        return np.exp(arg)


class AnalyticCoPolBeam(config.Reader, metaclass=abc.ABCMeta):

    crosspol_type = config.enum(["pure", "scaled"], default="pure")
    crosspol_scale_dB = config.Property(proptype=float, default=-40)

    supports_pointing = True

    @abc.abstractmethod
    def beam_func(
        self, tel_obj: TransitTelescope, feed_ind: int, freq_ind: int, altaz_pointing
    ):
        pass

    def __call__(
        self,
        tel_obj: TransitTelescope,
        feed_ind: int,
        freq_ind: int,
        altaz_pointing: Optional[np.ndarray] = None,
    ) -> np.ndarray:

        if altaz_pointing is None:
            altaz_pointing = np.radians([90, 180])

        copol_beam = self.beam_func(tel_obj, feed_ind, freq_ind, altaz_pointing)

        if tel_obj.num_pol_sky == 1:
            return copol_beam
        else:

            if self.pol_type == "pure":
                cocr_beam = np.stack([copol_beam, np.zeros_like(copol_beam)], axis=-1)
            elif self.pol_type == "scaled":
                cocr_beam = np.stack(
                    [copol_beam, copol_beam * (10 ** (self.crosspol_scale_dB / 10))],
                    axis=-1,
                )

            return cocr_to_thetaphi(
                cocr_beam,
                pol_type=tel_obj.polarisation[feed_ind],
                angpos=tel_obj._angpos,
                zenith=tel_obj.zenith,
                altaz_pointing=altaz_pointing,
            )


class AiryBeam(AnalyticCoPolBeam):

    diameter = config.Property(proptype=float, default=6.0)

    def beam_func(
        self, tel_obj: TransitTelescope, feed_ind: int, freq_ind: int, altaz_pointing
    ) -> np.ndarray:

        seps = pointing_offset_separation(
            tel_obj._angpos, tel_obj.zenith, altaz_pointing=altaz_pointing
        )

        return np.array(airy_beam(seps, tel_obj.wavelengths[freq_ind], self.diameter))


class GaussianBeam(AnalyticCoPolBeam):

    dish_diameter = config.Property(proptype=float, default=6)
    fwhm_factor = config.Property(proptype=float, default=1.0)

    def beam_func(
        self, tel_obj: TransitTelescope, feed_ind: int, freq_ind: int, altaz_pointing
    ) -> np.ndarray:

        seps = pointing_offset_separation(
            tel_obj._angpos, tel_obj.zenith, altaz_pointing=altaz_pointing
        )

        return np.array(
            gaussian(
                seps,
                tel_obj.wavelengths[freq_ind],
                self.dish_diameter,
                self.fwhm_factor,
            )
        )


class HEALPixBeamFile(config.Reader):

    filename = config.Property(proptype=str)
    single_input = config.Property(proptype=bool, default=True)
    freq_index_type = config.enum(["matched", "nearest"], default="matched")

    supports_pointing = False

    def _finalise_config(self):
        from draco.core.containers import HEALPixBeam

        self.beam_file = HEALPixBeam.from_file(
            self.filename, mode="r", distributed=False, ondisk=True
        )

        super()._finalise_config()

    def __call__(
        self, tel_obj: TransitTelescope, feed_ind: int, freq_ind: int
    ) -> np.ndarray:

        pol_type = tel_obj.polarisation[feed_ind]
        ipol = self.beam_file.pol.astype(str).tolist().index(pol_type)

        if self.single_input:
            ifeed = 0
        else:
            ifeed = feed_ind

        if self.freq_index_type == "matched":
            ifreq = freq_ind
            if self.beam_file.freq[ifreq] != tel_obj.frequencies[ifreq]:
                raise ValueError # Do this properly.
        elif self.freq_index_type == "nearest":
            abs_diffs = np.abs(tel_obj.frequencies[freq_ind] - self.beam_file.freq)
            ifreq = int(np.argmin(abs_diffs))
            # TODO raise warning if abs_diffs[ifreq] passes some thresold.

        beam = self.beam_file.beam[ifreq, ipol, ifeed, :]

        Eth = hp.ud_grade(beam["Et"], tel_obj._nside)
        Eph = hp.ud_grade(beam["Ep"], tel_obj._nside)

        return np.stack([Eth, Eph], axis=-1)


AVAILABLE_BEAMS = {
    "airy": AiryBeam,
    "gaussian": GaussianBeam,
    "healpix": HEALPixBeamFile,
}
