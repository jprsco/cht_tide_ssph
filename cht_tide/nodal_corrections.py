"""Nodal correction functions for tidal harmonic analysis.

Provides the amplitude node factors *f* and phase corrections *u* (in
degrees) for each supported tidal constituent, following Schureman's
*Manual of Harmonic Analysis and Prediction of Tides* (1958).

All public functions accept a dict of :class:`~cht_tide.astro.AstronomicalParameter`
objects (as returned by :func:`~cht_tide.astro.astro`) and return a scalar.
"""

# Copyright (C) 2020 Deltares — Freek Scheel <freek.scheel@deltares.nl>
# GNU Lesser General Public License v3 or later.

import numpy as np

d2r, r2d = np.pi / 180.0, 180.0 / np.pi

# The following functions take a dictionary of astronomical values (in degrees)
# and return dimensionless scale factors for constituent amplitudes.


def f_unity(a: dict) -> float:
    """Node factor equal to unity (no modulation).

    Parameters
    ----------
    a : dict
        Astronomical parameters (unused).

    Returns
    -------
    float
        Always ``1.0``.
    """
    return 1.0


def f_Mm(a: dict) -> float:
    """Amplitude node factor for Mm (Schureman equations 73, 65).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    omega = d2r * a["omega"].value
    i = d2r * a["i"].value
    I = d2r * a["I"].value
    mean = (2 / 3.0 - np.sin(omega) ** 2) * (1 - 3 / 2.0 * np.sin(i) ** 2)
    return (2 / 3.0 - np.sin(I) ** 2) / mean


def f_Mf(a: dict) -> float:
    """Amplitude node factor for Mf (Schureman equations 74, 66).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    omega = d2r * a["omega"].value
    i = d2r * a["i"].value
    I = d2r * a["I"].value
    mean = np.sin(omega) ** 2 * np.cos(0.5 * i) ** 4
    return np.sin(I) ** 2 / mean


def f_O1(a: dict) -> float:
    """Amplitude node factor for O1 (Schureman equations 75, 67).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    omega = d2r * a["omega"].value
    i = d2r * a["i"].value
    I = d2r * a["I"].value
    mean = np.sin(omega) * np.cos(0.5 * omega) ** 2 * np.cos(0.5 * i) ** 4
    return (np.sin(I) * np.cos(0.5 * I) ** 2) / mean


def f_J1(a: dict) -> float:
    """Amplitude node factor for J1 (Schureman equations 76, 68).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    omega = d2r * a["omega"].value
    i = d2r * a["i"].value
    I = d2r * a["I"].value
    mean = np.sin(2 * omega) * (1 - 3 / 2.0 * np.sin(i) ** 2)
    return np.sin(2 * I) / mean


def f_OO1(a: dict) -> float:
    """Amplitude node factor for OO1 (Schureman equations 77, 69).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    omega = d2r * a["omega"].value
    i = d2r * a["i"].value
    I = d2r * a["I"].value
    mean = np.sin(omega) * np.sin(0.5 * omega) ** 2 * np.cos(0.5 * i) ** 4
    return np.sin(I) * np.sin(0.5 * I) ** 2 / mean


def f_M2(a: dict) -> float:
    """Amplitude node factor for M2 (Schureman equations 78, 70).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    omega = d2r * a["omega"].value
    i = d2r * a["i"].value
    I = d2r * a["I"].value
    mean = np.cos(0.5 * omega) ** 4 * np.cos(0.5 * i) ** 4
    return np.cos(0.5 * I) ** 4 / mean


def f_K1(a: dict) -> float:
    """Amplitude node factor for K1 (Schureman equations 227, 226, 68).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    omega = d2r * a["omega"].value
    i = d2r * a["i"].value
    I = d2r * a["I"].value
    nu = d2r * a["nu"].value
    sin2Icosnu_mean = np.sin(2 * omega) * (1 - 3 / 2.0 * np.sin(i) ** 2)
    mean = 0.5023 * sin2Icosnu_mean + 0.1681
    return (
        0.2523 * np.sin(2 * I) ** 2 + 0.1689 * np.sin(2 * I) * np.cos(nu) + 0.0283
    ) ** (0.5) / mean


def f_L2(a: dict) -> float:
    """Amplitude node factor for L2 (Schureman equations 215, 213, 204).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    P = d2r * a["P"].value
    I = d2r * a["I"].value
    R_a_inv = (
        1 - 12 * np.tan(0.5 * I) ** 2 * np.cos(2 * P) + 36 * np.tan(0.5 * I) ** 4
    ) ** (0.5)
    return f_M2(a) * R_a_inv


def f_K2(a: dict) -> float:
    """Amplitude node factor for K2 (Schureman equations 235, 234, 71).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    omega = d2r * a["omega"].value
    i = d2r * a["i"].value
    I = d2r * a["I"].value
    nu = d2r * a["nu"].value
    sinsqIcos2nu_mean = np.sin(omega) ** 2 * (1 - 3 / 2.0 * np.sin(i) ** 2)
    mean = 0.5023 * sinsqIcos2nu_mean + 0.0365
    return (
        0.2533 * np.sin(I) ** 4 + 0.0367 * np.sin(I) ** 2 * np.cos(2 * nu) + 0.0013
    ) ** (0.5) / mean


def f_M1(a: dict) -> float:
    """Amplitude node factor for M1 (Schureman equations 206, 207, 195).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    P = d2r * a["P"].value
    I = d2r * a["I"].value
    Q_a_inv = (
        0.25
        + 1.5 * np.cos(I) * np.cos(2 * P) * np.cos(0.5 * I) ** (-0.5)
        + 2.25 * np.cos(I) ** 2 * np.cos(0.5 * I) ** (-4)
    ) ** (0.5)
    return f_O1(a) * Q_a_inv


def f_Modd(a: dict, n: int) -> float:
    """Amplitude node factor for odd M constituents (Schureman equation 149).

    Parameters
    ----------
    a : dict
        Astronomical parameters.
    n : int
        Species number of the constituent (e.g. 3 for M3).

    Returns
    -------
    float
        Dimensionless amplitude factor.
    """
    return f_M2(a) ** (n / 2.0)


# Node factors u, see Table 2 of Schureman.


def u_zero(a: dict) -> float:
    """Phase node correction equal to zero.

    Parameters
    ----------
    a : dict
        Astronomical parameters (unused).

    Returns
    -------
    float
        Always ``0.0``.
    """
    return 0.0


def u_Mf(a: dict) -> float:
    """Phase node correction for Mf.

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    return -2.0 * a["xi"].value


def u_O1(a: dict) -> float:
    """Phase node correction for O1.

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    return 2.0 * a["xi"].value - a["nu"].value


def u_J1(a: dict) -> float:
    """Phase node correction for J1.

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    return -a["nu"].value


def u_OO1(a: dict) -> float:
    """Phase node correction for OO1.

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    return -2.0 * a["xi"].value - a["nu"].value


def u_M2(a: dict) -> float:
    """Phase node correction for M2.

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    return 2.0 * a["xi"].value - 2.0 * a["nu"].value


def u_K1(a: dict) -> float:
    """Phase node correction for K1.

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    return -a["nup"].value


def u_L2(a: dict) -> float:
    """Phase node correction for L2 (Schureman equation 214).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    I = d2r * a["I"].value
    P = d2r * a["P"].value
    R = r2d * np.arctan(
        np.sin(2 * P) / (1 / 6.0 * np.tan(0.5 * I) ** (-2) - np.cos(2 * P))
    )
    return 2.0 * a["xi"].value - 2.0 * a["nu"].value - R


def u_K2(a: dict) -> float:
    """Phase node correction for K2.

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    return -2.0 * a["nupp"].value


def u_M1(a: dict) -> float:
    """Phase node correction for M1 (Schureman equation 202).

    Parameters
    ----------
    a : dict
        Astronomical parameters.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    I = d2r * a["I"].value
    P = d2r * a["P"].value
    Q = r2d * np.arctan((5 * np.cos(I) - 1) / (7 * np.cos(I) + 1) * np.tan(P))
    return a["xi"].value - a["nu"].value + Q


def u_Modd(a: dict, n: int) -> float:
    """Phase node correction for odd M constituents.

    Parameters
    ----------
    a : dict
        Astronomical parameters.
    n : int
        Species number of the constituent.

    Returns
    -------
    float
        Phase correction in degrees.
    """
    return n / 2.0 * u_M2(a)
