"""Tidal harmonic analysis and prediction.

Provides the :class:`Tide` class for predicting tidal water levels from
harmonic constituents and for decomposing observed time series into
constituent amplitudes and phases via least-squares fitting.
"""

# Copyright (C) 2020 Deltares — Freek Scheel <freek.scheel@deltares.nl>
# GNU Lesser General Public License v3 or later.

from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime, timedelta
from itertools import count, takewhile

import numpy as np
from scipy.optimize import fsolve, leastsq

import cht_tide.constituent as constituent
from cht_tide.astro import astro

d2r, r2d = np.pi / 180.0, 180.0 / np.pi


class Tide:
    """Tidal model built from harmonic constituents.

    Can be constructed directly from constituent data or fitted to an
    observed time series via :meth:`decompose`.

    Parameters
    ----------
    constituents : list, optional
        List of :class:`~cht_tide.constituent.BaseConstituent` objects.
    amplitudes : array-like, optional
        Amplitudes corresponding to *constituents* (metres).
    phases : array-like, optional
        Phases corresponding to *constituents* (degrees unless *radians*
        is ``True``).
    model : np.ndarray, optional
        Structured array with ``dtype == Tide.dtype`` containing
        constituent, amplitude, and phase data.
    radians : bool, optional
        If ``True``, phases are interpreted as radians (default ``False``).

    Raises
    ------
    ValueError
        If neither (constituents, amplitudes, phases) nor model is provided,
        or if their lengths are inconsistent.
    """

    dtype = np.dtype([("constituent", object), ("amplitude", float), ("phase", float)])

    def __init__(
        self,
        constituents=None,
        amplitudes=None,
        phases=None,
        model=None,
        radians: bool = False,
    ) -> None:
        if None not in [constituents, amplitudes, phases]:
            if len(constituents) == len(amplitudes) == len(phases):
                model = np.zeros(len(phases), dtype=Tide.dtype)
                model["constituent"] = np.array(constituents)
                model["amplitude"] = np.array(amplitudes)
                model["phase"] = np.array(phases)
            else:
                raise ValueError(
                    "Constituents, amplitudes and phases should all be arrays of equal length."
                )
        elif model is not None:
            if not model.dtype == Tide.dtype:
                raise ValueError("Model must be a numpy array with dtype == Tide.dtype")
        else:
            raise ValueError(
                "Must be initialised with constituents, amplitudes and phases; or a model."
            )
        if radians:
            model["phase"] = r2d * model["phase"]
        self.model = model[:]
        self.normalize()

    def prepare(self, *args, **kwargs):
        """Delegate to :meth:`_prepare` with this instance's constituents.

        Returns
        -------
        tuple
            ``(speed, u, f, V0)`` — see :meth:`_prepare`.
        """
        return Tide._prepare(self.model["constituent"], *args, **kwargs)

    @staticmethod
    def _prepare(constituents, t0, t=None, radians: bool = True) -> tuple:
        """Compute constituent speeds, node factors, and equilibrium arguments.

        Parameters
        ----------
        constituents : list
            Tidal constituents to process.
        t0 : datetime
            Reference time at which speed and V0 are evaluated.
        t : list of datetime, optional
            Times at which node factors are evaluated (default: ``[t0]``).
        radians : bool, optional
            Return angular values in radians when ``True`` (default).

        Returns
        -------
        tuple of (np.ndarray, list, list, np.ndarray)
            ``speed`` (shape ``(n, 1)``), ``u`` (list of ``(n, 1)`` arrays),
            ``f`` (list of ``(n, 1)`` arrays), ``V0`` (shape ``(n, 1)``).
        """
        if isinstance(t0, Iterable):
            t0 = t0[0]
        if t is None:
            t = [t0]
        if not isinstance(t, Iterable):
            t = [t]
        a0 = astro(t0)
        a = [astro(t_i) for t_i in t]

        V0 = np.array([c.V(a0) for c in constituents])[:, np.newaxis]
        speed = np.array([c.speed(a0) for c in constituents])[:, np.newaxis]
        u = [
            np.mod(np.array([c.u(a_i) for c in constituents])[:, np.newaxis], 360.0)
            for a_i in a
        ]
        f = [
            np.mod(np.array([c.f(a_i) for c in constituents])[:, np.newaxis], 360.0)
            for a_i in a
        ]

        if radians:
            speed = d2r * speed
            V0 = d2r * V0
            u = [d2r * each for each in u]
        return speed, u, f, V0

    def at(self, t) -> np.ndarray:
        """Return modelled tidal heights at given times.

        Parameters
        ----------
        t : array-like of datetime
            Times at which to evaluate the tidal height.

        Returns
        -------
        np.ndarray
            Tidal heights in metres.
        """
        t0 = t[0]
        hours = self._hours(t0, t)
        partition = 240.0
        t = self._partition(hours, partition)
        times = self._times(t0, [(i + 0.5) * partition for i in range(len(t))])
        speed, u, f, V0 = self.prepare(t0, times, radians=True)
        H = self.model["amplitude"][:, np.newaxis]
        p = d2r * self.model["phase"][:, np.newaxis]

        return np.concatenate(
            [
                Tide._tidal_series(t_i, H, p, speed, u_i, f_i, V0)
                for t_i, u_i, f_i in zip(t, u, f)
            ]
        )

    def highs(self, *args):
        """Generator yielding only the high tides.

        Parameters
        ----------
        *args
            Forwarded to :meth:`extrema`.

        Yields
        ------
        tuple
            ``(time, height, "H")`` for each high water.
        """
        for t in filter(lambda e: e[2] == "H", self.extrema(*args)):
            yield t

    def lows(self, *args):
        """Generator yielding only the low tides.

        Parameters
        ----------
        *args
            Forwarded to :meth:`extrema`.

        Yields
        ------
        tuple
            ``(time, height, "L")`` for each low water.
        """
        for t in filter(lambda e: e[2] == "L", self.extrema(*args)):
            yield t

    def form_number(self) -> float:
        """Return the tidal form number ``(K1 + O1) / (M2 + S2)``.

        Returns
        -------
        float
            Form number (dimensionless).
        """
        k1, o1, m2, s2 = (
            np.extract(self.model["constituent"] == c, self.model["amplitude"])
            for c in [
                constituent._K1,
                constituent._O1,
                constituent._M2,
                constituent._S2,
            ]
        )
        return (k1 + o1) / (m2 + s2)

    def classify(self) -> str:
        """Classify the tide type based on the form number.

        Returns
        -------
        str
            One of ``"semidiurnal"``, ``"mixed (semidiurnal)"``,
            ``"mixed (diurnal)"``, or ``"diurnal"``.
        """
        form = self.form_number()
        if 0 <= form <= 0.25:
            return "semidiurnal"
        elif 0.25 < form <= 1.5:
            return "mixed (semidiurnal)"
        elif 1.5 < form <= 3.0:
            return "mixed (diurnal)"
        else:
            return "diurnal"

    def extrema(self, t0, t1=None, partition: float = 2400.0):
        """Generate high and low tidal extrema.

        Parameters
        ----------
        t0 : datetime
            Start time after which extrema are sought.
        t1 : datetime, optional
            End time; if not given the generator is infinite.
        partition : float, optional
            Hours over which node factors are held constant (default 2400).

        Yields
        ------
        tuple of (datetime, float, str)
            ``(time, height, hilo)`` where ``hilo`` is ``"H"`` or ``"L"``.
        """
        if t1:
            for e in takewhile(lambda t: t[0] < t1, self.extrema(t0)):
                yield e
        else:
            delta = np.amin(
                [
                    90.0 / c.speed(astro(t0))
                    for c in self.model["constituent"]
                    if not c.speed(astro(t0)) == 0
                ]
            )
            offset = 24.0
            partitions = (
                (Tide._times(t0, i * partition) for i in count()),
                (Tide._times(t0, i * partition) for i in count(1)),
            )

            interval_count = int(np.ceil((partition + offset) / delta)) + 1
            amplitude = self.model["amplitude"][:, np.newaxis]
            phase = d2r * self.model["phase"][:, np.newaxis]

            for start, end in zip(*partitions):
                speed, [u], [f], V0 = self.prepare(
                    start, Tide._times(start, 0.5 * partition)
                )

                def d(t):
                    return np.sum(
                        -speed * amplitude * f * np.sin(speed * t + (V0 + u) - phase),
                        axis=0,
                    )

                def d2(t):
                    return np.sum(
                        -(speed**2.0)
                        * amplitude
                        * f
                        * np.cos(speed * t + (V0 + u) - phase),
                        axis=0,
                    )

                intervals = (
                    (delta * i - offset for i in range(interval_count)),
                    (delta * (i + 1) - offset for i in range(interval_count)),
                )
                for a, b in zip(*intervals):
                    if d(a) * d(b) < 0:
                        extrema = fsolve(d, (a + b) / 2.0, fprime=d2)[0]
                        time = Tide._times(start, extrema)
                        [height] = self.at([time])
                        hilo = "H" if d2(extrema) < 0 else "L"
                        if start < time < end:
                            yield (time, height, hilo)

    @staticmethod
    def _hours(t0, t) -> np.ndarray:
        """Return hourly offsets of *t* relative to *t0*.

        Parameters
        ----------
        t0 : datetime
            Reference time.
        t : datetime or array-like
            Times to convert to hourly offsets.

        Returns
        -------
        np.ndarray or float
            Hourly offsets from *t0*.
        """
        if not isinstance(t, Iterable):
            return Tide._hours(t0, [t])[0]
        elif isinstance(t[0], datetime):
            return np.array([(ti - t0).total_seconds() / 3600.0 for ti in t])
        else:
            return t

    @staticmethod
    def _partition(hours: np.ndarray, partition: float = 3600.0) -> list:
        """Partition a sorted array of hours into sub-arrays.

        Parameters
        ----------
        hours : np.ndarray
            Sorted array of hour values.
        partition : float, optional
            Maximum sub-array length in hours (default 3600).

        Returns
        -------
        list of np.ndarray
            Sub-arrays, each spanning at most *partition* hours.
        """
        partition = float(partition)
        relative = hours - hours[0]
        total_partitions = np.ceil(
            relative[-1] / partition + 10 * np.finfo(np.float64).eps
        ).astype("int")
        return [
            hours[np.floor(np.divide(relative, partition)) == i]
            for i in range(total_partitions)
        ]

    @staticmethod
    def _times(t0, hours) -> np.ndarray:
        """Return datetimes corresponding to hourly offsets from *t0*.

        Parameters
        ----------
        t0 : datetime
            Reference time.
        hours : float or array-like
            Hourly offsets from *t0*.

        Returns
        -------
        np.ndarray of datetime or datetime
            Corresponding datetimes.
        """
        if not isinstance(hours, Iterable):
            return Tide._times(t0, [hours])[0]
        elif not isinstance(hours[0], datetime):
            return np.array([t0 + timedelta(hours=h) for h in hours])
        else:
            return np.array(hours)

    @staticmethod
    def _tidal_series(t, amplitude, phase, speed, u, f, V0) -> np.ndarray:
        """Evaluate the tidal series at times *t* for one partition.

        Parameters
        ----------
        t : np.ndarray
            Hours since epoch for this partition.
        amplitude : np.ndarray
            Constituent amplitudes, shape ``(n, 1)``.
        phase : np.ndarray
            Constituent phases in radians, shape ``(n, 1)``.
        speed : np.ndarray
            Constituent speeds in rad/h, shape ``(n, 1)``.
        u : np.ndarray
            Node phase corrections in radians, shape ``(n, 1)``.
        f : np.ndarray
            Node amplitude factors, shape ``(n, 1)``.
        V0 : np.ndarray
            Equilibrium arguments in radians, shape ``(n, 1)``.

        Returns
        -------
        np.ndarray
            Tidal heights at times *t*.
        """
        return np.sum(amplitude * f * np.cos(speed * t + (V0 + u) - phase), axis=0)

    def normalize(self) -> None:
        """Normalise the model so amplitudes are positive and phases are in [0, 360)."""
        for i in range(len(self.model)):
            if self.model["amplitude"][i] < 0:
                self.model["amplitude"][i] = -self.model["amplitude"][i]
                self.model["phase"][i] = self.model["phase"][i] + 180.0
            self.model["phase"][i] = np.mod(self.model["phase"][i], 360.0)

    @classmethod
    def decompose(
        cls,
        heights: np.ndarray,
        t=None,
        t0=None,
        interval=None,
        constituents=constituent.noaa,
        initial=None,
        n_period: int = 2,
        callback=None,
        full_output: bool = False,
    ):
        """Fit tidal constituents to an observed height time series.

        Parameters
        ----------
        heights : np.ndarray
            Observed tidal heights.
        t : array-like, optional
            Array of datetimes or hours since *t0* for each observation.
        t0 : datetime, optional
            Reference time (required when *t* is in hours).
        interval : float, optional
            Uniform sampling interval in hours (used when *t* is not given).
        constituents : list, optional
            Constituents to fit (default: :data:`~cht_tide.constituent.noaa`).
        initial : Tide, optional
            Initial guess for amplitudes and phases.
        n_period : int, optional
            Minimum number of complete cycles required for a constituent
            to be included (default 2).
        callback : callable, optional
            Called at each solver iteration with the current residual array.
        full_output : bool, optional
            If ``True``, also return the raw ``leastsq`` output.

        Returns
        -------
        Tide
            Fitted tidal model.
        tuple, optional
            ``(Tide, lsq_output)`` when *full_output* is ``True``.

        Raises
        ------
        ValueError
            If insufficient time information is provided.
        """
        if t is not None:
            if isinstance(t[0], datetime):
                hours = Tide._hours(t[0], t)
                t0 = t[0]
            elif t0 is not None:
                hours = t
            else:
                raise ValueError(
                    "t can be an array of datetimes, or an array "
                    "of hours since t0 in which case t0 must be "
                    "specified."
                )
        elif None not in [t0, interval]:
            hours = np.arange(len(heights)) * interval
        else:
            raise ValueError(
                "Must provide t(datetimes), or t(hours) and "
                "t0(datetime), or interval(hours) and t0(datetime) "
                "so that each height can be identified with an "
                "instant in time."
            )

        # Remove duplicate constituents (those which travel at exactly the same
        # speed, irrespective of phase)
        constituents = list(OrderedDict.fromkeys(constituents))

        # No need for least squares to find the mean water level constituent z0,
        # work relative to mean
        constituents = [c for c in constituents if not c == constituent._Z0]
        z0 = np.mean(heights)
        heights = heights - z0

        # Only analyse frequencies which complete at least n_period cycles over
        # the data period.
        constituents = [
            c for c in constituents if 360.0 * n_period < hours[-1] * c.speed(astro(t0))
        ]
        n = len(constituents)

        sort = np.argsort(hours)
        hours = hours[sort]
        heights = heights[sort]

        partition = 240.0

        t = Tide._partition(hours, partition)
        times = Tide._times(t0, [(i + 0.5) * partition for i in range(len(t))])

        speed, u, f, V0 = Tide._prepare(constituents, t0, times, radians=True)

        def residual(hp):
            H, p = hp[:n, np.newaxis], hp[n:, np.newaxis]
            s = np.concatenate(
                [
                    Tide._tidal_series(t_i, H, p, speed, u_i, f_i, V0)
                    for t_i, u_i, f_i in zip(t, u, f)
                ]
            )
            res = heights - s
            if callback:
                callback(res)
            return res

        def D_residual(hp):
            H, p = hp[:n, np.newaxis], hp[n:, np.newaxis]
            ds_dH = np.concatenate(
                [
                    f_i * np.cos(speed * t_i + u_i + V0 - p)
                    for t_i, u_i, f_i in zip(t, u, f)
                ],
                axis=1,
            )

            ds_dp = np.concatenate(
                [
                    H * f_i * np.sin(speed * t_i + u_i + V0 - p)
                    for t_i, u_i, f_i in zip(t, u, f)
                ],
                axis=1,
            )

            return np.append(-ds_dH, -ds_dp, axis=0)

        amplitudes = np.ones(n) * (np.sqrt(np.dot(heights, heights)) / len(heights))
        phases = np.ones(n)

        if initial:
            for c0, amplitude, phase in initial.model:
                for i, c in enumerate(constituents):
                    if c0 == c:
                        amplitudes[i] = amplitude
                        phases[i] = d2r * phase

        initial = np.append(amplitudes, phases)

        lsq = leastsq(residual, initial, Dfun=D_residual, col_deriv=True, ftol=1e-7)

        model = np.zeros(1 + n, dtype=cls.dtype)
        model[0] = (constituent._Z0, z0, 0)
        model[1:]["constituent"] = constituents[:]
        model[1:]["amplitude"] = lsq[0][:n]
        model[1:]["phase"] = lsq[0][n:]

        if full_output:
            return cls(model=model, radians=True), lsq
        return cls(model=model, radians=True)
