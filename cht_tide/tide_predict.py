"""Tidal prediction convenience function.

Wraps :class:`~cht_tide.tide.Tide` to predict water levels from a
constituent DataFrame using the NOAA standard constituent set.
"""

import pandas as pd

import cht_tide.constituent as cons
from cht_tide.tide import Tide


def predict(data: pd.DataFrame, times, format: str = "np"):
    """Predict tidal water levels from harmonic constituents.

    Maps constituent names in *data* (e.g. ``"MM"``, ``"MF"``) to the
    corresponding NOAA objects, builds a :class:`~cht_tide.tide.Tide`
    model, and evaluates it at *times*.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by constituent name with ``"amplitude"`` (metres)
        and ``"phase"`` (degrees) columns.  Legacy integer-column DataFrames
        are also accepted (first non-index column = amplitude, second = phase).
    times : array-like of datetime
        Times at which to evaluate the tidal prediction.
    format : str, optional
        Output format: ``"np"`` (default) returns a numpy array;
        ``"dataframe"`` / ``"df"`` returns a :class:`pandas.DataFrame`
        indexed by time.

    Returns
    -------
    np.ndarray or pd.DataFrame
        Predicted tidal heights in metres.
    """
    all_constituents = [c for c in cons.noaa if c != cons._Z0]
    constituents = []
    amplitudes = []
    phases = []
    for name in data.index.to_list():
        noaa_name = name
        if name == "MM":
            noaa_name = "Mm"
        if name == "MF":
            noaa_name = "Mf"
        if name == "SA":
            noaa_name = "Sa"
        if name == "SSA":
            noaa_name = "Ssa"
        if name == "MU2":
            noaa_name = "mu2"
        if name == "NU2":
            noaa_name = "nu2"
        for cnst in all_constituents:
            if cnst.name == noaa_name:
                constituents.append(cnst)
                if "amplitude" in data.columns:
                    amplitudes.append(data.loc[name, "amplitude"])
                else:
                    amplitudes.append(data.loc[name, 1])
                if "phase" in data.columns:
                    phases.append(data.loc[name, "phase"])
                else:
                    phases.append(data.loc[name, 2])
                continue

    td = Tide(
        constituents=constituents,
        amplitudes=amplitudes,
        phases=phases,
    )
    v = td.at(times)

    if format == "dataframe" or format == "df":
        v = pd.DataFrame(v, index=times)

    return v
