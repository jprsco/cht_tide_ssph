# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:25:56 2021

@author: ormondt
"""

import pandas as pd

import cht_tide.constituent as cons
from cht_tide.tide import Tide


def predict(data, times, format="np"):
    all_constituents = [c for c in cons.noaa if c != cons._Z0]
    constituents = []
    amplitudes = []
    phases = []
    for name in data.index.to_list():
        okay = False
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
                # Check if amplitude is a column in the data
                if "amplitude" in data.columns:
                    amplitudes.append(data.loc[name, "amplitude"])
                else:
                    # Assume it is the first non-index column
                    amplitudes.append(data.loc[name, 1])
                if "phase" in data.columns:
                    phases.append(data.loc[name, "phase"])
                else:
                    # Assume it is the second non-index column
                    phases.append(data.loc[name, 2])
                okay = True
                continue
        # if not okay:
        #     print(
        #         f"Constituent {name} not found in list of NOAA constituents ! Skipping ..."
        #     )

    td = Tide(
        constituents=constituents,
        amplitudes=amplitudes,
        phases=phases,
    )
    v = td.at(times)

    if format == "dataframe" or format == "df":
        # Convert numpy array v to dataframe where index is time
        v = pd.DataFrame(v, index=times)

    return v
