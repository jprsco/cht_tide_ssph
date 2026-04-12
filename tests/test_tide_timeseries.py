from pathlib import Path

import numpy as np
import pandas as pd

from cht_tide.read_bca import SfincsBoundary
from cht_tide.tide_predict import predict


def test_timeseries_from_components():
    # Create dataframe with values
    data = {
        "amplitude": [
            0.39452,
            0.06667,
            0.08968,
            0.01860,
            0.05285,
            0.04300,
            0.01846,
            0.00857,
            0.01022,
            0.00451,
            0.00200,
            0.00101,
            0.00129,
            0.05300,
            0.07800,
        ],
        "phase": [
            14.38,
            40.10,
            352.95,
            36.25,
            217.27,
            234.41,
            220.42,
            233.82,
            348.71,
            347.03,
            180.00,
            89.38,
            202.68,
            50.50,
            176.30,
        ],
    }
    df = pd.DataFrame(
        data,
        index=[
            "M2",
            "S2",
            "N2",
            "K2",
            "K1",
            "O1",
            "P1",
            "Q1",
            "MF",
            "MM",
            "M4",
            "MS4",
            "MN4",
            "SSA",
            "SA",
        ],
    )
    df.index.name = "constituent"

    # Create timeseries
    times = pd.date_range(start="2023-01-01", end="2023-01-02", freq="10min")

    # Predict the tide
    v = predict(df, times)

    assert v.dtype == "float64"
    assert len(times) == len(v)

    range = np.max(v) - np.min(v)
    assert np.diff(v).max() < range / 10  # Check that the values smoothly change


def test_read_bca_file():
    test_folder = Path(__file__).parent

    bca_file = test_folder.joinpath("sfincs.bca")
    bnd_file = test_folder.joinpath("sfincs.bnd")

    sb = SfincsBoundary()
    sb.read_flow_boundary_points(bnd_file)
    sb.read_astro_boundary_conditions(bca_file)

    assert len(sb.flow_boundary_points) == 28
    assert isinstance(sb.flow_boundary_points[0].astro, pd.DataFrame)


def test_read_bca_and_predict_timeseries():
    test_folder = Path(__file__).parent
    bca_file = test_folder.joinpath("sfincs.bca")
    bnd_file = test_folder.joinpath("sfincs.bnd")

    sb = SfincsBoundary()
    sb.read_flow_boundary_points(bnd_file)
    sb.read_astro_boundary_conditions(bca_file)

    times = pd.date_range(start="2023-01-01", end="2023-01-02", freq="10min")

    # Predict the tide
    v = predict(sb.flow_boundary_points[0].astro, times)

    assert v.dtype == "float64"
    assert len(times) == len(v)

    range = np.max(v) - np.min(v)
    assert np.diff(v).max() < range / 10  # Check that the values smoothly change
