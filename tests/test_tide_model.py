import pandas as pd
import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Point

from cht_tide.fes2014 import TideModelFes2014
from cht_tide.tide_predict import predict


@pytest.mark.skip(reason="TODO resolve absolute filepaths to data")
def test_tide_model():
    name = "fes2014"
    path = r"c:\work\delftdashboard\data\tidemodels\fes2014"
    mdl = TideModelFes2014(name, path)
    _ds = mdl.get_data([350.0, 351.0], [50.0, 51.0], constituents=["m2"])

    x = [350.0, 351.0, 352.0]
    y = [50.0, 51.0, 52.0]

    # Make GeoDataFrame with points
    gdf = GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in zip(x, y)]},
        crs="EPSG:4326",
    )

    # ds=mdl.get_data_on_points(gdf=gdf, constituents=["m2"])
    gdf = mdl.get_data_on_points(gdf=gdf, constituents="all")

    # Create timeseries
    times = pd.date_range(start="2023-01-01", end="2023-01-02", freq="10min")

    # Predict the tide

    for i, point in gdf.iterrows():
        v = predict(point["astro"], times)

    assert v.dtype == "float64"
    assert len(times) == len(v)
