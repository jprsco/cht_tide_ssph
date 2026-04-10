"""Base tide model class.

Defines :class:`TideModel`, the abstract base for all concrete tidal dataset
implementations (e.g. FES2014, TPXO).  Handles metadata loading, optional
S3 file download, and interpolation of tidal constituents onto arbitrary
point sets.
"""

import os

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import toml
from botocore import UNSIGNED
from botocore.client import Config
from pyproj import CRS
from shapely.geometry import Point


class TideModel:
    """Abstract base class for a tidal model dataset.

    Subclasses must implement :meth:`get_data` and call
    :meth:`read_metadata` during initialisation.

    Attributes
    ----------
    database : object or None
        Parent :class:`~cht_tide.database.TideModelDatabase` instance.
    name : str
        Short identifier for this dataset.
    long_name : str
        Human-readable label.
    path : str
        Directory containing dataset files.
    main_constituents : list of str
        Names of the eight principal tidal constituents.
    files : list of str
        File names that must exist locally (used for S3 download checks).
    """

    def __init__(self) -> None:
        self.database = None
        self.name = ""
        self.long_name = ""
        self.path = ""
        self.main_constituents = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
        self.files = []

    def read_metadata(self) -> None:
        """Read dataset metadata from ``metadata.tml`` in ``self.path``.

        Sets attributes for every key found in the TOML file.  Also handles
        the legacy ``longname`` key and ensures ``self.long_name`` is always
        populated.
        """
        tml_file = os.path.join(self.path, "metadata.tml")
        tml = toml.load(tml_file)
        for key in tml:
            setattr(self, key, tml[key])
        # Long name for backwards compatibility
        if "longname" in tml:
            self.long_name = tml["longname"]
        # Make sure there is always a long_name
        if self.long_name == "":
            self.long_name = self.name

        self.crs = CRS(4326)

    def download(self) -> None:
        """Download any missing dataset files from S3.

        Does nothing if ``self.s3_bucket`` is ``None`` or all files are
        already present locally.
        """
        if self.s3_bucket is None:
            return
        # Check if download is needed
        for file in self.files:
            if not os.path.exists(os.path.join(self.path, file)):
                s3_client = boto3.client(
                    "s3", config=Config(signature_version=UNSIGNED)
                )
                break
        # Get all files defined in the toml file
        for file in self.files:
            if not os.path.exists(os.path.join(self.path, file)):
                print(f"Downloading {file} from tide model {self.name} ...")
                s3_client.download_file(
                    self.s3_bucket,
                    f"{self.s3_key}/{file}",
                    os.path.join(self.path, file),
                )

    def get_data_on_points(
        self,
        gdf: gpd.GeoDataFrame = None,
        x=None,
        y=None,
        crs=None,
        format: str = "gdf",
        constituents="all",
    ):
        """Interpolate tidal constituents onto a set of geographic points.

        Provide either *gdf* **or** *x*/*y*/*crs*.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame, optional
            Points at which to extract tidal data.
        x : array-like, optional
            X (longitude) coordinates of points.
        y : array-like, optional
            Y (latitude) coordinates of points.
        crs : any, optional
            Coordinate reference system for *x*/*y* (passed to GeoPandas).
        format : str, optional
            Output format: ``"gdf"`` / ``"geodataframe"`` (default) or
            ``"dataframe"`` / ``"df"`` / ``"pandas"``.
        constituents : str or list of str, optional
            Which constituents to extract (``"all"`` or ``"main"``).

        Returns
        -------
        gpd.GeoDataFrame or list of pd.DataFrame
            When *format* is ``"gdf"``, the input GeoDataFrame with an
            ``"astro"`` column containing per-station DataFrames.
            When *format* is ``"dataframe"``, a list of DataFrames.
        """
        # Download files if needed
        self.download()

        if constituents == "all":
            constituents = self.constituents
        elif constituents == "main":
            constituents = self.main_constituents

        if gdf is not None:
            gdf4326 = gdf.to_crs("EPSG:4326")
            xl = [gdf4326.geometry.x.min(), gdf4326.geometry.x.max()]
            yl = [gdf4326.geometry.y.min(), gdf4326.geometry.y.max()]
            xl[0] -= 0.25
            xl[1] += 0.25
            yl[0] -= 0.25
            yl[1] += 0.25
        else:
            gdf = pd.DataFrame()
            gdf["geometry"] = [Point(x, y) for x, y in zip(x, y)]
            gdf = gpd.GeoDataFrame(gdf, crs=crs)

        ds = self.get_data(xl, yl, constituents=constituents)

        if format == "gdf" or format == "geodataframe":
            if "astro" not in gdf.columns:
                gdf["astro"] = None
            for i, row in gdf.to_crs("EPSG:4326").iterrows():
                x = np.mod(row.geometry.x, 360.0)
                y = row.geometry.y
                ds["tvu"] = ds.amplitude * np.cos(ds.phase * np.pi / 180.0)
                ds["tvv"] = ds.amplitude * np.sin(ds.phase * np.pi / 180.0)
                dsp = ds.interp(lon=x, lat=y)
                df = pd.DataFrame()
                df["constituent"] = constituents
                dsp["amplitude"] = np.sqrt(dsp.tvu**2 + dsp.tvv**2)
                dsp["phase"] = np.mod(
                    np.arctan2(dsp.tvv, dsp.tvu) * 180.0 / np.pi, 360.0
                )
                df["amplitude"] = dsp.amplitude.to_numpy()
                df["phase"] = dsp.phase.to_numpy()
                df = df.set_index("constituent")
                gdf.at[i, "astro"] = df  # noqa: PD008
            return gdf
        elif format == "dataframe" or format == "df" or format == "pandas":
            lst = []
            for i, row in gdf.to_crs("EPSG:4326").iterrows():
                dsp = ds.interp(
                    lon=np.array(row.geometry.x), lat=np.array(row.geometry.y)
                )
                df = pd.DataFrame()
                df["constituent"] = constituents
                df["amplitude"] = dsp.amplitude.to_numpy()
                df["phase"] = dsp.phase.to_numpy()
                df.set_index("constituent")
                lst.append(df)
            return lst

    def add_offset(self, data, offset: float = 0):
        """Add a constant vertical offset (datum correction) to tidal data.

        If the ``A0`` constituent is already present in the DataFrame its
        amplitude is incremented; otherwise a new row is appended.

        Parameters
        ----------
        data : gpd.GeoDataFrame or list of pd.DataFrame
            Tidal data as returned by :meth:`get_data_on_points`.
        offset : float, optional
            Vertical offset in metres (default ``0``).

        Returns
        -------
        gpd.GeoDataFrame or list of pd.DataFrame
            Updated tidal data with the offset applied.
        """

        def _add_or_update(df: pd.DataFrame) -> pd.DataFrame:
            if "A0" in df.index:
                df.loc["A0", "amplitude"] += offset
            else:
                new_row = pd.DataFrame(
                    {"amplitude": [offset], "phase": [0]}, index=["A0"]
                )
                df = pd.concat([df, new_row])
            return df

        if isinstance(data, gpd.GeoDataFrame):
            for i in data.index:
                if isinstance(data.at[i, "astro"], pd.DataFrame):
                    data.at[i, "astro"] = _add_or_update(data.at[i, "astro"])
            return data

        elif isinstance(data, list):
            return [_add_or_update(df) for df in data]
