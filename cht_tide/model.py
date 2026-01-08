# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:58:08 2021

@author: Maarten van Ormondt
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
    """
    Tide model
    """

    def __init__(self):
        self.database = None
        self.name = ""
        self.long_name = ""
        self.path = ""
        self.main_constituents = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1"]
        self.files = []

    def read_metadata(self):
        # Read metadata file
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

    def download(self):
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
        self, gdf=None, x=None, y=None, crs=None, format="gdf", constituents="all"
    ):
        """
        x can be a list of x coordinates, or and array of x coordinates
        y can be a list of y coordinates, or and array of y coordinates
        """

        # Download files if needed
        self.download()

        # Return pandas dataframe with constituents as rows and amplitudes and phases as columns

        if constituents == "all":
            constituents = self.constituents
        elif constituents == "main":
            constituents = self.main_constituents

        if gdf is not None:
            # Transform gdf to lon, lat (that's what the tide model uses)
            gdf4326 = gdf.to_crs("EPSG:4326")
            # Get extent of gdf
            xl = [gdf4326.geometry.x.min(), gdf4326.geometry.x.max()]
            yl = [gdf4326.geometry.y.min(), gdf4326.geometry.y.max()]
            # Add a little buffer
            xl[0] -= 0.25
            xl[1] += 0.25
            yl[0] -= 0.25
            yl[1] += 0.25
        else:
            # Make gdf from x and y
            gdf = pd.DataFrame()
            gdf["geometry"] = [Point(x, y) for x, y in zip(x, y)]
            gdf = gpd.GeoDataFrame(gdf, crs=crs)

        # Get the data in the extent
        ds = self.get_data(xl, yl, constituents=constituents)

        if format == "gdf" or format == "geodataframe":
            if "astro" not in gdf.columns:
                gdf["astro"] = None
            # Create geodataframe with points
            # Loop over points
            for i, row in gdf.to_crs("EPSG:4326").iterrows():
                x = np.mod(row.geometry.x, 360.0)
                y = row.geometry.y
                # First convert tidal data to vector
                ds["tvu"] = ds.amplitude * np.cos(ds.phase * np.pi / 180.0)
                ds["tvv"] = ds.amplitude * np.sin(ds.phase * np.pi / 180.0)
                # Interpolate
                dsp = ds.interp(lon=x, lat=y)
                df = pd.DataFrame()
                df["constituent"] = constituents
                # Now convert back to amplitude and phase
                dsp["amplitude"] = np.sqrt(dsp.tvu**2 + dsp.tvv**2)
                dsp["phase"] = np.mod(
                    np.arctan2(dsp.tvv, dsp.tvu) * 180.0 / np.pi, 360.0
                )
                df["amplitude"] = dsp.amplitude.to_numpy()
                df["phase"] = dsp.phase.to_numpy()
                df = df.set_index("constituent")
                # Set "astro" column in row i to df
                gdf.at[i, "astro"] = df  # noqa: PD008
            return gdf
        elif format == "dataframe" or format == "df" or format == "pandas":
            # Return list with dataframes
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

    def add_offset(self, data, offset=0):

        def _add_or_update(df):
            if 'A0' in df.index:
                df.loc['A0', "amplitude"] += offset
            else:
                new_row = pd.DataFrame({
                    "amplitude": [offset],
                    "phase": [0]
                }, index=['A0'])
                df = pd.concat([df, new_row])
            return df

        if isinstance(data, gpd.GeoDataFrame):
            for i in data.index:
                if isinstance(data.at[i, "astro"], pd.DataFrame):
                    data.at[i, "astro"] = _add_or_update(data.at[i, "astro"])
            return data

        elif isinstance(data, list):
            # List of DataFrames (format="dataframe")
            return [_add_or_update(df) for df in data]