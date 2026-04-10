"""Tide station database management and prediction.

Provides :class:`TideStationsDataset` (a single collection of tide gauge
stations stored as a NetCDF file) and :class:`TideStationsDatabase`
(a registry that manages multiple datasets), plus the helper function
:func:`df2tekaltimeseries` for writing time series to Delft3D ``.tek``
files.
"""

import os

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import toml
import xarray as xr
from botocore import UNSIGNED
from botocore.client import Config

from cht_tide.tide_predict import predict


class TideStationsDataset:
    """A single collection of tide gauge stations in NetCDF format.

    Parameters
    ----------
    name : str
        Short identifier for this dataset.
    path : str
        Directory containing ``metadata.tml`` and the NetCDF data file.

    Attributes
    ----------
    gdf : gpd.GeoDataFrame
        Station locations; populated lazily by :meth:`get_gdf`.
    is_read : bool
        Whether the NetCDF data has been loaded into memory.
    station : list of dict
        Per-station metadata (``name``, ``id``, ``lon``, ``lat``).
    components : list of str
        Constituent names from the NetCDF file.
    """

    def __init__(self, name: str, path: str) -> None:
        self.name = name
        self.long_name = name
        self.path = path
        self.gdf = gpd.GeoDataFrame()
        self.is_read = False
        self.file = None
        self.s3_bucket = None
        self.s3_key = None
        self.s3_region = None
        self.read_metadata()

    def read_metadata(self) -> None:
        """Load dataset metadata from ``metadata.tml``."""
        metadata_path = os.path.join(self.path, "metadata.tml")
        if not os.path.exists(metadata_path):
            print(
                f"Warning! Tide stations metadata file not found: {metadata_path}"
            )
            return
        metadata = toml.load(metadata_path)
        if "longname" in metadata:
            self.long_name = metadata["longname"]
        elif "long_name" in metadata:
            self.long_name = metadata["long_name"]
        self.file = metadata["file"]
        if "s3_bucket" in metadata:
            self.s3_bucket = metadata["s3_bucket"]
        if "s3_key" in metadata:
            self.s3_key = metadata["s3_key"]
        if "s3_region" in metadata:
            self.s3_region = metadata["s3_region"]

    def check_file(self) -> bool:
        """Check whether the dataset NetCDF file exists locally.

        Returns
        -------
        bool
            ``True`` if the file is present (or no file is configured).
        """
        okay = True
        if self.file is not None:
            if not os.path.exists(os.path.join(self.path, self.file)):
                okay = False
        return okay

    def download(self) -> None:
        """Download the dataset file from S3 if it is missing locally."""
        if self.s3_bucket is None:
            return
        if not self.check_file():
            print(f"Downloading {self.file} for tide stations set {self.name} ...")
            s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            s3_client.download_file(
                self.s3_bucket,
                f"{self.s3_key}/{self.file}",
                os.path.join(self.path, self.file),
            )

    def read_data(self) -> None:
        """Load the NetCDF data file into memory.

        Populates ``self.station`` and ``self.components``.
        Does nothing if data has already been read.
        """
        if self.is_read:
            return
        filename = os.path.join(self.path, self.file)
        if not self.check_file():
            self.download()
        if not os.path.exists(filename):
            print(f"Warning! Tide stations dataset file not found: {filename}")
            return
        self.data = xr.load_dataset(filename)
        nr_stations = len(self.data["lon"])
        self.station = []
        for i in range(nr_stations):
            station = {}
            name = "unknown"
            name_s1 = self.data["stations"][:, i].to_numpy()
            try:
                name = "".join([x.decode("utf-8") for x in name_s1]).strip()
            except Exception:
                print("Error decoding name")
            id_s1 = self.data["idcodes"][:, i].to_numpy()
            id = "".join([x.decode("utf-8") for x in id_s1]).strip()
            station["name"] = name
            station["id"] = id
            station["lon"] = self.data["lon"][i].to_numpy()
            station["lat"] = self.data["lat"][i].to_numpy()
            self.station.append(station)
        nr_components = np.shape(self.data["components"])[1]
        self.components = []
        for i in range(nr_components):
            components_s1 = self.data["components"][:, i].to_numpy()
            self.components.append(
                "".join([x.decode("utf-8") for x in components_s1]).strip()
            )

        self.data.close()
        self.is_read = True

    def find_index_by_name(self, name: str) -> int | None:
        """Return the station index matching *name*, or ``None``.

        Parameters
        ----------
        name : str
            Station name to search for.

        Returns
        -------
        int or None
            Zero-based index of the matching station.
        """
        if not self.is_read:
            self.read_data()
        for i, station in enumerate(self.station):
            if station["name"] == name:
                return i
        return None

    def find_index_by_id(self, id: str) -> int | None:
        """Return the station index matching *id*, or ``None``.

        Parameters
        ----------
        id : str
            Station identifier to search for.

        Returns
        -------
        int or None
            Zero-based index of the matching station.
        """
        if not self.is_read:
            self.read_data()
        for i, station in enumerate(self.station):
            if station["id"] == id:
                return i
        return None

    def get_components(
        self,
        name: str = None,
        id: str = None,
        index: int = None,
        sort: bool = True,
    ) -> pd.DataFrame:
        """Return tidal harmonic components for one station.

        Exactly one of *name*, *id*, or *index* must be provided.

        Parameters
        ----------
        name : str, optional
            Station name.
        id : str, optional
            Station identifier.
        index : int, optional
            Zero-based station index (currently unused; use *name* or *id*).
        sort : bool, optional
            Sort components by amplitude descending and drop zero rows
            (default ``True``).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by constituent name with columns
            ``"amplitude"`` and ``"phase"``.
        """
        if name is not None:
            i = self.find_index_by_name(name)
        elif id is not None:
            i = self.find_index_by_id(id)
        else:
            print("Please provide either name or id.")
            return
        if i is None:
            print("Station not found.")
            return
        if not self.is_read:
            self.read_data()
        amplitudes = self.data["amplitude"][:, i].to_numpy()
        phases = self.data["phase"][:, i].to_numpy()
        df = pd.DataFrame(
            {"constituent": self.components, "amplitude": amplitudes, "phase": phases}
        )
        df = df.set_index("constituent")
        if sort:
            df = df.sort_values(by="amplitude", ascending=False)
            df = df[df.amplitude > 0.0]
        return df

    def predict(
        self,
        name: str = None,
        id: str = None,
        start=None,
        end=None,
        t=None,
        dt: float = None,
        offset: float = 0.0,
        format: str = "tek",
        filename: str = None,
    ) -> pd.DataFrame:
        """Predict tidal water levels for a station.

        Parameters
        ----------
        name : str, optional
            Station name.
        id : str, optional
            Station identifier.
        start : str or datetime, optional
            Start of the prediction period.
        end : str or datetime, optional
            End of the prediction period.
        t : array-like, optional
            Explicit array of times (overrides *start*/*end*).
        dt : float, optional
            Time step in seconds (default 600 s).
        offset : float, optional
            Constant offset to add to the predicted water level (metres).
        format : str, optional
            Output file format: ``"tek"`` or ``"csv"``.
        filename : str, optional
            If given, write the prediction to this file.

        Returns
        -------
        pd.DataFrame
            Predicted water levels indexed by time.
        """
        if name is not None:
            components = self.get_components(name=name)
        elif id is not None:
            components = self.get_components(id=id)
        else:
            print("Please provide either name or id.")
            return
        if start is not None and end is not None:
            if isinstance(start, str):
                start = pd.to_datetime(start)
            if isinstance(end, str):
                end = pd.to_datetime(end)
            if dt is None:
                dt = 600.0
            times = pd.date_range(start=start, end=end, freq=f"{dt}s")
        elif t is not None:
            times = t
        else:
            print("Please provide either t0 and t1, or t.")
            return
        prd = predict(components, times, format="df")
        prd = prd + offset
        if filename is not None:
            if format == "tek":
                df2tekaltimeseries(prd, filename)
            elif format == "csv":
                prd.to_csv(filename, header=False)
        return prd

    def get_gdf(self) -> gpd.GeoDataFrame:
        """Return station locations as a GeoDataFrame (EPSG:4326).

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with columns ``"id"``, ``"name"``, and ``"geometry"``.
        """
        if not self.is_read:
            self.read_data()
        if len(self.gdf) == 0:
            gdf_list = []
            for station in self.station:
                point = shapely.geometry.Point(station["lon"], station["lat"])
                d = {"id": station["id"], "name": station["name"], "geometry": point}
                gdf_list.append(d)
            self.gdf = gpd.GeoDataFrame(gdf_list, crs=4326)
        return self.gdf

    def station_names(self) -> tuple:
        """Return lists of station names and identifiers.

        Returns
        -------
        tuple of (list of str, list of str)
            ``(name_list, id_list)``.
        """
        name_list = []
        id_list = []
        for station in self.station:
            name_list.append(station["name"])
            id_list.append(station["id"])
        return name_list, id_list


class TideStationsDatabase:
    """Registry of tide station datasets.

    Reads dataset metadata from a local ``tide_stations.tml`` index file,
    optionally synchronising with an S3 bucket to discover new datasets.

    Parameters
    ----------
    path : str, optional
        Local directory containing the ``tide_stations.tml`` index file.
    s3_bucket : str, optional
        S3 bucket name for online synchronisation.
    s3_key : str, optional
        S3 key prefix under which tide station data lives.
    s3_region : str, optional
        AWS region of the S3 bucket.
    check_online : bool, optional
        If ``True``, query S3 for new datasets on construction.
    """

    def __init__(
        self,
        path: str = None,
        s3_bucket: str = None,
        s3_key: str = None,
        s3_region: str = None,
        check_online: bool = False,
    ) -> None:
        self.path = path
        self.dataset = {}
        self.s3_client = None
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.s3_region = s3_region
        self.read()
        if check_online:
            self.check_online_database()

    def read(self) -> None:
        """Read metadata for all datasets listed in ``tide_stations.tml``.

        Populates ``self.dataset`` (keyed by short name).
        """
        if self.path is None:
            print("Path to tide stations database not set !")
            return

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        tml_file = os.path.join(self.path, "tide_stations.tml")
        if not os.path.exists(tml_file):
            print(f"Warning! Tide stations database file not found: {tml_file}")
            return

        datasets = toml.load(tml_file)

        for d in datasets["dataset"]:
            name = d["name"]

            if "path" in d:
                path = d["path"]
            else:
                path = os.path.join(self.path, name)

            self.dataset[name] = TideStationsDataset(name, path)

    def check_online_database(self) -> None:
        """Synchronise the local database with the S3 bucket.

        Downloads ``tide_stations.tml`` from S3 and adds any new datasets
        (metadata only) to the local database, then re-reads the database.
        """
        if self.s3_client is None:
            self.s3_client = boto3.client(
                "s3", config=Config(signature_version=UNSIGNED)
            )
        if self.s3_bucket is None:
            return
        key = f"{self.s3_key}/tide_stations.tml"
        filename = os.path.join(self.path, "tide_stations_s3.tml")
        print("Updating tide stations database ...")
        try:
            self.s3_client.download_file(
                Bucket=self.s3_bucket,
                Key=key,
                Filename=filename,
            )
        except Exception:
            print(
                f"Failed to download {key} from {self.s3_bucket}. Database will not be updated."
            )
            return

        short_name_list, long_name_list = self.dataset_names()
        datasets_s3 = toml.load(filename)
        tide_stations_added = False
        added_names = []
        for d in datasets_s3["dataset"]:
            s3_name = d["name"]
            if s3_name not in short_name_list:
                print(f"Adding tide stations {s3_name} to local database ...")
                path = os.path.join(self.path, s3_name)
                os.makedirs(path, exist_ok=True)
                key = f"{self.s3_key}/{s3_name}/metadata.tml"
                filename = os.path.join(path, "metadata.tml")
                try:
                    self.s3_client.download_file(
                        Bucket=self.s3_bucket,
                        Key=key,
                        Filename=filename,
                    )
                except Exception as e:
                    print(e)
                    print(f"Failed to download {key}. Skipping tide stations dataset.")
                    continue
                tide_stations_added = True
                added_names.append(s3_name)
        if tide_stations_added:
            d = {}
            d["dataset"] = []
            for name in short_name_list:
                d["dataset"].append({"name": name})
            for name in added_names:
                d["dataset"].append({"name": name})
            with open(os.path.join(self.path, "tide_stations.tml"), "w") as tml:
                toml.dump(d, tml)
            self.dataset = {}
            self.read()

    def dataset_names(self) -> tuple:
        """Return lists of short and long dataset names.

        Returns
        -------
        tuple of (list of str, list of str)
            ``(short_name_list, long_name_list)``.
        """
        short_name_list = []
        long_name_list = []
        for key in self.dataset.keys():
            short_name_list.append(key)
            long_name_list.append(self.dataset[key].long_name)
        return short_name_list, long_name_list


def df2tekaltimeseries(df: pd.DataFrame, filename: str) -> None:
    """Write a time-indexed DataFrame to a Delft3D ``.tek`` file.

    Parameters
    ----------
    df : pd.DataFrame
        Time-indexed DataFrame with a single water-level column.
    filename : str
        Output file path.
    """
    nt = len(df)
    indexstr = df.index.strftime("%Y%m%d %H%M%S")
    with open(filename, "w") as f:
        f.write("* column 1 : Date\n")
        f.write("* column 2 : Time\n")
        f.write("* column 3 : WL\n")
        f.write("BL01\n")
        f.write(f"{nt} 3\n")
        j = 0
        for i, row in df.iterrows():
            tstr = indexstr[j]
            j += 1
            vstr = f"{row[0]:7.3f}"
            f.write(f"{tstr} {vstr}\n")
