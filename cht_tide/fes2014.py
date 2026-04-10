"""FES2014 tidal model reader.

Implements :class:`TideModelFes2014`, a subclass of :class:`~cht_tide.model.TideModel`
that reads tidal amplitude and phase data from the FES2014 NetCDF files stored
one file per constituent.
"""

import os

import xarray as xr

from cht_tide.model import TideModel


class TideModelFes2014(TideModel):
    """FES2014 tidal model dataset.

    Reads constituent data from a directory of NetCDF files, each named
    ``<constituent>.nc``, containing ``amplitude`` (in cm) and ``phase``
    (in degrees) variables on a regular lon/lat grid.

    Parameters
    ----------
    name : str
        Short name identifying this dataset.
    path : str
        Directory where the FES2014 NetCDF files and ``metadata.tml`` are stored.
    """

    def __init__(self, name: str, path: str) -> None:
        super().__init__()

        self.name = name
        self.path = path
        self.read_metadata()
        self.get_constituents()

    def get_constituents(self) -> None:
        """Populate ``self.constituents`` from NetCDF file names in ``self.path``."""
        filenames = os.listdir(self.path)
        self.constituents = []
        for filename in filenames:
            if filename.endswith(".nc"):
                self.constituents.append(filename.split(".")[0].upper())

    def get_data(self, xl: list, yl: list, constituents: str = "all") -> xr.Dataset:
        """Extract amplitude and phase data for a geographic bounding box.

        Parameters
        ----------
        xl : list of float
            Longitude bounds ``[lon_min, lon_max]`` in degrees east.
        yl : list of float
            Latitude bounds ``[lat_min, lat_max]`` in degrees north.
        constituents : str or list of str, optional
            Constituents to extract; ``"all"`` (default) returns all
            constituents found in the dataset.

        Returns
        -------
        xr.Dataset
            Dataset with dimensions ``constituent``, ``lat``, ``lon`` and
            variables ``amplitude`` (metres) and ``phase`` (degrees).
        """
        if constituents == "all":
            constituents = self.constituents

        if len(constituents) == 0:
            # Files were probably just downloaded from S3, so get the constituents
            self.get_constituents()

        nconst = len(constituents)

        if xl[0] < 0.0 and xl[1] < 0.0:
            xl = [xl[0] + 360.0, xl[1] + 360.0]

        ds = xr.Dataset()

        # Get dimensions from first file
        filename = os.path.join(self.path, f"{constituents[0]}.nc")
        with xr.open_dataset(filename) as data:
            ds0 = data.sel(lon=slice(xl[0], xl[1]), lat=slice(yl[0], yl[1]))
            lon = ds0.lon
            lat = ds0.lat

        # Set constituent dimension
        ds["constituent"] = constituents
        # Set lon and lat dimensions
        ds["lon"] = lon
        ds["lat"] = lat
        # Set amplitude and phase arrays
        ds["amplitude"] = xr.DataArray(
            data=nconst * [len(lat) * [len(lon) * [0.0]]],
            dims=["constituent", "lat", "lon"],
        )
        ds["phase"] = xr.DataArray(
            data=nconst * [len(lat) * [len(lon) * [0.0]]],
            dims=["constituent", "lat", "lon"],
        )

        # Loop through constituents
        for constituent in constituents:
            filename = os.path.join(self.path, f"{constituent}.nc")
            with xr.open_dataset(filename) as data:
                dsc = data.sel(lon=slice(xl[0], xl[1]), lat=slice(yl[0], yl[1]))
                # Convert amplitude from cm to metres
                ds["amplitude"].loc[constituent] = dsc["amplitude"].to_numpy() / 100.0
                ds["phase"].loc[constituent] = dsc["phase"].to_numpy()

        return ds
