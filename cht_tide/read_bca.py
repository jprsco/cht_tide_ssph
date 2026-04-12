"""Readers for SFINCS boundary condition files (.bnd and .bca).

Provides :class:`SfincsBoundary` to read SFINCS flow boundary point
locations and their tidal (astro) boundary conditions, plus helper
classes for parsing INI-style ``.bca`` files.
"""

import os

import pandas as pd


class SfincsBoundary:
    """Container for SFINCS boundary points and their tidal conditions.

    Attributes
    ----------
    flow_boundary_points : list of FlowBoundaryPoint
        Ordered list of boundary points read from the ``.bnd`` file.
    """

    def __init__(self) -> None:
        self.flow_boundary_points = []

    def read_flow_boundary_points(self, bnd_file: str) -> "SfincsBoundary":
        """Read boundary point locations from a SFINCS ``.bnd`` file.

        Parameters
        ----------
        bnd_file : str
            Path to the SFINCS boundary file (whitespace-delimited x/y
            columns, no header).

        Returns
        -------
        SfincsBoundary
            The instance (for method chaining).
        """
        if not bnd_file:
            return
        if not os.path.exists(bnd_file):
            return

        df = pd.read_csv(
            bnd_file,
            index_col=False,
            header=None,
            sep=r"\s+",
            names=["x", "y"],
        )

        for ind in range(len(df.x.to_numpy())):
            name = str(ind + 1).zfill(4)
            point = FlowBoundaryPoint(
                df.x.to_numpy()[ind], df.y.to_numpy()[ind], name=name
            )
            self.flow_boundary_points.append(point)

        return self

    def read_astro_boundary_conditions(self, bca_file: str) -> "SfincsBoundary":
        """Read tidal harmonic constituents from a SFINCS ``.bca`` file.

        Parameters
        ----------
        bca_file : str
            Path to the SFINCS astronomical boundary conditions file.

        Returns
        -------
        SfincsBoundary
            The instance (for method chaining).
        """
        if not bca_file:
            return
        if not os.path.exists(bca_file):
            return

        d = IniStruct(filename=bca_file)
        for ind, point in enumerate(self.flow_boundary_points):
            point.astro = d.section[ind].data

        return self


class Point:
    """Simple geographic point.

    Parameters
    ----------
    x : float
        X coordinate.
    y : float
        Y coordinate.
    name : str, optional
        Label for the point.
    crs : any, optional
        Coordinate reference system.
    """

    def __init__(self, x: float, y: float, name: str = None, crs=None) -> None:
        self.x = x
        self.y = y
        self.crs = crs
        self.name = name
        self.data = None


class FlowBoundaryPoint:
    """A single SFINCS flow boundary point with optional tidal data.

    Parameters
    ----------
    x : float
        X coordinate.
    y : float
        Y coordinate.
    name : str, optional
        Boundary point label.
    crs : any, optional
        Coordinate reference system.
    data : any, optional
        Time-series boundary data.
    astro : pd.DataFrame or None, optional
        Tidal harmonic data for this point.
    """

    def __init__(
        self,
        x: float,
        y: float,
        name: str = None,
        crs=None,
        data=None,
        astro=None,
    ) -> None:
        self.name = name
        self.geometry = Point(x, y, crs=crs)
        self.data = data
        self.astro = astro


class Section:
    """One section of an INI-style file.

    Attributes
    ----------
    name : str or None
        Section header name.
    keyword : list of Keyword
        Key/value pairs found in this section.
    data : pd.DataFrame or None
        Tabular data rows found in this section.
    """

    def __init__(self, name: str = None, keyword: list = [], data=None) -> None:
        self.name = None
        self.keyword = []
        self.data = None

    def get_value(self, keyword: str):
        """Return the value for a keyword (case-insensitive).

        Parameters
        ----------
        keyword : str
            Name of the keyword to look up.

        Returns
        -------
        str or None
            The value string, or ``None`` if the keyword is not present.
        """
        for kw in self.keyword:
            if kw.name.lower() == keyword.lower():
                return kw.value


class Keyword:
    """A single key/value pair from an INI-style file.

    Parameters
    ----------
    name : str, optional
        Keyword name.
    value : str, optional
        Keyword value.
    comment : str, optional
        Inline comment text.
    """

    def __init__(
        self, name: str = None, value: str = None, comment: str = None
    ) -> None:
        self.name = name
        self.value = value
        self.comment = comment


class IniStruct:
    """Parser for INI-style files used by SFINCS (``.bca``, etc.).

    Parameters
    ----------
    filename : str, optional
        Path to the file to parse immediately on construction.

    Attributes
    ----------
    section : list of Section
        Parsed sections in document order.
    """

    def __init__(self, filename: str = None) -> None:
        self.section = []

        if filename:
            self.read(filename)

    def read(self, filename: str) -> None:
        """Parse an INI-style file into sections, keywords, and data rows.

        Parameters
        ----------
        filename : str
            Path to the INI file to read.
        """
        import re

        self.section = []
        istart = []

        with open(filename, "r") as fid:
            lines = fid.readlines()

        # First go through lines and find start of sections
        for i, line in enumerate(lines):
            ll = line.strip()
            if len(ll) == 0:
                continue
            if ll[0] == "[" and ll[-1] == "]":
                section_name = ll[1:-1]
                sec = Section()
                sec.name = section_name
                istart.append(i)
                self.section.append(sec)

        # Now loop through sections
        for isec in range(len(self.section)):
            i1 = istart[isec] + 1
            if isec == len(self.section) - 1:
                i2 = len(lines)
            else:
                i2 = istart[isec + 1] - 1

            df = pd.DataFrame()

            for iline in range(i1, i2):
                ll = lines[iline].strip()

                if len(ll) == 0:
                    continue

                if ll[0] == "#":
                    continue

                if "=" in ll:
                    key = Keyword()

                    # Handle embedded # comment characters
                    if "#" in ll:
                        ipos = [(i.start()) for i in re.finditer("#", ll)]
                        if len(ipos) > 1:
                            ll = (
                                ll[0 : ipos[0]]
                                + ll[ipos[0] + 1 : ipos[1]]
                                + ll[ipos[1] + 1 :]
                            )

                    if "#" in ll:
                        j = ll.index("#")
                        key.comment = ll[j + 1 :].strip()
                        ll = ll[0:j].strip()

                    tx = ll.split("=")
                    key.name = tx[0].strip()
                    key.value = tx[1].strip()

                    self.section[isec].keyword.append(key)

                else:
                    a_list = ll.split()
                    list_of_floats = []
                    for item in a_list:
                        try:
                            list_of_floats.append(float(item))
                        except Exception:
                            list_of_floats.append(item)
                    a_series = pd.Series(list_of_floats)
                    df = pd.concat([df, a_series], axis=1)

            if not df.empty:
                df = df.transpose()
                df = df.set_index([0])
                self.section[isec].data = df
